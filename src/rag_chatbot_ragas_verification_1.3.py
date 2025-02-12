# 逐次処理のバージョン,結果の保存を省略 これで行きます！


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from datasets import Dataset
import os
from dotenv import load_dotenv
import pandas as pd
import time
import numpy as np

# API設定
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv("apikey.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("APIキーが設定されていません。apikey.env を確認してください。")

def load_questions_from_excel(file_path):
    try:
        print(f"Excelファイルを読み込み中: {file_path}")
        df = pd.read_excel(file_path)
        questions_and_answers = df.to_dict('records')
        print(f"-> {len(questions_and_answers)}件の質問を読み込みました")
        return questions_and_answers
    except Exception as e:
        print(f"Excelファイルの読み込み中にエラーが発生しました: {str(e)}")
        raise

def select_question_file():
    print("使用する質問データファイルを選択してください:")
    print("1. questions_validation.xlsx")
    print("2. questions_test.xlsx")
    print("3. questions_train.xlsx")
    choice = input("選択肢 (1/2/3): ")

    if choice == '1':
        return 'data/questions_validation.xlsx'
    elif choice == '2':
        return 'data/questions_test.xlsx'
    elif choice == '3':
        return 'data/questions_train.xlsx'
    else:
        print("無効な選択です。デフォルトのquestions_validation.xlsxを使用します。")
        return 'data/questions_validation.xlsx'

# メインの処理
try:
    print("\n=== 評価プロセスを開始します ===")
    
    # LLMとベクトルDBの準備
    llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.load_local("data/faiss_index", embedding_model, allow_dangerous_deserialization=True)
    
    # RAGチェーンの構築
    retriever = vector_db.as_retriever()
    rag_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
    
    # 質問データファイルの選択
    question_file = select_question_file()
    
    print("\n1. 質問データの読み込み中...")
    questions_and_answers = load_questions_from_excel(question_file)
    evaluation_data = []

    print("\n2. 質問の処理を開始します...")
    for i, qa in enumerate(questions_and_answers, 1):
        question = qa["question"]
        expected_answer = qa["expected_answer"]

        print(f"\n処理中 ({i}/{len(questions_and_answers)}): {question[:30]}...")

        try:
            # レートリミット対策
            time.sleep(0.5)

            # 検索と生成
            retrieved_docs = retriever.invoke(question)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]
            response = rag_chain.invoke({"query": question})
            generated_answer = response["result"]

            evaluation_data.append({
                "question": question,
                "answer": generated_answer,
                "contexts": retrieved_texts,
                "ground_truths": [expected_answer],
                "reference": expected_answer
            })
            print(f"  ✓ 処理完了")

        except Exception as e:
            print(f"  ✗ エラーが発生しました: {str(e)}")
            if "rate_limit" in str(e).lower():
                print("  > レートリミットに達しました。60秒待機します...")
                time.sleep(60)
            continue

    print("\n3. RAGAS評価を実行中...")
    df = pd.DataFrame(evaluation_data)
    dataset = Dataset.from_pandas(df)
    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, context_precision, context_recall, faithfulness]
    )

    print("\n=== 評価結果 ===")
    metrics = {
        'Answer Relevancy': results['answer_relevancy'],
        'Context Precision': results['context_precision'],
        'Context Recall': results['context_recall'],
        'Faithfulness': results['faithfulness']
    }
    
    # 結果を整形して出力
    print("評価結果:")
    for metric_name, values in metrics.items():
        avg_score = np.mean(values)
        print(f"  {metric_name}: {avg_score:.4f}")

    print("\n=== 評価プロセスが完了しました ===\n")

except Exception as e:
    print(f"\n✗ プログラムの実行中にエラーが発生しました: {str(e)}")