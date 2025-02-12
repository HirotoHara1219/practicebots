from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from datasets import Dataset  # datasetsライブラリを使用

import os
from dotenv import load_dotenv
import pandas as pd
import time
import csv
from datetime import datetime
from pathlib import Path


#① API系
# TOKENIZERSのパラレルオプションを無効化
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# .envファイルをロード
load_dotenv("apikey.env")
# OpenAI APIキーを取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("APIキーが設定されていません。apikey.env を確認してください。")


#② LLMを準備、ベクトルデータベースをロード
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("data/faiss_index", embedding_model, allow_dangerous_deserialization=True)

#③ RAGチェーンの構築（検索オブジェクトを作成）
retriever = vector_db.as_retriever()
rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

#④ ユーザーからの入力
# 評価用のデータリスト
evaluation_data = []
print("\nRAG システムの評価を行います！")
print("質問と期待される回答を入力してください（終了: exit）\n") 
# ユーザーからの入力を受け取り
questions_and_answers = []
while True:
    question = input("\n質問: ")
    if question.lower() == "exit":
        break
    expected_answer = input("期待される回答: ")
    questions_and_answers.append({"question": question, "expected_answer": expected_answer})

def save_evaluation_results(results, evaluation_data, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 詳細な評価結果をCSVに保存
        details_file = output_path / f"evaluation_details_{timestamp}.csv"
        with open(details_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                '質問', '生成された回答', '期待される回答', 
                'Answer Relevancy', 'Context Precision', 
                'Context Recall', 'Faithfulness',
                '検索されたコンテキスト', '処理時間(秒)'
            ])
            
            for i, data in enumerate(evaluation_data):
                writer.writerow([
                    data['question'],
                    data['answer'],
                    data['ground_truths'][0],
                    results['answer_relevancy'][i] if 'answer_relevancy' in results else 'N/A',
                    results['context_precision'][i] if 'context_precision' in results else 'N/A',
                    results['context_recall'][i] if 'context_recall' in results else 'N/A',
                    results['faithfulness'][i] if 'faithfulness' in results else 'N/A',
                    '\n'.join(data.get('contexts', [])),
                    data.get('processing_time', 'N/A')
                ])
        
        # 集計結果をCSVに保存
        summary_file = output_path / f"evaluation_summary_{timestamp}.csv"
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['メトリクス', '平均スコア'])
            # resultsオブジェクトの構造に応じて適切に処理
            metrics = {
                'answer_relevancy': results.get('answer_relevancy', []),
                'context_precision': results.get('context_precision', []),
                'context_recall': results.get('context_recall', []),
                'faithfulness': results.get('faithfulness', [])
            }
            for metric, scores in metrics.items():
                if scores:
                    writer.writerow([metric, sum(scores)/len(scores)])
        
        print(f"評価結果を保存しました: {output_dir}")
        
    except Exception as e:
        print(f"評価結果の保存中にエラーが発生しました: {str(e)}")
        raise

def load_questions_from_excel(file_path):
    try:
        print(f"Excelファイルを読み込み中: {file_path}")
        # Excelファイルを読み込み
        df = pd.read_excel(file_path)
        
        # カラム名の確認と変換
        if 'question' not in df.columns or 'expected_answer' not in df.columns:
            print("警告: カラム名を確認してください。'question'と'expected_answer'が必要です。")
            return []
        
        # DataFrameを辞書のリストに変換
        questions_and_answers = df.to_dict('records')
        print(f"-> {len(questions_and_answers)}件の質問を読み込みました")
        return questions_and_answers
        
    except Exception as e:
        print(f"Excelファイルの読み込み中にエラーが発生しました: {str(e)}")
        raise

# メインの処理
try:
    print("\n=== 評価プロセスを開始します ===")
    
    print("\n1. 質問データの読み込み中...")
    # CSVの代わりにExcelファイルを読み込む
    questions_and_answers = load_questions_from_excel('data/questions.xlsx')
    print(f"-> {len(questions_and_answers)}件の質問を読み込みました")
    
    evaluation_data = []

    print("\n2. 質問の処理を開始します...")
    for i, qa in enumerate(questions_and_answers, 1):
        start_time = time.time()
        
        question = qa["question"]
        expected_answer = qa["expected_answer"]

        print(f"\n処理中 ({i}/{len(questions_and_answers)}): {question[:30]}...")

        try:
            # レートリミットに引っかかった場合は少し待機
            time.sleep(0.5)  # 0.5秒待機

            # 検索フェーズ
            print("  - コンテキスト検索中...")
            retrieved_docs = retriever.invoke(question)
            retrieved_texts = [doc.page_content for doc in retrieved_docs]

            # 生成フェーズ
            print("  - 回答生成中...")
            response = rag_chain.invoke({"question": question, "chat_history": []})
            generated_answer = response["answer"]

            processing_time = time.time() - start_time

            # 評価用データを辞書形式で保存
            evaluation_data.append({
                "question": question,
                "answer": generated_answer,
                "contexts": retrieved_texts,
                "ground_truths": [expected_answer],
                "reference": expected_answer,
                "processing_time": round(processing_time, 2)
            })
            print(f"  ✓ 処理完了 ({round(processing_time, 2)}秒)")

        except Exception as e:
            print(f"  ✗ エラーが発生しました: {str(e)}")
            if "rate_limit" in str(e).lower():
                print("  > レートリミットに達しました。60秒待機します...")
                time.sleep(60)  # レートリミットの場合は1分待機
            continue

    print("\n3. RAGAS評価を実行中...")
    df = pd.DataFrame(evaluation_data)
    dataset = Dataset.from_pandas(df)
    results = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, context_precision, context_recall, faithfulness]
    )

    print("\n4. 評価結果を保存中...")
    save_evaluation_results(results, evaluation_data, 'data/evaluation')
    print("\n=== 評価プロセスが完了しました ===\n")

except Exception as e:
    print(f"\n✗ プログラムの実行中にエラーが発生しました: {str(e)}")

#⑥ 評価データセットの作成とRAGAS評価
# Pandas DataFrame に変換し、datasets.Dataset に変換
df = pd.DataFrame(evaluation_data)
dataset = Dataset.from_pandas(df)

# RAGAS で評価
results = evaluate(
    dataset=dataset,
    metrics=[answer_relevancy, context_precision, context_recall, faithfulness]
)

# 結果の表示
print("\nRAGAS 評価結果:")
print(results)