#並列処理を使用したバージョン

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
import csv
from datetime import datetime
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor #増えた
from functools import partial # 増えた
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
        
        if 'question' not in df.columns or 'expected_answer' not in df.columns:
            print("警告: カラム名を確認してください。'question'と'expected_answer'が必要です。")
            return []
        
        questions_and_answers = df.to_dict('records')
        print(f"-> {len(questions_and_answers)}件の質問を読み込みました")
        return questions_and_answers
        
    except Exception as e:
        print(f"Excelファイルの読み込み中にエラーが発生しました: {str(e)}")
        raise

def process_single_question(qa, retriever, rag_chain):
    question = qa["question"]
    expected_answer = qa["expected_answer"]
    
    try:
        time.sleep(0.5)  # レートリミット対策
        retrieved_docs = retriever.invoke(question)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        
        # ここで入力を正しい形式に変換
        response = rag_chain.invoke(question)  # 直接文字列を渡す

        # AIMessageオブジェクトから結果を取得
        generated_answer = response.content if hasattr(response, 'content') else response["result"]

        return {
            "question": question,
            "answer": generated_answer,
            "contexts": retrieved_texts,
            "ground_truths": [expected_answer],
            "reference": expected_answer
        }
        
    except Exception as e:
        print(f"質問の処理中にエラー: {question[:30]}... - {str(e)}")
        if "rate_limit" in str(e).lower():
            print("レートリミットに達しました。60秒待機します...")
            time.sleep(60)
        return None

def process_questions(questions, context_builder, llm):
    results = []
    completed_count = 0

    # 並列処理のためのスレッドプールを作成
    with ThreadPoolExecutor(max_workers=5) as executor:
        # process_single_questionを並列で実行
        futures = [executor.submit(process_single_question, qa, context_builder, llm) for qa in questions]

        for future in futures:
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    completed_count += 1
                    print("  ✓ 完了")
            except Exception as e:
                print(f"  ✗ エラー: {str(e)}")

    print(f"-> {completed_count}件の処理が完了しました")
    return results

def save_evaluation_results(results, evaluation_data, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        details_file = output_path / f"evaluation_details_{timestamp}.csv"
        with open(details_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                '質問', '生成された回答', '期待される回答', 
                'Answer Relevancy', 'Context Precision', 
                'Context Recall', 'Faithfulness',
                '検索されたコンテキスト'
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
                    '\n'.join(data.get('contexts', []))
                ])
        
        summary_file = output_path / f"evaluation_summary_{timestamp}.csv"
        with open(summary_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['メトリクス', '平均スコア'])
            
            metrics = {
                'Answer Relevancy': results['answer_relevancy'] if 'answer_relevancy' in results else [],
                'Context Precision': results['context_precision'] if 'context_precision' in results else [],
                'Context Recall': results['context_recall'] if 'context_recall' in results else [],
                'Faithfulness': results['faithfulness'] if 'faithfulness' in results else []
            }
            
            for metric_name, values in metrics.items():
                if values:
                    avg_score = sum(values) / len(values)
                    writer.writerow([metric_name, f"{avg_score:.3f}"])
        
        print(f"評価結果を保存しました: {output_dir}")
        
    except Exception as e:
        print(f"評価結果の保存中にエラーが発生しました: {str(e)}")
        raise

def main():
    try:
        print("\n=== 評価プロセスを開始します ===")
        
        # LLMとベクトルDBの準備
        llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.load_local("data/faiss_index", embedding_model, allow_dangerous_deserialization=True)
        
        retriever = vector_db.as_retriever()
        rag_chain = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)
        
        print("\n1. 質問データの読み込み中...")
        questions_and_answers = load_questions_from_excel('data/questions.xlsx')

        print("\n2. 質問の処理を開始します...")
        evaluation_data = process_questions(questions_and_answers, retriever, llm)
        print(f"-> {len(evaluation_data)}件の処理が完了しました")

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

        print("\n4. 評価結果を保存中...")
        save_evaluation_results(results, evaluation_data, 'data/evaluation')

    except Exception as e:
        print(f"\n✗ プログラムの実行中にエラーが発生しました: {str(e)}")

if __name__ == "__main__":
    main()
