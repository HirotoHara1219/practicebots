from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI  # OpenAI から ChatOpenAI に変更
from langchain.chains import ConversationalRetrievalChain

import os
from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# .envファイルをロード
load_dotenv("apikey.env")

# OpenAI APIキーを取得
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("APIキーが設定されていません。apikey.env を確認してください。")

# LLM（GPT-4）を準備
llm = ChatOpenAI(model="gpt-4", api_key=OPENAI_API_KEY)  # ChatOpenAI でチャットモデルを使用

# ベクトルデータベースをロード
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("data/faiss_index", embedding_model, allow_dangerous_deserialization=True)

# RAG 検索　　　　retriever（リトリーバー）は、FAISS のデータを検索するためのオブジェクトです。
retriever = vector_db.as_retriever()
rag_chain = ConversationalRetrievalChain.from_llm(llm, retriever)


# **💬 ユーザー入力を受け付ける**　
# rag_chainを利用する
# retrieval_chain は「検索（retriever）+ 生成（GPT-4）」を組み合わせた AI チェーン（RAG）です。

# ✅ 会話履歴を初期化
chat_history = []

# **💬 ユーザー入力を受け付ける**
print("ChatGPT × FAISS の RAG チャットボットへようこそ！\n質問を入力してください（終了するには 'exit' と入力）")


while True:
    question = input("\n💬 質問を入力してください（終了: exit）: ")
    
    # 「exit」でチャットを終了
    if question.lower() == "exit":
        print("🛑 チャットを終了します。")
        break

    try:
        # ✅ chat_history を含めて入力を渡す
        response = rag_chain.invoke({"question": question, "chat_history": chat_history})
        
        # ✅ 回答を表示
        print("\n🤖 回答:", response["answer"])

        # ✅ 会話履歴を更新（ユーザーの質問とAIの回答を保存）
        chat_history.append((question, response["answer"]))

    except Exception as e:
        print("\n❌ エラー:", str(e))