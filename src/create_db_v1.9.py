from langchain.vectorstores import FAISS #類似検索（最近傍探索）を高速に行うベクトルデータベース
from langchain.embeddings import HuggingFaceEmbeddings #自然言語を数値（ベクトル）に変換する埋め込みモデル
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 文章埋め込みモデルの準備    "sentence-transformers/all-MiniLM-L6-v2" は 高精度かつ軽量な埋め込みモデル
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 抽出したテキストをロード
with open("data/extracted_text.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

# テキストを分割（検索しやすくするため）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
documents = text_splitter.split_text(full_text)

# ベクトルデータベースを作成
vector_db = FAISS.from_texts(documents, embedding_model)
vector_db.save_local("data/faiss_index")

print("✅ ベクトルデータベースが作成されました！")
