from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 文章埋め込みモデルの準備（次元数を変更するために異なるモデルを選択）
embedding_model = HuggingFaceEmbeddings(model_name="roberta-base")

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