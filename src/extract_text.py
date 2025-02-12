import pdfplumber
import os

# PDFフォルダのパス
pdf_dir = "data"
output_file = "data/extracted_text.txt"

# PDF からテキストを抽出
all_text = "" 

for pdf_file in os.listdir(pdf_dir):
    if pdf_file.endswith(".pdf"):
        print(f"🔍 {pdf_file} からテキストを抽出中...")
        with pdfplumber.open(os.path.join(pdf_dir, pdf_file)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

# テキストを保存
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print("✅ PDF のテキスト抽出が完了しました！")
