import os
import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
DATA_ROOT = "data/NCERT"
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)
def chunk_text(text):
    return splitter.split_text(text)
def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text
def extract_ocr_text(path):
    text = ""
    pages = convert_from_path(path)
    for page in pages:
        text += pytesseract.image_to_string(page, lang="eng+hin+urd")
    return text
def extract_chapter_title(text, subject):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if subject in ["hindi", "urdu"]:
        for l in lines[:20]:
            if 3 <= len(l) <= 40 and not re.search(r"\d", l):
                return l
    for l in lines[:20]:
        if l.isupper() and len(l) <= 50:
            return l.title()
    return None
docs = []
for class_dir in os.listdir(DATA_ROOT):
    class_path = os.path.join(DATA_ROOT, class_dir)
    if not os.path.isdir(class_path):
        continue
    grade = re.search(r"\d+", class_dir).group()
    for subject_dir in os.listdir(class_path):
        subject_path = os.path.join(class_path, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        subject = subject_dir.lower().replace(" ", "").replace("_", "")
        for item in os.listdir(subject_path):
            item_path = os.path.join(subject_path, item)
            if item.lower().endswith(".pdf"):
                book = "default"
                file_path = item_path
            elif os.path.isdir(item_path):
                book = item.lower().replace(" ", "_")
                for file in os.listdir(item_path):
                    if not file.lower().endswith(".pdf"):
                        continue
                    file_path = os.path.join(item_path, file)
            else:
                continue
            chapter_id_match = re.search(r"(chapter\s*\d+)", file_path.lower())
            chapter_id = chapter_id_match.group(1) if chapter_id_match else os.path.basename(file_path)
            try:
                text = extract_pdf_text(file_path)
                if len(text.strip()) < 100:
                    text = extract_ocr_text(file_path)
            except:
                continue
            chapter_title = extract_chapter_title(text, subject) or chapter_id
            chapter = f"{chapter_id}: {chapter_title}"
            for chunk in chunk_text(text):
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "board": "NCERT",
                        "grade": grade,
                        "subject": subject,
                        "book": book,
                        "chapter": chapter,
                        "source": os.path.basename(file_path)
                    }
                ))
print(f"✅ Total chunks created: {len(docs)}")
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-small"
)
db = FAISS.from_documents(docs, embeddings)
db.save_local("vectorstore/faiss_index")
print("✅ Vector DB saved successfully")
