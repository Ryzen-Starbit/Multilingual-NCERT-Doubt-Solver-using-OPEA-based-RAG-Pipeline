import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
def extract_ocr_text(path: str) -> str:
    text = ""
    if path.lower().endswith(".pdf"):
        pages = convert_from_path(path)
        for page in pages:
            text += pytesseract.image_to_string(page, lang="eng+hin+urd")
    else:
        img = Image.open(path)
        text = pytesseract.image_to_string(img, lang="eng+hin+urd")
    return text
