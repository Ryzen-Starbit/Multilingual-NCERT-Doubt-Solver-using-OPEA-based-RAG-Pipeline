from langdetect import detect, LangDetectException
def adapt_language(text: str):
    if not text or len(text.strip()) < 3:
        return "en"
    try:
        return detect(text)
    except LangDetectException:
        return "en"
