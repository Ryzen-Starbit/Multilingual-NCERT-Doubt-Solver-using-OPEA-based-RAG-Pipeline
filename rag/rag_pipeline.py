import time
import torch
import re
from collections import defaultdict
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from opea.evaluator import explain
from opea.adapter import adapt_language
translator_hi=None
translator_ur=None
translator_en=None
def get_hi_translator():
    global translator_hi
    if translator_hi is None:
        translator_hi=pipeline("translation",model="Helsinki-NLP/opus-mt-en-hi")
    return translator_hi
def get_ur_translator():
    global translator_ur
    if translator_ur is None:
        translator_ur=pipeline("translation",model="Helsinki-NLP/opus-mt-en-ur")
    return translator_ur
def get_en_translator():
    global translator_en
    if translator_en is None:
        translator_en=pipeline("translation",model="Helsinki-NLP/opus-mt-hi-en")
    return translator_en
SUBJECT_NORMALIZATION={
    "science":"science",
    "math":"maths",
    "maths":"maths",
    "english":"english",
    "hindi":"hindi",
    "urdu":"urdu",
    "social science":"socialscience",
    "socialscience":"socialscience"
}
embeddings=HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
db=FAISS.load_local("vectorstore/faiss_index",embeddings,allow_dangerous_deserialization=True)
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map="cpu",torch_dtype=torch.float32)
model.eval()
def not_found(start):
    return {"answer":"Not found in NCERT.","citation":None,"latency":round(time.time()-start,2)}
def extract_native_answer(context,question):
    words=re.findall(r'[\u0900-\u097F\u0600-\u06FF]+',question)
    lines=[l.strip() for l in context.split("\n") if len(l.strip())>15]
    if not lines:
        return "Not found in NCERT."
    def score(l):
        return sum(1 for w in words if w in l)*5+len(l)*0.1
    best=max(lines,key=score)
    best=re.sub(r'[^\u0900-\u097F\u0600-\u06FF\s।,.?!]',"",best)
    if len(best)<10:
        return "Not found in NCERT."
    return best[:200]
def extract_english_answer(context,question):
    q_words=re.findall(r'[a-zA-Z]+',question.lower())
    lines=[l.strip() for l in context.split("\n") if len(l.strip())>20]
    if not lines:
        return "Not found in NCERT."
    def score(l):
        return sum(1 for w in q_words if w in l.lower())*4+len(l)*0.1
    best=max(lines,key=score)
    best=re.sub(r'[^a-zA-Z0-9\s.,?!]',"",best)
    if len(best)<20:
        return "Not found in NCERT."
    return best[:220]
def ask_question(question,grade,subject,book=None):
    start=time.time()
    lang=adapt_language(question)
    subject=SUBJECT_NORMALIZATION.get(subject.lower(),subject.lower())
    if subject=="hindi" and lang!="hi":
        return not_found(start)
    if subject=="urdu" and lang!="ur":
        return not_found(start)
    cross_lang=(subject in ["science","socialscience","maths","english"] and lang in ["hi","ur"])
    search_queries=[question]
    if cross_lang:
        try:
            q_en=get_en_translator()(question,max_length=128)[0]["translation_text"]
            search_queries.insert(0,q_en)
        except:
            pass
    docs=[]
    for q in search_queries:
        docs.extend(db.similarity_search(q,k=12,filter={"grade":str(grade),"subject":subject}))
    docs=list({d.page_content:d for d in docs}.values())
    if not docs:
        return not_found(start)
    chapter_map=defaultdict(list)
    for d in docs:
        chapter_map[d.metadata["chapter"]].append(d)
    best_docs=max(chapter_map.values(),key=len)[:3]
    context="\n".join(d.page_content for d in best_docs)[:2500]
    citation=explain(best_docs)
    if subject in ["hindi","urdu"]:
        answer=extract_native_answer(context,question)
        return {"answer":answer,"citation":citation,"latency":round(time.time()-start,2)}
    if cross_lang:
        english_answer=extract_english_answer(context,q_en if "q_en" in locals() else question)
        if english_answer=="Not found in NCERT.":
            return not_found(start)
        try:
            if lang=="hi":
                english_answer=get_hi_translator()(english_answer,max_length=128)[0]["translation_text"]
            else:
                english_answer=get_ur_translator()(english_answer,max_length=128)[0]["translation_text"]
        except:
            pass
        return {"answer":english_answer,"citation":citation,"latency":round(time.time()-start,2)}
    prompt=f"""You are an NCERT textbook answer extractor.
STRICT RULES:
- Use ONLY the given NCERT text.
- Answer in ENGLISH ONLY.
- 1–2 complete sentences.
- No explanation.
- If missing, reply exactly: Not found in NCERT.
Text:
{context}
Question:
{question}
Answer:"""
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=1024)
    with torch.no_grad():
        output=model.generate(**inputs,do_sample=False,temperature=0.0,max_new_tokens=80)
    answer=tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    if len(answer.split())<4 or "not found" in answer.lower():
        answer="Not found in NCERT."
    return {"answer":answer,"citation":citation,"latency":round(time.time()-start,2)}
