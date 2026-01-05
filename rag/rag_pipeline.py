import time
import torch
from collections import defaultdict
from transformers import AutoTokenizer,AutoModelForCausalLM,pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from opea.evaluator import explain
from opea.adapter import adapt_language
translator=None
def get_translator():
    global translator
    if translator is None:
        translator=pipeline("translation",model="Helsinki-NLP/opus-mt-hi-en")
    return translator
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
def extract_answer(context):
    for sep in ["।",".","\n"]:
        parts=context.split(sep)
        for p in parts:
            if len(p.strip())>20:
                return p.strip()+("।" if sep=="।" else "")
    return "Not found in NCERT."
def ask_question(question,grade,subject,book=None):
    start=time.time()
    lang=adapt_language(question)
    subject=SUBJECT_NORMALIZATION.get(subject.lower(),subject.lower())
    if subject=="hindi" and lang!="hi":
        return not_found(start)
    if subject=="urdu" and lang!="ur":
        return not_found(start)
    search_queries=[question]
    if subject in ["science","socialscience","maths","english"] and lang in ["hi","ur"]:
        search_queries.insert(0,get_translator()(question,max_length=128)[0]["translation_text"])
    docs=[]
    for q in search_queries:
        docs.extend(db.similarity_search(q,k=12,filter={"grade":str(grade),"subject":subject}))
    docs=list({d.page_content:d for d in docs}.values())
    if not docs:
        return not_found(start)
    chapter_map=defaultdict(list)
    for d in docs:
        chapter_map[d.metadata["chapter"]].append(d)
    best_docs=max(chapter_map.values(),key=len)[:2]
    context="\n".join(d.page_content for d in best_docs)
    context=tokenizer.decode(tokenizer(context,truncation=True,max_length=450,return_tensors="pt")["input_ids"][0],skip_special_tokens=True)
    citation=explain(best_docs)
    if subject in ["hindi","urdu"]:
        return {"answer":extract_answer(context),"citation":citation,"latency":round(time.time()-start,2)}
    prompt=f"""You are an NCERT textbook answer extractor.
Answer ONLY from the given text.
No explanation.
1–2 sentences.
If missing, reply exactly: Not found in NCERT.
Text:
{context}
Question:
{question}
Answer:"""
    inputs=tokenizer(prompt,return_tensors="pt",truncation=True,max_length=720)
    with torch.no_grad():
        output=model.generate(**inputs,max_new_tokens=70,do_sample=False,temperature=0.0)
    answer=tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    if len(answer.split())<3:
        answer="Not found in NCERT."
    return {"answer":answer,"citation":citation,"latency":round(time.time()-start,2)}
