from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.document import Document
import time

# Dynamic RAG: Web Search + Scrape + Chunk + ChromaDB

# Step 1: Search the web
def search_web(query, max_results=3):
    ddgs = DDGS()
    results = []

    # get the titles and links to the websites and store them in results
    for r in ddgs.text(query, max_results=max_results):
        results.append({"title": r["title"], "href": r["href"]})
    return results

# Step 2: Scrape the content from the URLs
def scrape_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        # Extract paragraphs only
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text.strip()
    except Exception as e:
        print(f"[WARN] Failed to scrape {url}: {e}")
        return ""

# Step 3: Chunk the scraped text into LangChain Documents
def chunk_text(text, source):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text], metadatas=[{"source": source}])
    return chunks

# Step 4: Embed and store in Chroma
def build_chroma_db(docs, persist_dir="chroma_db"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_dir)
    return db

# === FULL PIPELINE ===
def dynamic_rag_pipeline(query):
    print(f"🔎 Searching web for: {query}")
    results = search_web(query)

    all_docs = []
    for r in results:
        print(f"🌐 Scraping: {r['href']}")
        text = scrape_content(r['href'])
        if text:
            chunks = chunk_text(text, r["href"])
            all_docs.extend(chunks)
        time.sleep(1.5)  # avoid rate-limiting

    if not all_docs:
        print("❌ No content could be retrieved.")
        return None

    print(f"✅ Retrieved {len(all_docs)} chunks. Storing in vector DB...")
    vectordb = build_chroma_db(all_docs)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return retriever


# Loading the fine tuned model
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HF_REPO = "Nazmoose/MedLlama-LoRA"
BASE_REPO = "NousResearch/Llama-2-7b-hf"

def load_model(HF_TOKEN):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer from your LoRA adapter repo (inherits from base)
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, token=HF_TOKEN)  # set `token=True` for Colab auth

    # Load base LLaMA model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN
    )

    # Load LoRA adapters into base model
    model = PeftModel.from_pretrained(base_model, HF_REPO, token=HF_TOKEN)
    model.eval()
    return tokenizer, model


# Functions to generate a clean response
def clean_response(raw_text: str) -> str:
    """
    Extracts only the first ###Answer: section from the model output.
    """
    if "###Answer:" in raw_text:
        answer_section = raw_text.split("###Answer:")[1].strip()
        # Optionally stop at next metadata marker
        for stop_token in ["###Rationale", "###Source", "###Tags", "###Used in", "###Context"]:
            if stop_token in answer_section:
                answer_section = answer_section.split(stop_token)[0].strip()
                break
        return answer_section
    else:
        return raw_text.strip()

def generate_clean_with_rag(user_msg, retriever, tokenizer, model):
    SYSTEM_PROMPT = (
        "You are MedLLaMA, a model fine-tuned for clinical Q&A. "
        "Respond with medically relevant answers but do not provide professional advice. "
        "Use the provided context to answer accurately."
    )

    # === RAG integration ===
    retrieved_docs = retriever.get_relevant_documents(user_msg)
    rag_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

Context:
{rag_context}

Question:
{user_msg}
[/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    print(clean_response(response.strip()))
