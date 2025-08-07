import streamlit as st
from dotenv import load_dotenv
import os
from dynamic_rag_module import load_model, dynamic_rag_pipeline, generate_clean_with_rag

# Load HF token from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Load model + tokenizer once
@st.cache_resource
def get_model():
    return load_model()  # You should return tokenizer, model

tokenizer, model = get_model()

st.title("🩺 MedLLaMA with Dynamic RAG")
st.markdown("Ask medical questions — powered by a fine-tuned LLaMA2 and real-time web retrieval.")

user_question = st.text_input("💬 Your question", "")

if user_question:
    with st.spinner("🔎 Searching and answering..."):
        # Step 1: RAG search
        context_chunks = dynamic_rag_pipeline(user_question)  # Returns top-k retrieved texts

        # Step 2: Combine prompt + context + generate response
        full_answer = generate_clean_with_rag(user_question, context_chunks, tokenizer, model)

    st.markdown("### 🧠 Answer")
    st.success(full_answer)

    st.markdown("### 📚 Retrieved Context")
    for i, chunk in enumerate(context_chunks):
        st.markdown(f"**[{i+1}]** {chunk}")
