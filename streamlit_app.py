import os, torch, streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv
load_dotenv()

HF_REPO = "Nazmoose/MedLlama-LoRA"
BASE_REPO = "NousResearch/Llama-2-7b-hf"

@st.cache_resource
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_REPO, token=os.getenv("HF_TOKEN"))
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_REPO,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base_model, HF_REPO, token=os.getenv("HF_TOKEN"))
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

SYSTEM_PROMPT = (
    "You are MedLLaMA, a model fine-tuned for clinical Q&A. "
    "Respond with medically relevant answers but do not provide professional advice."
)

st.set_page_config(page_title="MedLLaMA", page_icon="🩺")
st.title("🩺 MedLLaMA – Clinical Q&A Assistant")
st.markdown("> ⚠️ This app is for educational/demo purposes only.")

if "history" not in st.session_state:
    st.session_state.history = []

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

user_msg = st.chat_input("Ask a medical question...")
if user_msg:
    st.chat_message("user").markdown(user_msg)
    st.session_state.history.append(("user", user_msg))

    prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_msg} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    st.chat_message("assistant").markdown(answer)
    st.session_state.history.append(("assistant", answer))
