# 🩺 Medical Assistant LLM (MedLLaMA)

MedLLaMA is a fine-tuned version of LLaMA-2 designed for clinical instruction following. It was trained on the [Llama2-MedTuned-Instructions](https://huggingface.co/datasets/nlpie/Llama2-MedTuned-Instructions) dataset (200,000 samples) using LoRA and 4-bit quantization (QLoRA). The model specializes in tasks like medical question answering, named entity recognition (NER), and relation extraction.

---

## 💡 Features

- ✅ Fine-tuned LLaMA-2–7B using LoRA + 4-bit QLoRA
- ✅ Instruction-style prompting (`[INST] ... [/INST]`) for clinical NLP tasks
- ✅ GPT-4-based evaluation showed a **+2.1 improvement in accuracy** and **+1.8 in helpfulness** over the base model across 30 prompts
- ✅ Streamlit app for real-time interaction

---

## 🚀 Try It Locally

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
2. Create a .env file with your Hugging Face token:
   HF_TOKEN=your_token_here
3. Run the app:
   streamlit run streamlit_app.py

## 📊 Evaluation Methodology
Sampled 30 prompts from the validation split of the MedTuned dataset

Compared base LLaMA-2 and MedLLaMA outputs side-by-side

GPT-4 scored each pair on accuracy, completeness, and helpfulness

MedLLaMA consistently outperformed the base model

### 📎 License & Disclaimer
This project is for research and educational purposes only.
The model is not a substitute for professional medical advice.
