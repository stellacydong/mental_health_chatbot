---
title: Mental Health Counselor Assistant
emoji: 🧠
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: 1.44.1
app_file: app.py
pinned: false
license: mit
---
# 🧠 Mental Health Counselor Assistant

**Mental Health Counselor Assistant** is an AI-powered Streamlit application hosted on [Hugging Face Spaces](https://huggingface.co/spaces/scdong/mental_health_chatbot). It assists mental health professionals by classifying user inputs and generating supportive counselor-style responses using large language models (LLMs) like **Flan-T5** and **Mistral-7B**.

---

## ⚡ Quick Start

The app uses `google/flan-t5-base` for fast, low-latency response generation.

> 📝 `app.py` is optimized for **speed**, using only `/flan-t5-base`.  
> While the results may not be as expressive as other models, it loads and responds much faster.  
> For higher quality but slower results, try the other two apps included.

---

## 📦 Project Structure

```
mental_health_chatbot/
├── app.py                                # Fast demo with Flan-T5 only (used on Hugging Face)
├── app_use_Mistral-7B.py                 # Local LLM with quantized Mistral-7B via llama.cpp
├── app_with_FlanT5_FlanAlpacaGPT4_FlanUL2.py  # Hosted Hugging Face models (slower, richer output)
├── requirements.txt
├── Dockerfile
├── README.md
├── LICENSE

├── dataset/
│   └── Kaggle_Mental_Health_Conversations_train.csv

├── utils/
│   └── helper_functions.py

├── notebooks/
│   ├── Flan-T5_on_Kaggle_Dataset.ipynb
│   ├── Mistral-7B_on_Kaggle_Dataset.ipynb
│   └── ML_on_Kaggle_Dataset.ipynb

└── log/
    └── chat_log_2025-04-11_05-31-25.csv
```

---

## 💡 Features

- 🔍 **Intent Classification** (XGBoost):  
  Tags inputs as `advice`, `validation`, `information`, or `question`.

- 🤖 **LLM-Based Suggestion Generation**  
  Choose from:
  - `google/flan-t5-base` (fastest)
  - `declare-lab/flan-alpaca-gpt4-xl`
  - `google/flan-ul2`
  - `mistralai/Mistral-7B-Instruct-v0.1` (local only)
  

- 💾 **Session Logging**  
  Automatically saves conversation history to CSV in `/log`.

---

## 🚀 Run Locally (Optional)

```bash
git clone https://huggingface.co/spaces/scdong/mental_health_chatbot
cd mental_health_chatbot

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

For the Mistral or full-model version:

```bash
streamlit run app_use_Mistral-7B.py
# or
streamlit run app_with_FlanT5_FlanAlpacaGPT4_FlanUL2.py
```

---

## 📓 Dataset

This project uses:
- 🧾 `Kaggle_Mental_Health_Conversations_train.csv`  
  A labeled dataset of user statements and counselor responses used for training and testing classification.

---

## 🧑‍⚕️ Intended Use

This tool is designed for:
- Mental health professionals exploring AI assistance
- NLP researchers and students
- Developers building mental health chatbot prototypes

> ⚠️ **Not a substitute for professional mental health advice.**

---

## 📜 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more details.
```