
---

```markdown
# 🧠 Mental Health Counselor Assistant

A privacy-friendly, intelligent assistant designed to help mental health professionals explore **response suggestions** based on patient input. Built with `XGBoost` for intent classification and `Mistral-7B` for natural language generation, this app helps support clinical conversations with empathy and structure — while keeping you in full control.

---

## ✅ Problem Statement

Mental health counselors are tasked with responding to a wide range of complex emotions and situations. In high-pressure or uncertain moments, this tool helps:

- Predict the **intent** behind a patient's message (advice-seeking, validation, informational, or inquisitive)
- Generate a supportive, **AI-assisted counselor response**
- Log conversations for review and learning
- Preserve privacy with local inference — built for **HIPAA-aware workflows**

---

## ✨ Features

- 🔍 **Intent Prediction** — XGBoost classifier trained on annotated mental health dialogue
- 💬 **Response Generation** — LLM-backed replies using quantized [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b/)
- 🧠 **Multi-turn Memory** — Maintains up to 6 rounds of conversation
- ✅ **Export to JSON + CSV** — With timestamps for traceability
- 🧰 **HIPAA-conscious Design** — No third-party API calls required; all runs locally

---

## 🖼️ App Preview

![counselor assistant demo](demo/demo.gif)

---

## 🚀 How to Run

### 🔧 Local (with Mistral-7B)

> 🧩 You must download a quantized `.gguf` model file from Hugging Face or TheBloke, e.g.:
>
> `/Users/yourname/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf`

```bash
git clone https://github.com/yourname/mental-health-chatbot.git
cd mental-health-chatbot
pip install -r requirements.txt
streamlit run app_local.py
```

---

### 🐳 Docker (Recommended)

```bash
docker build -t counselor-assistant .
docker run -p 8501:8501 counselor-assistant
```

---

## 📁 Project Structure

```
mental-health-chatbot/
│
├── app_local.py              # Streamlit app (uses llama-cpp + XGBoost)
├── requirements.txt
├── Dockerfile
├── demo/                     # Optional: screenshots or .gif
├── dataset/
│   └── Kaggle_Mental_Health_Conversations_train.csv
├── models/
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
├── README.md
└── LICENSE
```

---

## 📌 Design Decisions

- 🧠 Used `TfidfVectorizer` + `XGBoost` as a **lightweight, interpretable** classifier
- 🔒 LLM inference handled locally with `llama-cpp-python` — no external data leaks
- 📦 Docker support to simulate realistic deployment and improve portability
- 🎯 Optimized for educational and prototyping use in **clinician settings**

---

## 📍 Future Improvements

- Add long-form audio transcription support
- Integrate feedback loop for model tuning
- Auto-summary of sessions
- Hugging Face Space version (hosted with opt-in privacy tradeoffs)

---

## 👩‍⚕️ A Note to Reviewers

This project was developed for an interview to showcase how **AI and empathy** can work hand-in-hand to support mental health professionals. Thank you for reviewing, and I hope it sparks a great conversation about the intersection of tech and care. 💙

---

## 📄 License

MIT — free to use and modify, but please be thoughtful in healthcare applications.
```

---

