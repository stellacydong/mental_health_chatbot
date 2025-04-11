# 🧠 Mental Health Counselor Assistant

This Streamlit app helps mental health professionals by offering **AI-powered suggestions** for how to support their patients, based on real input.

---

## ✅ Problem Statement

Mental health counselors often encounter complex, emotionally charged statements. This tool:
- Predicts the **intent category** (e.g., advice, validation, information, question)
- Generates a thoughtful, supportive reply using a local large language model (LLM)

This is a decision support tool — not a diagnostic system.

---

## 🔧 Features

- 🧠 **Intent Classification** with XGBoost + TF-IDF
- 💬 **Multi-Turn Chat Memory** (up to 6 turns)
- 🤖 **LLM Responses** using quantized Mistral-7B via `llama-cpp`
- 📊 Confidence estimates for model predictions
- 💾 Export chats to `.csv` or `.json` with timestamps
- 🎨 Friendly UI with examples, reset button, and usage tips

---

## 🚀 How to Run

### ✅ Option 1: Local (Recommended on Mac M1/M2)
```bash
pip install -r requirements.txt
streamlit run app_local.py



