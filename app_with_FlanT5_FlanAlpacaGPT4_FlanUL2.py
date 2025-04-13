# Streamlit App: Counselor Assistant (XGBoost + Selectable LLMs from Hugging Face)

import streamlit as st
import os
import pandas as pd
import json
import time
import csv
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from transformers import pipeline

# --- Page Setup ---
st.set_page_config(page_title="Counselor Assistant", layout="centered")

# --- Styling ---
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; padding: 1rem 2rem; border-radius: 12px; }
        h1 { color: #2c3e50; text-align: center; font-size: 2.4rem; }
        .user { color: #1f77b4; font-weight: bold; }
        .assistant { color: #2ca02c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.title("🧠 Mental Health Counselor Assistant")
st.markdown("""
Welcome, counselor 👋

This tool offers **AI-powered suggestions** to support you when responding to your patients.

### What it does:
- 🧩 Predicts what type of support is best: *Advice*, *Validation*, *Information*, or *Question*
- 💬 Generates an LLM-powered suggestion for you
- 💾 Lets you save your session for reflection

This is here to support — not replace — your clinical instincts 💚
""")

# --- Load and label dataset ---
df = pd.read_csv("dataset/Kaggle_Mental_Health_Conversations_train.csv")
df = df[['Context', 'Response']].dropna().copy()

# Auto-labeling: heuristics for labeling responses
keywords_to_labels = {
    'advice': ['try', 'should', 'suggest', 'recommend'],
    'validation': ['understand', 'feel', 'valid', 'normal'],
    'information': ['cause', 'often', 'disorder', 'symptom'],
    'question': ['how', 'what', 'why', 'have you']
}

def auto_label_response(response):
    response = response.lower()
    for label, keywords in keywords_to_labels.items():
        if any(word in response for word in keywords):
            return label
    return 'information'

df['response_type'] = df['Response'].apply(auto_label_response)
df['combined_text'] = df['Context'] + " " + df['Response']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['response_type'])

# TF-IDF vectorizer on combined text
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['combined_text'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# XGBoost Classifier
xgb_model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
xgb_model.fit(X_train, y_train)

# --- Select Model Option ---
model_options = {
    "google/flan-t5-base": "✅ Flan-T5 (Fast, Clean)",
    "declare-lab/flan-alpaca-gpt4-xl": "💬 Flan Alpaca GPT4 (Human-sounding)",
    "google/flan-ul2": "🧠 Flan-UL2 (Deeper reasoning)"
}

model_choice = st.selectbox("🧠 Choose a Response Model", list(model_options.keys()), format_func=lambda x: model_options[x])

@st.cache_resource(show_spinner="Loading selected language model...")
def load_llm(model_name):
    return pipeline("text2text-generation", model=model_name)

llm = load_llm(model_choice)

# --- Utility Functions ---
def predict_response_type(user_input):
    vec = vectorizer.transform([user_input])
    pred = xgb_model.predict(vec)
    proba = xgb_model.predict_proba(vec).max()
    label = le.inverse_transform(pred)[0]
    return label, proba

def build_prompt(user_input, response_type):
    prompts = {
        "advice": f"A patient said: \"{user_input}\". What advice should a mental health counselor give to support them?",
        "validation": f"A patient said: \"{user_input}\". How can a counselor validate and empathize with their emotions?",
        "information": f"A patient said: \"{user_input}\". Explain what might be happening from a mental health perspective.",
        "question": f"A patient said: \"{user_input}\". What thoughtful follow-up questions should a counselor ask?"
    }
    return prompts.get(response_type, prompts["information"])

def generate_llm_response(user_input, response_type):
    prompt = build_prompt(user_input, response_type)
    start = time.time()
    with st.spinner("Thinking through a helpful response for your patient..."):
        result = llm(prompt, max_length=150, do_sample=True, temperature=0.7)
    end = time.time()
    st.info(f"Response generated in {end - start:.1f} seconds")
    return result[0]["generated_text"].strip()

def trim_memory(history, max_turns=6):
    return history[-max_turns * 2:]

def save_conversation(history):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"chat_log_{now}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Role", "Content", "Intent", "Confidence"])
        for entry in history:
            writer.writerow([
                entry.get("role", ""),
                entry.get("content", ""),
                entry.get("label", ""),
                round(float(entry.get("confidence", 0)) * 100)
            ])
    st.success(f"Saved to chat_log_{now}.csv")

# --- Session State Setup ---
if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- Display Sample Prompts ---
with st.expander("💡 Sample inputs you can try"):
    st.markdown("""
    - My patient is constantly feeling overwhelmed at work.
    - A student says they panic every time they have to speak in class.
    - Someone told me they think they’ll never feel okay again.
    """)

# --- Text Area + Word Counter ---
MAX_WORDS = 1000
word_count = len(st.session_state.user_input.split())
st.markdown(f"**📝 Input Length:** {word_count} / {MAX_WORDS} words")

st.session_state.user_input = st.text_area(
    "💬 What did your patient say?",
    value=st.session_state.user_input,
    placeholder="e.g. I just feel like I'm never going to get better.",
    height=100
)

# --- Button Layout ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    send = st.button("💡 Suggest Response")
with col2:
    save = st.button("📁 Save This")
with col3:
    reset = st.button("🔁 Reset")

# --- Button Logic ---
if send and st.session_state.user_input:
    user_input = st.session_state.user_input
    predicted_type, confidence = predict_response_type(user_input)
    reply = generate_llm_response(user_input, predicted_type)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({
        "role": "assistant",
        "content": reply,
        "label": predicted_type,
        "confidence": confidence
    })
    st.session_state.history = trim_memory(st.session_state.history)

if save:
    save_conversation(st.session_state.history)

if reset:
    st.session_state.history = []
    st.session_state.user_input = ""
    st.success("Conversation has been cleared.")

# --- Chat History Display ---
st.markdown("---")
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.markdown(f"🧍‍♀️ **Patient:** {turn['content']}")
    else:
        st.markdown(f"👩‍⚕️👨‍⚕️ **Suggested Counselor Response:** {turn['content']}")
        st.caption(f"_Intent: {turn['label']} (Confidence: {turn['confidence']:.0%})_")
    st.markdown("---")

