import streamlit as st
from utils.helper_functions import *
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
from llama_cpp import Llama

st.set_page_config(page_title="Counselor Assistant", layout="centered")

st.markdown("""
    <style>
        .main { background-color: #f9f9f9; padding: 1rem 2rem; border-radius: 12px; }
        h1 { color: #2c3e50; text-align: center; font-size: 2.4rem; }
        .user { color: #1f77b4; font-weight: bold; }
        .assistant { color: #2ca02c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Health Counselor Assistant")
st.markdown("""
Hi there, counselor 👋

This tool is here to offer **supportive, AI-generated suggestions** when you’re not quite sure how to respond to a patient.

### How it helps:
- 🧩 Predicts the type of support your patient might need (advice, validation, information, & question.)
- 💬 Generates a supportive counselor response
- 📁 Lets you save and track conversations for reflection

It's a sidekick, not a substitute for your clinical judgment 💚
""")

df = pd.read_csv("dataset/Kaggle_Mental_Health_Conversations_train.csv")
df = df[['Context', 'Response']].dropna().copy()

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

le = LabelEncoder()
y = le.fit_transform(df['response_type'])

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(df['combined_text'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

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

MODEL_PATH = os.path.expanduser("/Users/Pi/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
@st.cache_resource(show_spinner=True)
def load_llm():
    return Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=os.cpu_count())

llm = load_llm()

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
        result = llm(prompt, max_tokens=300, temperature=0.7)
    end = time.time()
    st.info(f"Response generated in {end - start:.1f} seconds")
    return result['choices'][0]['text'].strip()

def trim_memory(history, max_turns=6):
    return history[-max_turns * 2:]

def save_conversation(history):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/chat_log_combined.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Role", "Content", "Intent", "Confidence"])
        for entry in history:
            writer.writerow([
                now,
                entry.get("role", ""),
                entry.get("content", ""),
                entry.get("label", ""),
                round(float(entry.get("confidence", 0)), 2)
            ])
    st.success("Saved to chat_log_combined.csv")

if "history" not in st.session_state:
    st.session_state.history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

MAX_WORDS = 1000
word_count = len(st.session_state.user_input.split())
st.markdown(f"**📝 Input Length:** {word_count} / {MAX_WORDS} words")
st.session_state.user_input = st.text_area(
    "💬 What did your patient say?",
    value=st.session_state.user_input,
    placeholder="e.g. I just feel like I'm never going to get better.",
    height=100
)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    send = st.button("💡 Suggest Response")
with col2:
    save = st.button("📁 Save This")
with col3:
    reset = st.button("🔁 Reset")

if send and st.session_state.user_input:
    user_input = st.session_state.user_input
    predicted_type, confidence = predict_response_type(user_input)
    reply = generate_llm_response(user_input, predicted_type)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": reply, "label": predicted_type, "confidence": confidence})
    st.session_state.history = trim_memory(st.session_state.history)

if save:
    save_conversation(st.session_state.history)

if reset:
    st.session_state.history = []
    st.session_state.user_input = ""
    st.success("Conversation has been cleared.")

st.markdown("---")
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.markdown(f"🧍‍♀️ **Patient:** {turn['content']}")
    else:
        st.markdown(f"👨‍⚕️ **Suggested Counselor Response:** {turn['content']}")
        st.caption(f"_Intent: {turn['label']} (Confidence: {turn['confidence']:.0%})_")
    st.markdown("---")

