# Streamlit App: Counselor Assistant using XGBoost + Flan-T5 (Cloud Version)

import streamlit as st
import os
import pandas as pd
import json
import time
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from transformers import pipeline

st.set_page_config(page_title="Counselor Assistant", layout="centered")

st.markdown("""
    <style>
        .main { background-color: #f4f4f9; padding: 1rem 2rem; border-radius: 12px; }
        h1 { color: #2c3e50; text-align: center; font-size: 2.4rem; }
        .user { color: #1f77b4; font-weight: bold; }
        .assistant { color: #2ca02c; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("Mental Health Counselor Assistant")
st.markdown("""
Welcome, counselor 👩‍⚕️👨‍⚕️

This assistant is designed to provide you with **supportive, evidence-based suggestions** when you're unsure how to best respond to a patient’s concerns.

Just enter what your patient shared with you, and this tool will:
- Predict the type of support that fits best (e.g., advice, validation, information, and question)
- Generate a suggested counselor reply
- Let you save the conversation for your records

This is not a diagnostic tool — it’s here to support **your clinical intuition**.
""")

# Load and prepare the dataset
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

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

# Replace Mistral-7B with Flan-T5 hosted model
@st.cache_resource(show_spinner="Loading Flan-T5 model...")
def load_llm():
    return pipeline("text2text-generation", model="google/flan-t5-base")

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
        result = llm(prompt, max_length=150, do_sample=True, temperature=0.7)
    end = time.time()
    st.info(f"Response generated in {end - start:.1f} seconds")
    return result[0]["generated_text"].strip()

def trim_memory(history, max_turns=6):
    return history[-max_turns * 2:]

def save_conversation(history):
    with open("chat_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open("chat_log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Role", "Content"])
        for entry in history:
            writer.writerow([entry.get("role", ""), entry.get("content", "")])
    st.success("Saved to chat_history.json and chat_log.csv")

# Streamlit UI
if "history" not in st.session_state:
    st.session_state.history = []

with st.expander("💡 Sample inputs you can try"):
    st.markdown("""
    - My patient is constantly feeling overwhelmed at work.
    - A student says they panic every time they have to speak in class.
    - Someone told me they think they’ll never feel okay again.
    """)

user_input = st.text_area("💬 What did your patient say?", placeholder="e.g. I just feel like I'm never going to get better.", height=100)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    send = st.button("Suggest Response")
with col2:
    save = st.button("📁 Save This")
with col3:
    reset = st.button("🔁 Reset")

if send and user_input:
    predicted_type, confidence = predict_response_type(user_input)
    reply = generate_llm_response(user_input, predicted_type)

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": reply, "label": predicted_type, "confidence": confidence})
    st.session_state.history = trim_memory(st.session_state.history)

if save:
    save_conversation(st.session_state.history)

if reset:
    st.session_state.history = []
    st.success("Conversation has been cleared.")

st.markdown("---")
for turn in st.session_state.history:
    if turn["role"] == "user":
        st.markdown(f"🧍‍♀️ **Patient:** {turn['content']}")
    else:
        st.markdown(f"👩‍⚕️👨‍⚕️ **Suggested Counselor Response:** {turn['content']}")
        st.caption(f"_Intent: {turn['label']} (Confidence: {turn['confidence']:.0%})_")
    st.markdown("---")

