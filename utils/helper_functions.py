# utils/helper_functions.py

import time
import json
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Labeling logic
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

def build_prompt(user_input, response_type):
    prompts = {
        "advice": f"A patient said: \"{user_input}\". What advice should a mental health counselor give to support them?",
        "validation": f"A patient said: \"{user_input}\". How can a counselor validate and empathize with their emotions?",
        "information": f"A patient said: \"{user_input}\". Explain what might be happening from a mental health perspective.",
        "question": f"A patient said: \"{user_input}\". What thoughtful follow-up questions should a counselor ask?"
    }
    return prompts.get(response_type, prompts["information"])

def predict_response_type(user_input, model, vectorizer, label_encoder):
    vec = vectorizer.transform([user_input])
    pred = model.predict(vec)
    proba = model.predict_proba(vec).max()
    label = label_encoder.inverse_transform(pred)[0]
    return label, proba

def generate_llm_response(prompt, llm):
    start = time.time()
    result = llm(prompt, max_tokens=300, temperature=0.7)
    end = time.time()
    elapsed = round(end - start, 1)
    return result['choices'][0]['text'].strip(), elapsed

def trim_memory(history, max_turns=6):
    return history[-max_turns * 2:]

def save_conversation(history):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"chat_log_{timestamp}.csv"
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Role", "Content", "Intent", "Confidence"])
        for turn in history:
            writer.writerow([
                turn.get("role", ""),
                turn.get("content", ""),
                turn.get("label", ""),
                round(float(turn.get("confidence", 0)) * 100)
            ])
    print(f"Saved to {file_name}")
    return file_name

