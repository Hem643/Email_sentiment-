
import time
import imaplib
import email
from email.header import decode_header
import streamlit as st
from collections import defaultdict
import re
import json
import os
from transformers import pipeline
import plotly.graph_objects as go

# Initialize sentiment classification pipeline
sentiment_pipeline = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

LAST_EXECUTION_FILE = "Execution10.txt"
EMAIL_CACHE_FILE = "emails_cache12.json"

def read_last_execution_time():
    if os.path.exists(LAST_EXECUTION_FILE):
        with open(LAST_EXECUTION_FILE, "r") as f:
            return float(f.read().strip())
    return 0

def save_last_execution_time(timestamp):
    with open(LAST_EXECUTION_FILE, "w") as f:
        f.write(str(timestamp))

def get_remaining_time(last_execution_time):
    current_time = time.time()
    return max(0, 86400 - (current_time - last_execution_time))

def save_email_cache(emails_by_domain):
    with open(EMAIL_CACHE_FILE, "w") as f:
        json.dump(emails_by_domain, f)

def load_email_cache():
    if os.path.exists(EMAIL_CACHE_FILE):
        with open(EMAIL_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def fetch_all_emails(server, username, password):
    try:
        mail = imaplib.IMAP4_SSL(server)
        mail.login(username, password)
        mail.select("inbox")
        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()
        emails_by_domain = defaultdict(list)
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    sender = msg.get("From")
                    match = re.search(r"@([\w.-]+)", sender)
                    domain = match.group(1) if match else "Unknown"
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode(errors='ignore')
                                emails_by_domain[domain].append(body)
                    else:
                        body = msg.get_payload(decode=True).decode(errors='ignore')
                        emails_by_domain[domain].append(body)
        mail.logout()
        return emails_by_domain
    except:
        return None

def detect_emotions(emails):
    categories = {"angry": 0, "threat": 0, "neutral": 0, "joy": 0, "happy": 0}
    for email_content in emails:
        analysis = sentiment_pipeline(email_content[:512])
        sentiment_score = int(analysis[0]['label'].split()[0])
        if sentiment_score == 1:
            categories["angry"] += 1
        elif sentiment_score == 2:
            categories["threat"] += 1
        elif sentiment_score == 3:
            categories["neutral"] += 1
        elif sentiment_score == 4:
            categories["joy"] += 1
        else:
            categories["happy"] += 1
    return categories

def plot_emotion_scale(categories):
    total = sum(categories.values())
    sentiment_score = (
        (categories["happy"] * 5 + categories["joy"] * 4 + categories["neutral"] * 3 + categories["threat"] * 2 + categories["angry"] * 1) / total
        if total > 0 else 3
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        title={'text': "Emotion Scale"},
        gauge={
            'axis': {'range': [1, 5], 'tickvals': [1, 2, 3, 4, 5], 'ticktext': ["Angry", "Threat", "Neutral", "Joy", "Happy"]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [1, 2], 'color': "red"},
                {'range': [2, 3], 'color': "orange"},
                {'range': [3, 4], 'color': "yellow"},
                {'range': [4, 5], 'color': "green"}
            ],
            'threshold': {'line': {'color': "blue", 'width': 4}, 'value': sentiment_score}
        }
    ))
    return fig

def main():
    st.title("Email Sentiment Analysis Chatbot")

    if "last_fetch_time" not in st.session_state:
        st.session_state.last_fetch_time = read_last_execution_time()
        st.session_state.emails_by_domain = load_email_cache()

    remaining_time = get_remaining_time(st.session_state.last_fetch_time)

    if remaining_time == 0:
        st.session_state.emails_by_domain = {}
        st.session_state.last_fetch_time = time.time()
        save_last_execution_time(st.session_state.last_fetch_time)

    email_input = st.text_input("Enter your email:")
    password_input = st.text_input("Enter your password:", type="password")

    provider = st.selectbox("Select your provider:", ["Gmail", "Outlook"])
    if st.button("Fetch Emails"):
        server = "imap.gmail.com" if provider == "Gmail" else "imap-mail.outlook.com"
        emails = fetch_all_emails(server, email_input, password_input)
        if emails:
            st.session_state.emails_by_domain = emails
            save_email_cache(emails)
            st.session_state.last_fetch_time = time.time()
            save_last_execution_time(time.time())
            st.success("Emails fetched successfully!")

    if st.session_state.emails_by_domain:
        selected_domain = st.selectbox("Select a domain to analyze:", list(st.session_state.emails_by_domain.keys()))
        if st.button("Analyze Sentiment"):
            emails = st.session_state.emails_by_domain.get(selected_domain, [])
            if emails:
                categories = detect_emotions(emails)
                st.write(f"Sentiment analysis for {selected_domain}: {categories}")
                fig = plot_emotion_scale(categories)
                st.plotly_chart(fig)
            else:
                st.warning("No emails found.")

if __name__ == "__main__":
    main()
