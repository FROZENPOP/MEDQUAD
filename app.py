import streamlit as st
from groq import Groq
import os

GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Medical QA Bot", page_icon="💬")
st.title("💬 Medical Question Answering Bot (Groq + LLaMA 3)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Show user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Send to Groq LLaMA-3 model
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful and accurate medical assistant. Keep responses short and fact-based."},
            *st.session_state.messages
        ]
    )

    reply = response.choices[0].message.content
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
