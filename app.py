import streamlit as st
from groq import Groq
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

 # Load FAISS index and metadata
index = faiss.read_index("medquad_index.faiss")
with open("medquad_answers.pkl", "rb") as f:
    metadata = pickle.load(f)
questions = metadata["questions"]
answers = metadata["answers"]

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === STREAMLIT UI ===
st.set_page_config(page_title="Medical QA Bot", page_icon="💬")
st.title("Medical Question Answering Bot")

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

    #FAISS INDEXING
    query_vec = embedder.encode([user_input]).astype("float32")
    top_k = 3
    D, I = index.search(query_vec, k=top_k)
    contexts = []
    for idx in I[0]:
        q = questions[idx]
        a = answers[idx]
        contexts.append(f"Q: {q}\nA: {a}")
    context = "Retrieved from medical data:\n\n" + "\n\n".join(contexts)

    base_prompt = f"""
    You are a medical assistant AI that prioritizes factual accuracy and only answers medical questions using provided context from a trusted dataset.

    If the question is a general greeting or casual message (like "hello", "thanks", or "how are you?"), respond politely and conversationally.

    If a medical question is asked but is NOT covered by the provided context answer with a general explaination and say before answering:
    "I do not have the medical data to answer that, but I can offer this general explanation. However, I heavily recommend speaking with a medical professional for accurate information."

    Only give detailed answers if the relevant information is included in the provided context.

    Medical context:
    """


    # Send to model
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": base_prompt
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\n"
                    f"{user_input}\n"
                )
            }
        ]
    )

    reply = response.choices[0].message.content
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
