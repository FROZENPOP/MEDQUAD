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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []  # conversation history
if "show_source" not in st.session_state:
    st.session_state.show_source = False
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3


# Sidebar with toggles and settings
with st.sidebar:
    st.header("Settings")

    # Temperature
    st.session_state.temperature = st.slider("Response Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)

    #Tokens, response len
    max_tokens = st.slider("Max response length (tokens)", 50, 2000, 512, 50)

    # Response style selector
    response_style = st.radio("Response Style", ["Detailed", "General"], index=0)

    #Queries
    top_k = st.slider("Number of relevant entries to retrieve (Top K)", min_value=1, max_value=10, value=3)

    # Show source toggle
    st.session_state.show_source = st.checkbox("Show source context with answers", value=st.session_state.show_source)

    # Reset conversation 
    if st.button("Reset Conversation"):
        st.session_state.conversation = []
        st.session_state.chat_history = []
        st.session_state.messages = []


    
# Show conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If showing source and message is assistant, show the retrieved context too
        if message["role"] == "assistant" and st.session_state.show_source and "source" in message:
            st.markdown("**Source context:**")
            for q, a in message["source"]:
                st.markdown(f"- **Q:** {q}")
                st.markdown(f"  \n**A:** {a}")

# User input
user_input = st.chat_input("Ask a medical question...")

if user_input:
    # Show user message in chat
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # FAISS search for top-k context
    query_vec = embedder.encode([user_input]).astype("float32")
    D, I = index.search(query_vec, k=top_k)
    contexts = []
    for idx in I[0]:
        q = questions[idx]
        a = answers[idx]
        contexts.append((q, a))

    # Prepare context string for prompt
    context_text = "Retrieved from medical data:\n\n" + "\n\n".join([f"Q: {q}\nA: {a}" for q, a in contexts])

    # Modify instructions based on response style
    if response_style == "Detailed":
        style_instruction = "Provide a thorough and comprehensive response using all relevant medical context."
    else:
        style_instruction = "Respond concisely, summarizing key points from the context without excessive detail."


    base_prompt = f"""
You are a medical assistant AI that prioritizes factual accuracy and only answers medical questions using provided context from a trusted dataset.

If the question is a general greeting or casual message (like "hello", "thanks", or "how are you?"), respond politely and conversationally.

If a medical question is asked but is NOT covered by the provided context answer with a general explanation and say before answering:
"I do not have the medical data to answer that, but I can offer this general explanation. However, I heavily recommend speaking with a medical professional for accurate information."

{style_instruction}

Medical context:
"""

    # Call Groq model with temperature parameter
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
                    f"{context_text}\n\n"
                    f"{user_input}\n"
                )
            }
        ],
        max_tokens=max_tokens,
        temperature=st.session_state.temperature,
    )

    reply = response.choices[0].message.content

    # Show assistant message
    st.chat_message("assistant").markdown(reply)

    # Append assistant message along with source if toggled on
    msg_data = {"role": "assistant", "content": reply}
    if st.session_state.show_source:
        msg_data["source"] = contexts
    st.session_state.messages.append(msg_data)

# Download conversation transcript helper
def get_transcript():
    lines = []
    for msg in st.session_state.messages:
        lines.append(f"{msg['role'].capitalize()}: {msg['content']}\n")
        if msg["role"] == "assistant" and "source" in msg and st.session_state.show_source:
            lines.append("Source context:\n")
            for q, a in msg["source"]:
                lines.append(f"Q: {q}\nA: {a}\n")
            lines.append("\n")
    return "".join(lines)

if st.session_state.messages:
    transcript = get_transcript()
    st.download_button(
        label="Download conversation transcript",
        data=transcript,
        file_name="medquad_conversation.txt",
        mime="text/plain"
    )

