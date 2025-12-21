# ============================
# 📦 Import Required Libraries
# ============================

import streamlit as st                     # Streamlit for UI
import time                                # For timestamps
import os                                  # OS-level utilities

# Custom app modules
from app.config import EURI_API_KEY         # API key configuration
from app.chat_utils import get_chat_model, ask_chat_model
from app.pdf_utils import extract_text_from_pdf
from app.ui import pdf_uploader
from app.vectorstore_utils import create_faiss_index, retrive_similar_documents

# LangChain text splitter for chunking documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================
# ⚙️ Streamlit Page Configuration
# ============================

st.set_page_config(
    page_title="MedChatBot",
    page_icon = "👨‍⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================
# 🎨 Custom CSS Styling
# ============================

st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        color: white;
    }
    .chat-message.assistant {
        background-color: #f0f2f6;
        color: black;
    }
    .timestamp {
        font-size: 0.8rem;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)


# ============================
# 🧠 Session State Initialization
# ============================

# Stores full chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Stores FAISS vector index
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Stores LLM chat model
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None


# ============================
# 🏥 App Header
# ============================

st.markdown("""
<div style="text-align:center;">
    <h1 style="color:#ff4b4b;">🏥 MediChat Pro</h1>
    <p>Your Intelligent Medical Document Assistant</p>
</div>
""", unsafe_allow_html=True)


# ============================
# 📂 Sidebar: PDF Upload Section
# ============================

with st.sidebar:
    st.markdown("### 📁 Document Upload")
    st.markdown("Upload medical PDFs to begin.")

    # Custom PDF uploader component
    uploaded_files = pdf_uploader()

    # If user uploads PDFs
    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded")

        # Button to process documents
        if st.button("🚀 Process Documents", type="primary"):
            with st.spinner("Processing medical documents..."):

                # ----------------------------
                # 1️⃣ Extract text from PDFs
                # ----------------------------
                all_texts = []
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    all_texts.append(text)

                # ----------------------------
                # 2️⃣ Split text into chunks
                # ----------------------------
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = []
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))

                # ----------------------------
                # 3️⃣ Create FAISS Vector Store
                # ----------------------------
                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore

                # ----------------------------
                # 4️⃣ Initialize Chat Model
                # ----------------------------
                chat_model = get_chat_model(EURI_API_KEY)
                st.session_state.chat_model = chat_model

                st.success("✅ Documents processed successfully!")
                st.balloons()


# ============================
# 💬 Chat Interface
# ============================

st.markdown("### 💬 Chat with Your Medical Documents")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.caption(msg["timestamp"])


# ============================
# 🧾 User Input Handling
# ============================

if prompt := st.chat_input("Ask about your medical documents..."):

    timestamp = time.strftime("%H:%M")

    # Store user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(timestamp)

    # ============================
    # 🤖 Assistant Response
    # ============================

    if st.session_state.vectorstore and st.session_state.chat_model:

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents..."):

                # 1️⃣ Retrieve similar chunks from FAISS
                relevant_docs = retrive_similar_documents(
                    st.session_state.vectorstore, prompt
                )

                # 2️⃣ Combine retrieved text as context
                context = "\n\n".join(
                    [doc.page_content for doc in relevant_docs]
                )

                # 3️⃣ Construct system prompt
                system_prompt = f"""
                You are MediChat Pro, an intelligent medical document assistant.

                Based on the following medical documents, answer accurately.
                If the answer is not present, clearly say so.

                Medical Documents:
                {context}

                User Question:
                {prompt}

                Answer:
                """

                # 4️⃣ Ask the LLM
                response = ask_chat_model(
                    st.session_state.chat_model,
                    system_prompt
                )

            # Display assistant response
            st.markdown(response)
            st.caption(timestamp)

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp
            })

    else:
        # Error if documents not processed
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process documents first!")
            st.cap(myenv) (base) PS C:\Users\user\OneDrive\Documents\Generative_AI\MediChatbot> git push -u origin main                            
remote: Repository not found.
fatal: repository 'https://github.com/your-username/medichat-pro.git/' not foundtion(timestamp)


# ============================
# 🔻 Footer
# ============================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
    🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence
</div>
""", unsafe_allow_html=True)
