# ============================
# 📦 Import Required Libraries
# ============================

import streamlit as st                     # Streamlit for UI
import time                                # For timestamps
import os                                  # OS-level utilities
import hashlib                             # For generating document IDs
from datetime import datetime              # For timestamps

# Custom app modules
from app.config import EURI_API_KEY         # API key configuration
from app.chat_utils import get_chat_model, ask_chat_model
from app.pdf_utils import extract_text_from_pdf
from app.ui import pdf_uploader
from app.vectorstore_utils import create_faiss_index, retrive_similar_documents
from app.auth import authenticate           # Authentication module

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
# 🔐 Authentication Check
# ============================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_name = None

def login():
    st.markdown("""
    <div style="text-align:center; margin-top: 50px;">
        <h1 style="color:#ff4b4b;">🏥 MediChat Pro</h1>
        <p>Please log in to access the medical system.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Log In", type="primary", use_container_width=True):
            user = authenticate(username, password)
            if user:
                st.session_state.authenticated = True
                st.session_state.user_role = user["role"]
                st.session_state.user_name = user["name"]
                st.rerun()
            else:
                st.error("Invalid username or password")
                
        st.info("Demo Credentials:\n\nDoctor: doctor / password123\nNurse: nurse / nurse123\nPatient: patient / patient123")

def logout():
    st.session_state.authenticated = False
    st.session_state.user_role = None
    st.session_state.user_name = None
    st.rerun()

if not st.session_state.authenticated:
    login()
    st.stop()  # Stop execution if not authenticated

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
    .doc-card {
        padding: 0.8rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        margin-bottom: 0.5rem;
        border-left: 3px solid #ff4b4b;
    }
    .doc-card-success {
        border-left-color: #28a745;
    }
    .doc-card-error {
        border-left-color: #dc3545;
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

# Stores document metadata
if "documents" not in st.session_state:
    st.session_state.documents = {}

# Stores all document chunks with metadata
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []


# ============================
# 🔧 Helper Functions
# ============================

def generate_doc_id(filename, content):
    """Generate unique document ID based on filename and content"""
    hash_input = f"{filename}{content[:100]}".encode()
    return hashlib.md5(hash_input).hexdigest()[:8]


def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ============================
# 🏥 App Header & User Info
# ============================

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    <div>
        <h1 style="color:#ff4b4b; margin-bottom:0;">🏥 MediChat Pro</h1>
        <p>Your Intelligent Medical Document Assistant</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"**👤 {st.session_state.user_name}**")
    st.markdown(f"Role: `{st.session_state.user_role.title()}`")
    if st.button("Log Out", key="logout_btn", type="secondary"):
        logout()

st.divider()

# ============================
# 📂 Sidebar: PDF Upload Section
# ============================

with st.sidebar:
    st.markdown("### 📁 Document Upload")
    
    # Role-based Access Control
    if st.session_state.user_role in ["doctor", "nurse"]:
        st.markdown("Upload medical PDFs to begin.")

        # Custom PDF uploader component
        uploaded_files = pdf_uploader()

        # If user uploads PDFs
        if uploaded_files:
            st.success(f"{len(uploaded_files)} document(s) uploaded")

            # Button to process documents
            if st.button("🚀 Process Documents", type="primary"):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_files = len(uploaded_files)
                processed_docs = {}
                all_chunks_with_metadata = []
                
                for idx, file in enumerate(uploaded_files):
                    try:
                        # Update progress
                        progress = (idx + 1) / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {file.name}... ({idx + 1}/{total_files})")
                        
                        # ----------------------------
                        # 1️⃣ Extract text from PDF
                        # ----------------------------
                        start_time = time.time()
                        text = extract_text_from_pdf(file)
                        
                        if not text or len(text.strip()) == 0:
                            raise ValueError("No text extracted from PDF")
                        
                        # ----------------------------
                        # 2️⃣ Split text into chunks
                        # ----------------------------
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = text_splitter.split_text(text)
                        
                        # ----------------------------
                        # 3️⃣ Create document metadata
                        # ----------------------------
                        doc_id = generate_doc_id(file.name, text)
                        processing_time = time.time() - start_time
                        
                        # Store document metadata
                        processed_docs[doc_id] = {
                            "filename": file.name,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "file_size": format_file_size(file.size),
                            "num_chunks": len(chunks),
                            "processing_time": f"{processing_time:.2f}s",
                            "status": "success",
                            "text_length": len(text)
                        }
                        
                        # Create chunks with metadata
                        for chunk_idx, chunk in enumerate(chunks):
                            chunk_data = {
                                "content": chunk,
                                "metadata": {
                                    "source": file.name,
                                    "doc_id": doc_id,
                                    "chunk_id": chunk_idx,
                                    "total_chunks": len(chunks)
                                }
                            }
                            all_chunks_with_metadata.append(chunk_data)
                        
                    except Exception as e:
                        # Handle errors gracefully
                        doc_id = generate_doc_id(file.name, str(time.time()))
                        processed_docs[doc_id] = {
                            "filename": file.name,
                            "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "file_size": format_file_size(file.size),
                            "status": "error",
                            "error_message": str(e)
                        }
                
                # ----------------------------
                # 4️⃣ Create FAISS Vector Store
                # ----------------------------
                if all_chunks_with_metadata:
                    status_text.text("Creating vector index...")
                    
                    # Extract just the text for FAISS
                    texts = [doc["content"] for doc in all_chunks_with_metadata]
                    vectorstore = create_faiss_index(texts)
                    
                    st.session_state.vectorstore = vectorstore
                    st.session_state.all_chunks = all_chunks_with_metadata
                    st.session_state.documents = processed_docs
                    
                    # ----------------------------
                    # 5️⃣ Initialize Chat Model
                    # ----------------------------
                    if not st.session_state.chat_model:
                        chat_model = get_chat_model(EURI_API_KEY)
                        st.session_state.chat_model = chat_model
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ All documents processed!")
                    time.sleep(1)
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Successfully processed {len([d for d in processed_docs.values() if d['status'] == 'success'])} document(s)!")
                    st.balloons()
                else:
                    st.error("❌ No documents could be processed successfully.")
    else:
        st.info("🔒 Document upload is restricted to medical staff.")
    
    # ----------------------------
    # 📋 Document List Display
    # ----------------------------
    if st.session_state.documents:
        st.markdown("---")
        st.markdown("### 📋 Processed Documents")
        
        for doc_id, doc_info in st.session_state.documents.items():
            with st.expander(f"📄 {doc_info['filename']}", expanded=False):
                if doc_info['status'] == 'success':
                    st.markdown(f"""
                    <div class="doc-card doc-card-success">
                        <strong>✅ Status:</strong> Processed<br>
                        <strong>📊 Chunks:</strong> {doc_info['num_chunks']}<br>
                        <strong>📏 Size:</strong> {doc_info['file_size']}<br>
                        <strong>⏱️ Processing Time:</strong> {doc_info['processing_time']}<br>
                        <strong>🕐 Uploaded:</strong> {doc_info['upload_time']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="doc-card doc-card-error">
                        <strong>❌ Status:</strong> Error<br>
                        <strong>⚠️ Error:</strong> {doc_info.get('error_message', 'Unknown error')}<br>
                        <strong>🕐 Uploaded:</strong> {doc_info['upload_time']}
                    </div>
                    """, unsafe_allow_html=True)


# ============================
# 💬 Chat Interface
# ============================

st.markdown("### 💬 Chat with Your Medical Documents")

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")
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

                # 2️⃣ Get source documents
                sources = set()
                context_parts = []
                
                for i, doc in enumerate(relevant_docs):
                    # Find corresponding metadata
                    if i < len(st.session_state.all_chunks):
                        chunk_data = st.session_state.all_chunks[i]
                        source = chunk_data["metadata"].get("source", "Unknown")
                        sources.add(source)
                        context_parts.append(f"[From {source}]\n{doc.page_content}")
                    else:
                        context_parts.append(doc.page_content)
                
                context = "\n\n".join(context_parts)

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
            if sources:
                st.caption(f"📄 Sources: {', '.join(sources)}")
            st.caption(timestamp)

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": timestamp,
                "sources": list(sources)
            })

    else:
        # Error if documents not processed
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload and process documents first!. If you are a patient, ask your doctor to upload documents.")
            st.caption(timestamp)


# ============================
# 🔻 Footer
# ============================

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
    🤖 Powered by Euri AI & LangChain | 🏥 Medical Document Intelligence
</div>
""", unsafe_allow_html=True)
