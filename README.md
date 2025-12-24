# 🏥 MediChat Pro – Medical Document Chatbot (RAG-based)

MediChat Pro is an intelligent **medical document assistant** built using **Streamlit, LangChain, FAISS, and LLMs**.  
It allows users to upload medical PDFs and ask questions directly from those documents using **Retrieval-Augmented Generation (RAG)**.

Now enhanced with **Enterprise Features**: Authentication, Batch Processing, and Performance Analytics.

---

## 🚀 Key Features

### 1️⃣ Intelligent Chat
- **Context-Aware**: Retrieve accurate answers from uploaded medical documents.
- **Source Attribution**: See exactly which document the answer came from.
- **Robust AI**: Powered by Euri AI with retry mechanisms and error handling.

### 2️⃣ Advanced Document Processing
- **Batch Upload**: Process multiple PDF files simultaneously.
- **Progress Tracking**: Real-time status updates for document ingestion.
- **Metadata**: Track file size, page count, and processing status.

### 3️⃣ Security & Access Control
- **Role-Based Access**:
  - �‍⚕️ **Doctor**: Full access (Upload + Chat)
  - 👩‍⚕️ **Nurse**: Full access (Upload + Chat)
  - 👤 **Patient**: View only (Restricted upload)
- **Session Management**: Secure login/logout functionality.

### 4️⃣ Performance Analytics
- **Evaluation Framework**: Built-in benchmarking tools (`evaluation/evaluate.py`).
- **Proven Speed**: Average response latency ~1.79s.
- **High Recall**: Optimized chunking strategy for medical texts.

---

## 🧠 Architecture (RAG Pipeline)

1. **Secure Login**: Verify user credentials and role.
2. **PDF Ingestion**: Extract text from multiple PDFs (`MediPDF_Processor`).
3. **Chunking**: Smart text splitting (1000 chars) for optimal context.
4. **Vector Indexing**: FAISS + Sentence Transformers (`MediVectorStore`).
5. **Semantic Retrieval**: Find relevant chunks across all docs.
6. **LLM Generation**: Euri AI model generates the final answer (`MediChatUtils`).

---

## 📂 Project Structure

```
MediChatbot/
├── app/
│   ├── auth.py              # User authentication logic
│   ├── chat_utils.py        # Chat model interaction
│   ├── pdf_utils.py         # PDF text extraction
│   ├── vectorstore_utils.py # FAISS index handler
│   ├── ui.py                # UI components
│   └── config.py            # API configuration
├── MediPDF_Processor/       # Core PDF logic modules
├── MediVectorStore/         # Core Vector Store modules
├── MediChatUtils/           # Core Chat logic modules
├── evaluation/              # Performance benchmarking
│   ├── evaluate.py          # Bechnmark script
│   └── evaluation_dataset.json
├── main.py                  # Main Streamlit Application
└── requirements.txt         # Dependencies
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/medichat-pro.git
cd medichat-pro
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Configure API Key
Create `app/config.py` (if not exists) and add:
```python
EURI_API_KEY = "your_api_key_here"
```

### 4️⃣ Run the Application
```bash
streamlit run main.py
```
Open browser at: `http://localhost:8501`

---

## 🔐 Login Credentials (Demo)

Use these accounts to test role-based access:

| Role | Username | Password | Features |
|------|----------|----------|----------|
| **Doctor** | `doctor` | `password123` | Full Access |
| **Nurse** | `nurse` | `nurse123` | Full Access |
| **Patient** | `patient` | `patient123` | Chat Only |

---

## ⚠️ Medical Disclaimer

- This application is for educational and informational purposes only.
- It is not a substitute for professional medical advice, diagnosis, or treatment.

---

### 👨‍💻 Author

**Aman Husain**  
Data Scientist | GenAI Enthusiast | RAG & LLM Practitioner
