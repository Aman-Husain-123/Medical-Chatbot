# Medical-Chatbot
Medical Chatbot Project With RAG And Vector DB

# 🏥 MediChat Pro – Medical Document Chatbot (RAG-based)

MediChat Pro is an intelligent **medical document assistant** built using **Streamlit, LangChain, FAISS, and LLMs**.  
It allows users to upload medical PDFs and ask questions directly from those documents using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

- 📄 Upload multiple medical PDF documents
- 🔍 Semantic search using FAISS vector database
- 🧠 Context-aware answers using LLMs
- 💬 Chat-style conversational interface
- 🕒 Timestamped chat history
- 🎨 Clean & professional medical UI
- ⚠️ Graceful handling when answers are not found in documents

---

## 🧠 Architecture (RAG Pipeline)

1. **PDF Upload**
2. **Text Extraction**
3. **Text Chunking**
4. **Vector Embedding + FAISS Index**
5. **Semantic Retrieval**
6. **LLM Response Generation**
7. **Streamlit Chat UI**

---

## 🛠️ Tech Stack

| Category | Technology |
|--------|------------|
| Frontend | Streamlit |
| LLM Framework | LangChain |
| Vector Store | FAISS |
| Language Model | Euri AI (LLM API) |
| Text Processing | RecursiveCharacterTextSplitter |
| Backend | Python |
| Embeddings | LLM-based embeddings |

---

## 📂 Project Structure

MedChat-Pro/
│
├── app/
│ ├── chat_utils.py # LLM initialization & querying
│ ├── pdf_utils.py # PDF text extraction
│ ├── vectorstore_utils.py # FAISS index creation & retrieval
│ ├── ui.py # UI helper components
│ └── config.py # API keys & configuration
│
├── assets/
│ └── doctor_icon.png # (Optional) App icon
│
├── app.py # Main Streamlit application
├── requirements.txt # Project dependencies
├── README.md # Project documentation
└── .gitignore


---


---
## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/medichat-pro.git
cd medichat-pro


### 2️⃣ Create & activate virtual environment

Windows
python -m venv myenv
myenv\Scripts\activate


macOS / Linux
python3 -m venv myenv
source myenv/bin/activate


### 3️⃣ Install dependencies
pip install -r requirements.txt


###🔐 Environment Configuration

Create app/config.py and add:
EURI_API_KEY = "your_api_key_here"

⚠️ Never commit API keys to GitHub


--- 

### ▶️ Run the Application
streamlit run app.py


The app will be available at:
http://localhost:8501


---

## 📸 Output / Application Preview

### 🏥 Home Interface
- Clean medical-themed UI built with Streamlit  
- Sidebar for PDF uploads and document processing  
- Chat interface inspired by modern conversational AI tools  

---

### 📂 Document Upload & Processing
- Users can upload **multiple medical PDF documents**
- Clicking **“Process Documents”**:
  - Extracts text from PDFs
  - Splits text into chunks
  - Creates FAISS vector embeddings
  - Initializes the LLM chat model
- Success confirmation with visual feedback

---

### 💬 Chat with Medical Documents
- Users can ask **natural language questions**
- System retrieves **relevant document chunks**
- LLM generates **context-aware responses**
- Each message includes a **timestamp**
- If information is not present in documents, the model clearly states so

---

### ⚠️ Error Handling
- Displays a warning if the user tries to chat before processing documents
- Prevents hallucinated answers by restricting responses to document context

---

### 🖼️ Sample Screenshots (Optional)

You can add screenshots like this:

```md
<img width="1911" height="908" alt="image" src="https://github.com/user-attachments/assets/5139e01b-6190-4d85-8052-313e1f9c121b" />



---

### 🧪 How to Use

- Upload one or more medical PDF files
- Click Process Documents
- Ask questions related to uploaded documents
- Get context-aware medical responses

---

⚠️ Medical Disclaimer

- This application is for educational and informational purposes only.
- It is not a substitute for professional medical advice, diagnosis, or treatment


---

### 📌 Future Enhancements

- 🔐 User authentication
- 🧾 Source citation highlighting
- 🌐 Deployment on cloud (AWS / Streamlit Cloud)
- 🧠 Multi-model support
- 🧪 Medical safety & validation layer

---

🤝 Contributing

- Contributions are welcome!
- Fork the repo
- Create a feature branch
- Submit a pull request


---

### ⭐ Acknowledgements

- Streamlit
- LangChain
- FAISS
- Euri AI


---

### 👨‍💻 Author

Aman Husain
Data Scientist | GenAI Enthusiast | RAG & LLM Practitioner

--- 

⭐ If you found this project helpful, don’t forget to star the repository!

---

If you want next:
- 📊 **Architecture diagram**
- 🐳 **Dockerfile**
- ☁️ **Deployment guide**
- 🧪 **Medical safety guardrails**

Just tell me 👍


