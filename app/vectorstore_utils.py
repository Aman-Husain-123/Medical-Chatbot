from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Storing Data
def create_faiss_index(text):
    # Model which i am utilizing
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_texts(texts=text, embedding=embeddings)

# Retrieval info from vector db
def retrive_similar_documents(faiss_index, query, k=4):
    return faiss_index.similarity_search(query, k=k)