import json
import time
import os
import sys

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import EURI_API_KEY
from app.chat_utils import get_chat_model, ask_chat_model
from app.pdf_utils import extract_text_from_pdf
from app.vectorstore_utils import create_faiss_index, retrive_similar_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_dataset(path):
    with open(path, 'r') as f:
        return json.load(f)

def setup_system(pdf_path):
    print(f"Setting up system with {pdf_path}...")
    start_time = time.time()
    
    # Extract
    text = extract_text_from_pdf(pdf_path)
    
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    # Index
    vectorstore = create_faiss_index(chunks)
    
    # Chat Model
    chat_model = get_chat_model(EURI_API_KEY)
    
    setup_time = time.time() - start_time
    print(f"Setup complete in {setup_time:.2f}s")
    
    return vectorstore, chat_model

def evaluate(dataset_path, pdf_path):
    dataset = load_dataset(dataset_path)
    vectorstore, chat_model = setup_system(pdf_path)
    
    results = []
    
    print("\nStarting evaluation...")
    for item in dataset:
        query = item['question']
        print(f"Evaluating: {query}")
        
        # Measure Retrieval + Generation
        t0 = time.time()
        
        # Retrieve
        relevant_docs = retrive_similar_documents(vectorstore, query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Construct Prompt
        system_prompt = f"""Based on the following medical documents, answer accurately.
If the answer is not present, clearly say so.

Medical Documents:
{context}

User Question:
{query}

Answer:"""

        # Generate
        if chat_model:
            try:
                # We use invoke directly or ask_chat_model
                response = chat_model.invoke(system_prompt).content
            except Exception as e:
                response = f"Error: {e}"
        else:
            response = "No model available"
            
        latency = time.time() - t0
        
        # Simple Keyword Metric
        keywords = item.get('keywords', [])
        hit_count = sum(1 for k in keywords if k.lower() in response.lower())
        keyword_recall = hit_count / len(keywords) if keywords else 0
        
        result_item = {
            "id": item['id'],
            "query": query,
            "response": response,
            "latency": latency,
            "keyword_recall": keyword_recall,
            "ground_truth": item['ground_truth']
        }
        results.append(result_item)
        print(f"  Latency: {latency:.2f}s | Keyword Recall: {keyword_recall:.2f}")

    # Save results
    os.makedirs('evaluation/results', exist_ok=True)
    with open('evaluation/results/eval_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # Calculate averages
    avg_latency = sum(r['latency'] for r in results) / len(results)
    avg_recall = sum(r['keyword_recall'] for r in results) / len(results)
    
    print("\nEvaluation Summary:")
    print(f"Average Latency: {avg_latency:.2f}s")
    print(f"Average Keyword Recall: {avg_recall:.2f}")

if __name__ == "__main__":
    # Assuming we run this from the project root or evaluation folder
    # Adjust paths accordingly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_file = os.path.join(base_dir, "evaluation", "evaluation_dataset.json")
    # Using the sample PDF we moved earlier
    pdf_file = os.path.join(base_dir, "MediChatbot", "MediPDF_Processor", "medical_history_1.pdf")
    
    # Check if PDF exists there, if not try another location
    if not os.path.exists(pdf_file):
        # Fallback to local sample_data if distinct
        pdf_file = os.path.join(base_dir, "MediChatbot", "sample_data", "medical_history_1.pdf")
        
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file not found at {pdf_file}")
        # Try to find it in the current dir structure
        # The user has: c:\Users\user\OneDrive\Documents\Generative_AI\MediChatbot\MediPDF_Processor\medical_history_1.pdf
        # My base_dir calculation might be slightly off depending on where I run it.
        # Let's hardcode the probable path for resilience
        pdf_file = r"c:\Users\user\OneDrive\Documents\Generative_AI\MediChatbot\MediPDF_Processor\medical_history_1.pdf"

    evaluate(dataset_file, pdf_file)
