import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    A vector store implementation using FAISS for efficient similarity search.
    Uses sentence-transformers/all-mpnet-base-v2 for generating embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the VectorStore with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logger.info(f"Initializing VectorStore with model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.documents = []
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
    
    def create_index(self):
        """
        Create a FAISS index for storing document embeddings.
        Uses IndexFlatL2 for exact L2 distance search.
        """
        try:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"FAISS index created with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def add_documents(self, chunks: List[str]):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of text chunks to add to the index
        """
        if self.index is None:
            self.create_index()
        
        try:
            # Generate embeddings for all chunks
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.model.encode(chunks, show_progress_bar=True)
            
            # Convert to numpy array and ensure correct dtype
            embeddings = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store the original text chunks
            self.documents.extend(chunks)
            
            logger.info(f"Successfully added {len(chunks)} documents to the index")
            logger.info(f"Total documents in index: {len(self.documents)}")
            
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """
        Perform similarity search to retrieve top-k most similar documents.
        
        Args:
            query: Query string to search for
            k: Number of top results to return
            
        Returns:
            List of tuples containing (document_text, distance)
        """
        if self.index is None or len(self.documents) == 0:
            logger.warning("Index is empty. No documents to search.")
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self.model.encode([query]).astype('float32')
            
            # Perform search
            distances, indices = self.index.search(query_embedding, k)
            
            # Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(distance)))
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def get_index_stats(self) -> dict:
        """
        Get statistics about the current index.
        
        Returns:
            Dictionary containing index statistics
        """
        return {
            "total_documents": len(self.documents),
            "dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0
        }


# Convenience functions for backward compatibility
def create_faiss_index(chunks: List[str]) -> VectorStore:
    """
    Create a FAISS index from text chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        VectorStore instance with indexed documents
    """
    vector_store = VectorStore()
    vector_store.add_documents(chunks)
    return vector_store


def retrieve_similar_documents(vector_store: VectorStore, query: str, k: int = 4) -> List[str]:
    """
    Retrieve similar documents from the vector store.
    
    Args:
        vector_store: VectorStore instance
        query: Query string
        k: Number of results to return
        
    Returns:
        List of similar document texts
    """
    results = vector_store.similarity_search(query, k)
    return [doc for doc, _ in results]


if __name__ == "__main__":
    # Quick test
    print("Testing VectorStore...")
    
    # Sample medical documents
    sample_docs = [
        "Patient has a history of hypertension and diabetes.",
        "Blood pressure readings show consistent elevation.",
        "Glucose levels are within normal range after medication.",
        "Patient reports chest pain and shortness of breath.",
        "ECG shows normal sinus rhythm."
    ]
    
    # Create vector store and add documents
    vs = VectorStore()
    vs.add_documents(sample_docs)
    
    # Test similarity search
    query = "What are the patient's cardiovascular symptoms?"
    results = vs.similarity_search(query, k=3)
    
    print(f"\nQuery: {query}")
    print("\nTop 3 similar documents:")
    for i, (doc, distance) in enumerate(results, 1):
        print(f"{i}. (Distance: {distance:.4f}) {doc}")
    
    # Print stats
    print(f"\nIndex stats: {vs.get_index_stats()}")
