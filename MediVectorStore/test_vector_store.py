import unittest
import numpy as np
from vector_store_utils import VectorStore, create_faiss_index, retrieve_similar_documents


class TestVectorStore(unittest.TestCase):
    """Unit tests for VectorStore class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_docs = [
            "Patient has a history of hypertension and diabetes.",
            "Blood pressure readings show consistent elevation.",
            "Glucose levels are within normal range after medication.",
            "Patient reports chest pain and shortness of breath.",
            "ECG shows normal sinus rhythm.",
            "Patient is allergic to penicillin and sulfa drugs.",
            "Family history includes heart disease and stroke."
        ]
    
    def test_initialization(self):
        """Test VectorStore initialization"""
        vs = VectorStore()
        self.assertIsNotNone(vs.model)
        self.assertGreater(vs.dimension, 0)
        self.assertEqual(len(vs.documents), 0)
        print("✓ Test initialization passed")
    
    def test_create_index(self):
        """Test FAISS index creation"""
        vs = VectorStore()
        vs.create_index()
        self.assertIsNotNone(vs.index)
        self.assertEqual(vs.index.ntotal, 0)  # No documents added yet
        print("✓ Test create_index passed")
    
    def test_add_documents(self):
        """Test adding documents to the index"""
        vs = VectorStore()
        vs.add_documents(self.sample_docs)
        
        self.assertEqual(len(vs.documents), len(self.sample_docs))
        self.assertEqual(vs.index.ntotal, len(self.sample_docs))
        print("✓ Test add_documents passed")
    
    def test_similarity_search(self):
        """Test similarity search functionality"""
        vs = VectorStore()
        vs.add_documents(self.sample_docs)
        
        # Search for cardiovascular symptoms
        query = "What are the patient's heart-related symptoms?"
        results = vs.similarity_search(query, k=3)
        
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], tuple)
        self.assertIsInstance(results[0][0], str)  # Document text
        self.assertIsInstance(results[0][1], float)  # Distance
        
        # Check that results are relevant (should include chest pain and ECG)
        result_texts = [doc for doc, _ in results]
        self.assertTrue(
            any("chest pain" in doc.lower() or "ecg" in doc.lower() 
                for doc in result_texts)
        )
        print("✓ Test similarity_search passed")
    
    def test_similarity_search_empty_index(self):
        """Test similarity search on empty index"""
        vs = VectorStore()
        results = vs.similarity_search("test query", k=3)
        
        self.assertEqual(len(results), 0)
        print("✓ Test similarity_search_empty_index passed")
    
    def test_get_index_stats(self):
        """Test getting index statistics"""
        vs = VectorStore()
        vs.add_documents(self.sample_docs)
        
        stats = vs.get_index_stats()
        self.assertEqual(stats['total_documents'], len(self.sample_docs))
        self.assertGreater(stats['dimension'], 0)
        self.assertEqual(stats['index_size'], len(self.sample_docs))
        print("✓ Test get_index_stats passed")
    
    def test_create_faiss_index_convenience(self):
        """Test convenience function create_faiss_index"""
        vs = create_faiss_index(self.sample_docs)
        
        self.assertIsInstance(vs, VectorStore)
        self.assertEqual(len(vs.documents), len(self.sample_docs))
        print("✓ Test create_faiss_index convenience function passed")
    
    def test_retrieve_similar_documents_convenience(self):
        """Test convenience function retrieve_similar_documents"""
        vs = create_faiss_index(self.sample_docs)
        results = retrieve_similar_documents(vs, "diabetes symptoms", k=2)
        
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], str)
        print("✓ Test retrieve_similar_documents convenience function passed")
    
    def test_multiple_searches(self):
        """Test multiple consecutive searches"""
        vs = VectorStore()
        vs.add_documents(self.sample_docs)
        
        queries = [
            "blood pressure",
            "allergies",
            "family medical history"
        ]
        
        for query in queries:
            results = vs.similarity_search(query, k=2)
            self.assertGreater(len(results), 0)
            self.assertLessEqual(len(results), 2)
        
        print("✓ Test multiple_searches passed")
    
    def test_relevance_ordering(self):
        """Test that results are ordered by relevance (distance)"""
        vs = VectorStore()
        vs.add_documents(self.sample_docs)
        
        query = "high blood pressure"
        results = vs.similarity_search(query, k=5)
        
        # Check that distances are in ascending order (lower = more similar)
        distances = [dist for _, dist in results]
        self.assertEqual(distances, sorted(distances))
        print("✓ Test relevance_ordering passed")


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 60)
    print("Running VectorStore Unit Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVectorStore)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
