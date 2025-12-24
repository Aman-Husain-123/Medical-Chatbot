import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from chat_utils import (
    get_chat_response,
    create_prompt,
    retry_with_backoff,
    get_chat_model,
    ask_chat_model,
    _call_chat_model
)


class TestChatUtils(unittest.TestCase):
    """Unit tests for chat_utils module"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_query = "What are the patient's symptoms?"
        self.test_context = "Patient reports chest pain and shortness of breath."
        self.test_api_key = "test-api-key-123"
    
    def test_create_prompt(self):
        """Test prompt creation"""
        prompt = create_prompt(self.test_query, self.test_context)
        
        self.assertIn(self.test_query, prompt)
        self.assertIn(self.test_context, prompt)
        self.assertIn("MediChat Pro", prompt)
        self.assertIn("Medical Documents Context:", prompt)
        print("✓ Test create_prompt passed")
    
    def test_create_prompt_formatting(self):
        """Test that prompt is properly formatted"""
        prompt = create_prompt(self.test_query, self.test_context)
        
        # Check structure
        self.assertIn("User Question:", prompt)
        self.assertIn("Answer:", prompt)
        print("✓ Test create_prompt_formatting passed")
    
    @patch('chat_utils.create_chat_model')
    def test_get_chat_response_success(self, mock_create_model):
        """Test successful chat response"""
        # Mock the chat model and response
        mock_response = Mock()
        mock_response.content = "The patient has chest pain and shortness of breath."
        
        mock_model = Mock()
        mock_model.invoke.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Call the function
        response = get_chat_response(
            self.test_query,
            self.test_context,
            self.test_api_key
        )
        
        # Assertions
        self.assertEqual(response, "The patient has chest pain and shortness of breath.")
        mock_create_model.assert_called_once()
        mock_model.invoke.assert_called_once()
        print("✓ Test get_chat_response_success passed")
    
    def test_get_chat_response_empty_query(self):
        """Test that empty query raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_chat_response("", self.test_context, self.test_api_key)
        
        self.assertIn("Query cannot be empty", str(context.exception))
        print("✓ Test get_chat_response_empty_query passed")
    
    def test_get_chat_response_empty_api_key(self):
        """Test that empty API key raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_chat_response(self.test_query, self.test_context, "")
        
        self.assertIn("API key is required", str(context.exception))
        print("✓ Test get_chat_response_empty_api_key passed")
    
    @patch('chat_utils.create_chat_model')
    def test_get_chat_response_empty_context(self, mock_create_model):
        """Test handling of empty context"""
        mock_response = Mock()
        mock_response.content = "Response without context"
        
        mock_model = Mock()
        mock_model.invoke.return_value = mock_response
        mock_create_model.return_value = mock_model
        
        # Should not raise error, but log warning
        response = get_chat_response(
            self.test_query,
            "",  # Empty context
            self.test_api_key
        )
        
        self.assertIsNotNone(response)
        print("✓ Test get_chat_response_empty_context passed")
    
    def test_retry_decorator(self):
        """Test retry mechanism with exponential backoff"""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, initial_delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "Success"
        
        start_time = time.time()
        result = failing_function()
        elapsed_time = time.time() - start_time
        
        self.assertEqual(result, "Success")
        self.assertEqual(call_count, 3)
        # Should have delays: 0.1 + 0.2 = 0.3 seconds minimum
        self.assertGreater(elapsed_time, 0.25)
        print("✓ Test retry_decorator passed")
    
    def test_retry_decorator_max_retries(self):
        """Test that retry gives up after max retries"""
        call_count = 0
        
        @retry_with_backoff(max_retries=2, initial_delay=0.05)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent failure")
        
        with self.assertRaises(Exception) as context:
            always_failing_function()
        
        self.assertEqual(call_count, 3)  # Initial + 2 retries
        self.assertIn("Permanent failure", str(context.exception))
        print("✓ Test retry_decorator_max_retries passed")
    
    @patch('chat_utils.create_chat_model')
    def test_get_chat_model_function(self, mock_create_model):
        """Test get_chat_model convenience function"""
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        result = get_chat_model(self.test_api_key)
        
        mock_create_model.assert_called_once_with(
            api_key=self.test_api_key,
            model="gpt-4.1-nano",
            temperature=0.7
        )
        self.assertEqual(result, mock_model)
        print("✓ Test get_chat_model_function passed")
    
    @patch('chat_utils.create_chat_model')
    def test_ask_chat_model_function(self, mock_create_model):
        """Test ask_chat_model convenience function"""
        mock_response = Mock()
        mock_response.content = "Test response"
        
        mock_model = Mock()
        mock_model.invoke.return_value = mock_response
        
        result = ask_chat_model(mock_model, "Test prompt")
        
        self.assertEqual(result, "Test response")
        mock_model.invoke.assert_called_once_with("Test prompt")
        print("✓ Test ask_chat_model_function passed")
    
    @patch('chat_utils.create_chat_model')
    def test_api_error_handling(self, mock_create_model):
        """Test handling of API errors"""
        mock_model = Mock()
        mock_model.invoke.side_effect = Exception("API Error: Rate limit exceeded")
        mock_create_model.return_value = mock_model
        
        with self.assertRaises(Exception) as context:
            get_chat_response(self.test_query, self.test_context, self.test_api_key)
        
        # Should have retried 3 times (initial + 2 retries)
        self.assertEqual(mock_model.invoke.call_count, 3)
        print("✓ Test api_error_handling passed")


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 60)
    print("Running Chat Utils Unit Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestChatUtils)
    
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
