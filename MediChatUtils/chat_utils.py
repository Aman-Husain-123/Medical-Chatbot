import time
import logging
from typing import Optional, Callable, Any
from functools import wraps
from euriai.langchain import create_chat_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MODEL = "gpt-4.1-nano"
DEFAULT_TEMPERATURE = 0.7
MAX_RETRIES = 3
INITIAL_DELAY = 1  # seconds
MAX_DELAY = 16  # seconds


def retry_with_backoff(max_retries: int = MAX_RETRIES, initial_delay: float = INITIAL_DELAY):
    """
    Decorator to retry a function with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {str(e)}")
                        raise
                    
                    # Log the error and retry
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {str(e)}. "
                        f"Retrying in {delay} seconds..."
                    )
                    
                    time.sleep(delay)
                    
                    # Exponential backoff with cap
                    delay = min(delay * 2, MAX_DELAY)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def create_prompt(query: str, context: str) -> str:
    """
    Create a structured prompt combining the query and context.
    
    Args:
        query: User's question
        context: Retrieved context from vector database
        
    Returns:
        Formatted prompt string
    """
    prompt = f"""You are MediChat Pro, an intelligent medical document assistant.

Based on the following medical documents, answer the user's question accurately and concisely.
If the answer is not present in the provided context, clearly state that you don't have enough information.

Medical Documents Context:
{context}

User Question:
{query}

Answer:"""
    
    return prompt


@retry_with_backoff(max_retries=MAX_RETRIES, initial_delay=INITIAL_DELAY)
def _call_chat_model(chat_model, prompt: str) -> str:
    """
    Internal function to call the chat model with retry logic.
    
    Args:
        chat_model: Initialized chat model instance
        prompt: Formatted prompt string
        
    Returns:
        Model's response as string
    """
    try:
        response = chat_model.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error calling chat model: {str(e)}")
        raise


def get_chat_response(
    query: str,
    context: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE
) -> Optional[str]:
    """
    Get a chat response from the EURI AI model.
    
    This function combines the user query with retrieved context, sends it to the
    chat model, and returns the response. It includes error handling and automatic
    retry with exponential backoff.
    
    Args:
        query: User's question
        context: Retrieved context from vector database
        api_key: EURI AI API key
        model: Model name to use (default: gpt-4.1-nano)
        temperature: Model temperature (default: 0.7)
        
    Returns:
        Model's response as string, or None if all retries fail
        
    Raises:
        ValueError: If query or context is empty
        Exception: If API call fails after all retries
    """
    # Validation
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    
    if not context or not context.strip():
        logger.warning("Context is empty. Proceeding with query only.")
        context = "No relevant context found."
    
    if not api_key:
        raise ValueError("API key is required")
    
    try:
        logger.info(f"Processing query: {query[:100]}...")
        
        # Create the chat model
        chat_model = create_chat_model(
            api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Create the prompt
        prompt = create_prompt(query, context)
        logger.debug(f"Prompt created (length: {len(prompt)} chars)")
        
        # Call the model with retry logic
        response = _call_chat_model(chat_model, prompt)
        
        logger.info(f"Successfully received response (length: {len(response)} chars)")
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_chat_response: {str(e)}")
        raise


# Convenience function for backward compatibility
def get_chat_model(api_key: str, model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
    """
    Create and return a chat model instance.
    
    Args:
        api_key: EURI AI API key
        model: Model name to use
        temperature: Model temperature
        
    Returns:
        Initialized chat model
    """
    return create_chat_model(
        api_key=api_key,
        model=model,
        temperature=temperature
    )


def ask_chat_model(chat_model, prompt: str) -> str:
    """
    Ask the chat model a question with retry logic.
    
    Args:
        chat_model: Initialized chat model instance
        prompt: Prompt string
        
    Returns:
        Model's response
    """
    return _call_chat_model(chat_model, prompt)


if __name__ == "__main__":
    # Quick test (requires valid API key)
    print("Testing chat_utils module...")
    
    # Sample data
    test_query = "What are the patient's symptoms?"
    test_context = """
    Patient reports chest pain and shortness of breath.
    Blood pressure readings show consistent elevation.
    ECG shows normal sinus rhythm.
    """
    
    # Note: Replace with actual API key for testing
    test_api_key = "your-api-key-here"
    
    print(f"\nQuery: {test_query}")
    print(f"Context: {test_context[:100]}...")
    
    try:
        # This will fail without a valid API key
        response = get_chat_response(test_query, test_context, test_api_key)
        print(f"\nResponse: {response}")
    except Exception as e:
        print(f"\nTest failed (expected without valid API key): {str(e)}")
        print("\nTo test with a real API key, replace 'your-api-key-here' with your actual EURI AI API key.")
