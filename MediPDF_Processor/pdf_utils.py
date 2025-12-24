import logging
from pypdf import PdfReader
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_file):
    """
    Extracts text content from a PDF file.
    
    Args:
        pdf_file: Path to the PDF file or a file-like object.
        
    Returns:
        str: Extracted text content or an empty string if extraction fails.
    """
    text = ""
    try:
        # Check if pdf_file is a path and if it exists
        if isinstance(pdf_file, str) and not os.path.exists(pdf_file):
            logger.error(f"File not found: {pdf_file}")
            return ""

        reader = PdfReader(pdf_file)
        
        # Iterate through pages and extract text
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                logger.warning(f"No text extracted from page {i+1} of {pdf_file}")
                
        logger.info(f"Successfully extracted text from {pdf_file}")
        
    except Exception as e:
        logger.error(f"An error occurred while parsing PDF {pdf_file}: {str(e)}")
        # Return empty string as a fallback
        return ""
        
    return text.strip()

if __name__ == "__main__":
    # Quick internal test
    sample_pdf = "medical_history_1.pdf"
    if os.path.exists(sample_pdf):
        print("Testing extraction...")
        content = extract_text_from_pdf(sample_pdf)
        print(f"Extracted {len(content)} characters.")
        if content:
            print("Preview:", content[:100], "...")
    else:
        print(f"Sample PDF {sample_pdf} not found.")
