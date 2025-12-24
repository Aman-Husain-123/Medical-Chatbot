from pdf_utils import extract_text_from_pdf
import os

def run_tests():
    print("--- Running PDF Extraction Tests ---")
    
    # Test 1: Valid PDF
    valid_pdf = "medical_history_1.pdf"
    print(f"\nTest 1: Valid PDF ({valid_pdf})")
    if os.path.exists(valid_pdf):
        text = extract_text_from_pdf(valid_pdf)
        if text:
            print(f"SUCCESS: Extracted {len(text)} characters.")
            print(f"Preview: {text[:150]}...")
        else:
            print("FAILURE: No text extracted from valid PDF.")
    else:
        print(f"SKIPPED: {valid_pdf} not found.")

    # Test 2: Non-existent file
    invalid_path = "non_existent.pdf"
    print(f"\nTest 2: Non-existent file ({invalid_path})")
    text = extract_text_from_pdf(invalid_path)
    if text == "":
        print("SUCCESS: Correctly handled missing file.")
    else:
        print("FAILURE: Did not handle missing file correctly.")

    # Test 3: Invalid file format (not a PDF)
    txt_file = "test.txt"
    with open(txt_file, "w") as f:
        f.write("This is not a PDF.")
        
    print(f"\nTest 3: Invalid file format ({txt_file})")
    text = extract_text_from_pdf(txt_file)
    if text == "":
        print("SUCCESS: Correctly handled invalid file format.")
    else:
        print("FAILURE: Did not handle invalid file format correctly.")
    
    # Cleanup
    if os.path.exists(txt_file):
        os.remove(txt_file)

if __name__ == "__main__":
    run_tests()
