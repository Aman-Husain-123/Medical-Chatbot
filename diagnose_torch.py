import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print("Torch imported successfully!")
except Exception as e:
    print("\n--- ERROR IMPORTING TORCH ---")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    import traceback
    traceback.print_exc()

try:
    import numpy
    print(f"\nNumpy version: {numpy.__version__}")
except Exception as e:
    print(f"\nError importing numpy: {e}")

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
except Exception as e:
    print(f"Error importing transformers: {e}")
