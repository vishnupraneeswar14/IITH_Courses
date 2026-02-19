import os
import sys
import pickle
import numpy as np

# --- Setup ---
# Add current directory to path to find my_ml_lib and create_best_model
sys.path.insert(0, os.path.abspath(os.getcwd()))

# Color codes for output
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"
INFO = "\033[94mINFO\033[0m"

print("="*60)
print("--- MyML Assignment 3: P5 Submission Verifier ---")
print("="*60)
print(f"Running in: {os.getcwd()}")
print("This script will check if your best model is loadable by the autograder.\n")

model_loaded_successfully = False
final_summary = []

try:
    # --- 1. Check for required files ---
    print(f"{INFO} Step 1: Checking for required files...")
    assert os.path.exists("create_best_model.py"), "'create_best_model.py' not found."
    print(f"  {PASS} Found 'create_best_model.py'.")
    
    model_npz_path = os.path.join("saved_models", "best_model.npz")
    model_pkl_path = os.path.join("saved_models", "best_model.pkl")
    assert os.path.exists(model_npz_path) or os.path.exists(model_pkl_path), \
           "'best_model.npz' or 'best_model.pkl' not found in 'saved_models/'."
    
    saved_model_path = model_npz_path if os.path.exists(model_npz_path) else model_pkl_path
    print(f"  {PASS} Found '{saved_model_path}'.")
    final_summary.append(f"{PASS} Required files found.")
    
    # --- 2. Load the model architecture ---
    print(f"\n{INFO} Step 2: Loading model architecture...")
    
    # This imports the student's file
    from create_best_model import initialize_best_model 
    
    model_p5 = initialize_best_model()
    assert model_p5 is not None, "'initialize_best_model' returned None."
    print(f"  {PASS} Successfully initialized model from 'create_best_model.py'.")
    print(f"    {INFO} Model Architecture Initialized:")
    print("-" * 20)
    print(model_p5) # Print the model's __repr__
    print("-" * 20)
    final_summary.append(f"{PASS} Model architecture initialized.")

    # --- 3. Load the saved weights/parameters ---
    print(f"\n{INFO} Step 3: Loading saved model weights/parameters...")
    
    if os.path.exists(model_npz_path) and hasattr(model_p5, 'load_state_dict'):
        print(f"  Attempting to load state dict from '{model_npz_path}'...")
        # This will print the Warnings from the student's load_state_dict
        model_p5.load_state_dict(model_npz_path) 
        print(f"  {PASS} 'load_state_dict' executed without error.")
        print(f"  {WARN} -----------------------------------------------------------------")
        print(f"  {WARN} CRITICAL: Read the output above this line carefully!")
        print(f"  {WARN} If you see 'Warning: Missing keys' OR 'Warning: Unexpected keys',")
        print(f"  {WARN} your model architecture in 'create_best_model.py' DOES NOT MATCH")
        print(f"  {WARN} the model you saved! This will result in ~10% accuracy.")
        print(f"  {WARN} -----------------------------------------------------------------")
        final_summary.append(f"{PASS} Autograd model loaded (check warnings!).")

    elif os.path.exists(model_pkl_path):
        print(f"  Attempting to load pickle file from '{model_pkl_path}'...")
        with open(model_pkl_path, 'rb') as f:
            model_p5 = pickle.load(f) # OvR model replaces the instance
        print(f"  {PASS} Successfully loaded pickled model from 'best_model.pkl'.")
        final_summary.append(f"{PASS} OvR model loaded.")
    
    else:
        # This case should be caught by Step 1, but as a safeguard
        raise FileNotFoundError("Could not find matching saved model for initialized architecture.")
    
    model_loaded_successfully = True

except AssertionError as e:
    print(f"\n  {FAIL} {e}")
    final_summary.append(f"{FAIL} {e}")
except ImportError as e:
    print(f"\n  {FAIL} Could not import from 'create_best_model.py' or 'my_ml_lib': {e}")
    print(f"  {WARN} Make sure your 'my_ml_lib' is importable and 'create_best_model.py' is correct.")
    final_summary.append(f"{FAIL} Import error.")
except Exception as e:
    print(f"\n  {FAIL} An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    final_summary.append(f"{FAIL} Runtime error.")

# --- Final Summary ---
print("\n" + "="*60)
print("--- Verification Summary ---")
for line in final_summary:
    print(line)

if model_loaded_successfully:
    print(f"\n{PASS} Verification script finished successfully.")
    print(f"{INFO} Your model appears to be loadable by the autograder.")
    print(f"{WARN} PLEASE DOUBLE-CHECK FOR KEY MISMATCH WARNINGS IN STEP 3!")
else:
    print(f"\n{FAIL} Verification script failed.")
    print(f"{INFO} Your model is NOT loadable by the autograder. Please fix the errors above.")
print("="*60)