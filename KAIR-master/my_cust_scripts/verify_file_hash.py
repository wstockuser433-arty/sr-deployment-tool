import hashlib
import os

def calculate_md5(file_path, chunk_size=1024*1024):
    """
    Computes MD5 hash in 1MB chunks to be memory efficient.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")

    md5_hasher = hashlib.md5()
    
    # Get total size for manual progress tracking
    file_size = os.path.getsize(file_path)
    processed = 0
    
    print(f"Reading file: {file_path} ({file_size / (1024**3):.2f} GB)")
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            md5_hasher.update(chunk)
            processed += len(chunk)
            # Print progress every 10GB to avoid spamming the console
            if (processed // (10*1024**3)) > ((processed - len(chunk)) // (1024**3)):
                print(f"Progress: {processed / (10*1024**3):.1f} GB processed...")
                
    return md5_hasher.hexdigest()

# def main():
#     # --- CONFIGURATION ---
#     # Replace these with your actual details
#     FILE_PATH = ''
#     EXPECTED_HASH = 'your_expected_md5_hash_string_here'
#     # ---------------------

#     try:
#         calculated = calculate_md5(FILE_PATH)
        
#         print("-" * 30)
#         print(f"Calculated Hash: {calculated}")
#         print(f"Expected Hash:   {EXPECTED_HASH}")
#         print("-" * 30)

#         if calculated.lower() == EXPECTED_HASH.lower():
#             print("SUCCESS: The file is valid and matches the expected hash.")
#         else:
#             print("FAILURE: The hash does NOT match. The file might be corrupted.")

#     except Exception as e:
#         print(f"An error occurred: {e}")

import glob

def main():
    # Folder containing your zip files
    SEARCH_PATH = '/home/hassan/Documents/Hassan/SR Enhancement Work/Datasets/WorldStrat/worldstrat_datafolder/*.zip'

    # Dictionary mapping filenames to their expected hashes
    # You can also just manually check them one by one if you don't have a list
    FILE_CHECKS = {
        'lr_dataset_l2a.zip':'7aa1878a37d22a6c7c4b84b022a14ad7',
        'metadata.csv': '1a66ac42b9a688be18debd0d95633fa1',
        'stratified_train_val_test_split.csv':'874612b59bbf7987f7de7edd48a30c70'
    }

    for file_path in glob.glob(SEARCH_PATH):
        filename = os.path.basename(file_path)
        print(f"\n--- Processing: {filename} ---")
        print(file_path)
        calculated = calculate_md5(file_path)
        expected = FILE_CHECKS.get(filename, "Unknown")
        
        print(f"Calculated: {calculated}")
        print(f"Expected:   {expected}")
        
        if calculated.lower() == expected.lower():
            print("Status: MATCH")
        else:
            print("Status: MISMATCH")

if __name__ == "__main__":
    main()