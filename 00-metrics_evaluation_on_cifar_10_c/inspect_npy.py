import argparse
import numpy as np
import os

def inspect_npy_file(file_path):
    """
    Loads and prints information about a .npy file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # Load with mmap_mode='r' to avoid loading the entire large file into memory
        data = np.load(file_path, mmap_mode='r')
        
        print(f"--- Inspection of {file_path} ---")
        print(f"Shape: {data.shape}")
        print(f"Data Type: {data.dtype}")
        
        # Calculate statistics carefully since it might be memory-mapped
        # If the file is extremely large, even these might be slow, but for 150MB it should generally be fine.
        # Alternatively, calculate stats on a subset.
        print("\nCalculating statistics (might take a moment for large files)...")
        # To avoid MemoryError on very large arrays, we sample or just print min/max if simple.
        # CIFAR-10-C files are 150MB, so basic stats are fine.
        subset = data[:1000] if len(data) > 1000 else data
        print("- For the first up to 1000 elements:")
        print(f"  Min:  {np.min(subset)}")
        print(f"  Max:  {np.max(subset)}")
        print(f"  Mean: {np.mean(subset):.4f}")
        print(f"  Std:  {np.std(subset):.4f}")
        
        print("\nFirst 5 entries (or fewer if small):")
        print(data[:5])
        print("-------------------------------")
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a .npy file without loading it entirely into memory.")
    parser.add_argument("file_path", type=str, help="Path to the .npy file")
    args = parser.parse_args()
    
    inspect_npy_file(args.file_path)
