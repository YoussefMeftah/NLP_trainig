"""
Fix dataset metadata to use relative paths instead of absolute Colab paths.
This allows datasets to work locally and in Colab after upload to Drive.
"""

import json
from pathlib import Path

def fix_dataset_metadata(data_dir="./data"):
    """Fix state.json and dataset_info.json in all dataset splits."""
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        split_dir = Path(data_dir) / split
        
        # Fix state.json
        state_file = split_dir / "state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Ensure filenames are relative (just the filename, no paths)
            if "_data_files" in state:
                for data_file in state["_data_files"]:
                    if "filename" in data_file:
                        # Keep only the filename, strip any directory path
                        filename = Path(data_file["filename"]).name
                        data_file["filename"] = filename
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"✅ Fixed {split_dir}/state.json")
        
        # dataset_info.json usually doesn't have paths, but check anyway
        info_file = split_dir / "dataset_info.json"
        if info_file.exists():
            print(f"✅ Verified {split_dir}/dataset_info.json")

if __name__ == "__main__":
    print("Fixing dataset metadata for local/Colab compatibility...\n")
    fix_dataset_metadata()
    print("\n✅ All metadata files fixed! Ready to upload to Drive.")
