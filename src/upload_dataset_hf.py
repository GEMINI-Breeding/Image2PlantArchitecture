import argparse
import os
from huggingface_hub import HfApi, create_repo

def upload_dataset(dataset_path, repo_id, token, private=True):
    api = HfApi()
    
    print(f"Creating repository {repo_id}...")
    try:
        create_repo(repo_id, token=token, repo_type="dataset", private=private, exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    print(f"Uploading folder {dataset_path} to {repo_id}...")
    # We only upload images and xml folders to keep it clean
    for folder in ["images", "xml"]:
        folder_path = os.path.join(dataset_path, folder)
        if os.path.exists(folder_path):
            print(f"Uploading {folder}...")
            api.upload_folder(
                folder_path=folder_path,
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=folder,
                token=token
            )
        else:
            print(f"Warning: {folder} folder not found in {dataset_path}")

    # Create and upload README.md
    readme_content = """---
license: mit
task_categories:
- image-to-text
tags:
- biology
- plant-phenotyping
- synthetic-data
- cowpea
- plant-architecture
---

# Cowpea-Architecture-XML

This dataset contains simulated images of Cowpea plants paired with organ-level architecture representations in XML format. 

## Dataset Structure
- `images/`: Original plant images (.jpeg)
- `xml/`: Plant architecture annotations (.xml)

## Citation
If you use this dataset, please cite:
**"A Vision Language Model for Generating XML-based Organ-level Plant Architecture Representations of Cowpea from Simulated Images"**
"""
    readme_path = "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)
    
    print("Uploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token
    )
    os.remove(readme_path)

    print("Upload complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--dataset_path", type=str, 
                        default="",
                        help="Path to the dataset directory")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repository ID (e.g., 'username/dataset-name')")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face API token with write access")
    parser.add_argument("--public", action="store_true", help="Make the repository public (default: private)")

    args = parser.parse_args()

    upload_dataset(args.dataset_path, args.repo_id, args.token, private=not args.public)
