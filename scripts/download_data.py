import os
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


def download_and_extract_dataset(url: str, dataset_name: str, target_dir_name: str):
    """Download and extract a dataset from a zip URL."""
    
    # Create assets directory if it doesn't exist
    assets_dir = Path("data/raw")
    assets_dir.mkdir(exist_ok=True)
    
    # Download the zip file
    print(f"Downloading {dataset_name} dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    if total_size > 1024 * 1024:
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    elif total_size > 1024 * 1024 * 1024:
        print(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    else:
        print(f"Total size: {total_size} bytes")
    
    # Save the zip file temporarily with progress bar
    zip_path = assets_dir / f"{dataset_name.lower()}_master.zip"
    with open(zip_path, "wb") as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading", unit_divisor=1024) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    # Extract the zip file with progress bar
    print(f"Extracting {dataset_name} dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(total=len(file_list), desc="Extracting") as pbar:
            for file in file_list:
                zip_ref.extract(file, assets_dir)
                pbar.update(1)
    
    # Find the extracted directory (first directory in the zip)
    extracted_dirs = [d for d in assets_dir.iterdir() if d.is_dir() and d.name.endswith('-master')]
    if not extracted_dirs:
        raise ValueError(f"Could not find extracted directory for {dataset_name}")
    
    extracted_dir = extracted_dirs[0]
    target_dir = assets_dir / target_dir_name
    
    # Remove existing target directory if it exists
    if target_dir.exists():
        shutil.rmtree(target_dir)
    
    # Rename the extracted directory
    extracted_dir.rename(target_dir)
    
    # Clean up the zip file
    zip_path.unlink()
    
    print(f"{dataset_name} dataset downloaded and extracted to {target_dir}")


def download_esc50_data():
    """Download and extract ESC-50 dataset from GitHub."""
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    download_and_extract_dataset(url, "ESC-50", "esc50")


def download_urbansound8k_data():
    """Download and extract UrbanSound8K dataset."""
    url = "https://goo.gl/8hY5ER"
    download_and_extract_dataset(url, "UrbanSound8K", "UrbanSound8K")


if __name__ == "__main__":
    print("This script will download the ESC-50 and UrbanSound8K datasets.")
    print("ESC-50: Environmental Sound Classification dataset")
    print("UrbanSound8K: Urban sound dataset with 8,732 labeled sound excerpts")
    print()
    
    esc50_response = input("Do you want to proceed with downloading ESC-50? (y/n): ").lower().strip()
    print()
    urbansound8k_response = input("Do you want to proceed with downloading UrbanSound8K? (y/n): ").lower().strip()
    print()
    
    if esc50_response in ['y', 'yes']:
        download_esc50_data()
    if urbansound8k_response in ['y', 'yes']:
        download_urbansound8k_data()
