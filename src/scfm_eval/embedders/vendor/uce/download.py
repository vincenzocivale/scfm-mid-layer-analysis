"""Figshare download utility from UCE/utils.py."""
import os
import tarfile

import requests
from tqdm import tqdm


def figshare_download(url: str, save_path: str) -> None:
    """Download a file from figshare if it doesn't already exist."""
    if os.path.exists(save_path):
        return
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"Downloading {save_path} from {url} ...")
    response = requests.get(url, stream=True)
    total_bytes = int(response.headers.get("content-length", 0))
    block = 1024
    bar = tqdm(total=total_bytes, unit="iB", unit_scale=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(block):
            bar.update(len(chunk))
            f.write(chunk)
    bar.close()
    if save_path.endswith(".tar.gz"):
        with tarfile.open(save_path) as tar:
            tar.extractall(path=os.path.dirname(save_path))
        print("Extracted archive.")
