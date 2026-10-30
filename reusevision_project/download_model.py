"""Download public, fine-tuned YOLO weights for motherboard hardware-part detection."""
from pathlib import Path
import sys

from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "cvtechniques/Motherboard-Part-Locator"
ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
TARGET = WEIGHTS_DIR / "best.pt"


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"Weights already present: {TARGET}")
        return
    print(f"Checking public model repository: {REPO_ID}")
    try:
        files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="model")
        candidates = [name for name in files if name.lower().endswith(".pt")]
        if not candidates:
            raise RuntimeError("No .pt YOLO checkpoint was found in the public model repository.")
        candidates.sort(key=lambda name: ("best" not in Path(name).name.lower(), len(name), name.lower()))
        chosen = candidates[0]
        print(f"Downloading checkpoint: {chosen}")
        downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=chosen, repo_type="model"))
        if downloaded.stat().st_size < 1_000_000:
            raise RuntimeError("Downloaded checkpoint is unexpectedly small; refusing to install it.")
        TARGET.write_bytes(downloaded.read_bytes())
        print(f"Model weights saved to: {TARGET}")
        print("Model class scope: CPU Socket, PCIe Slot, Power Connector, and RAM Slot (as listed on the model card).")
        print("Note: this model detects motherboard parts/slots, not full CPU/GPU/RAM-stick/HDD/SSD/PSU objects.")
    except Exception as exc:
        print("Model download failed. Check your internet connection and try again.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
