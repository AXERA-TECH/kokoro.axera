import json
import sys
import os
from pathlib import Path

# Add src to sys.path to import kokoro_onnx
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from kokoro_onnx.config import get_vocab
except ImportError:
    print("Error: Could not import kokoro_onnx.config. Make sure you are in the root of the repo.")
    sys.exit(1)

def export_vocab(output_path):
    try:
        vocab = get_vocab()
    except Exception as e:
        print(f"Error loading vocab: {e}")
        return

    print(f"Loaded {len(vocab)} tokens.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, idx in vocab.items():
            # Escape special characters to keep the file parseable line by line
            # We use tab as separator
            safe_token = token.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            f.write(f"{safe_token}\t{idx}\n")
    
    print(f"Exported vocab to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_vocab.py <output_vocab.txt>")
        sys.exit(1)
    
    export_vocab(sys.argv[1])
