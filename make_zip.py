import os
import zipfile

PROJECT_DIR = "local-RAG"          # folder name of your project
OUTPUT_ZIP = "local-RAG.zip"

# Folders/files to skip (large, private, or regenerable)
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache",
                "node_modules", ".idea", ".vscode", "runs", "logs"}
EXCLUDE_EXT = {".pyc", ".pdf", ".faiss", ".enc", ".sqlite", ".db",
               ".bin", ".safetensors", ".pt", ".gguf"}
MAX_FILE_MB = 5                    # skip any single file bigger than this

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            path = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            if ext in EXCLUDE_EXT:
                continue
            if os.path.getsize(path) > MAX_FILE_MB * 1024 * 1024:
                print("Skipped (too large):", path)
                continue
            zf.write(path, os.path.relpath(path, os.path.dirname(PROJECT_DIR) or "."))

size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
print(f"Created {OUTPUT_ZIP} ({size_mb:.2f} MB)")