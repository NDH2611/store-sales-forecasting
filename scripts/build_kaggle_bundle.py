"""Build only first-party source/configs; no credentials/data/weights/dependencies."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root/"outputs"/"store-sales-source.zip"
target.parent.mkdir(exist_ok=True)
files = [root/"pyproject.toml"] + sorted((root/"src").rglob("*.py")) + sorted((root/"configs").glob("*.json"))
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                  stderr=subprocess.DEVNULL).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
except (OSError, subprocess.CalledProcessError):
    sha, dirty = "unavailable", True
provenance = {"git_sha": sha, "git_dirty": dirty, "files_sha256": {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file in files:
        archive.write(file, str(file.relative_to(root)))
    archive.writestr("SOURCE_PROVENANCE.json", json.dumps(provenance, indent=2))
print(target)
