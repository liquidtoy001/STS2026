"""Download the eight comparison utterances from a revision-pinned public mirror.

Keep the complete original metadata file for compatibility with the Kaggle run.
Audio/data remain under ignored data/. Downloaded bytes are checked against the
mirror's LFS SHA256, not claimed to be independently checked against upstream.
"""

import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
import wave


REPOSITORY = "flexthink/ljspeech"
IDS = [f"LJ001-{i:04d}" for i in range(1, 9)]


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def main():
    destination = Path(__file__).resolve().parents[1] / "data/LJSpeech-1.1"
    destination.mkdir(parents=True, exist_ok=True)
    info = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPOSITORY}"))
    revision = info["sha"]
    paths = ["metadata.csv"] + [f"wavs/{name}.wav" for name in IDS]
    records = []
    for relative in paths:
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{relative}"
        directory = relative.rsplit("/", 1)[0] if "/" in relative else ""
        listing = json.loads(fetch(f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/{directory}"))
        entry = next(row for row in listing if row["path"] == relative)
        expected = entry["lfs"]["oid"]
        target = destination / relative
        data = target.read_bytes() if target.exists() else fetch(url)
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise ValueError(f"SHA256 mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(data)
        record = {"path": relative, "sha256": digest, "bytes": len(data), "url": url}
        if relative.endswith(".wav"):
            with wave.open(str(target)) as audio:
                if (audio.getnchannels(), audio.getframerate(), audio.getsampwidth()) != (1, 22050, 2):
                    raise ValueError(f"Unexpected audio format: {relative}")
                record["samples"] = audio.getnframes()
                record["seconds"] = audio.getnframes() / audio.getframerate()
        records.append(record)
        print(relative, len(data), digest, flush=True)
    rows = {line.split("|", 1)[0] for line in (destination / "metadata.csv").read_text(encoding="utf-8").splitlines()}
    if not set(IDS).issubset(rows):
        raise ValueError("Fixed IDs missing from metadata.")
    (destination / "download_manifest.json").write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "completion_time_source": "runtime_clock",
        "upstream": "https://keithito.com/LJ-Speech-Dataset/",
        "mirror": REPOSITORY, "revision": revision, "files": records,
    }, indent=2), encoding="utf-8")
    print("Ready:", destination)


if __name__ == "__main__":
    main()
