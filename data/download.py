"""Download the MovingAI MAPF benchmark into data/movingai/ (git-ignored).

    python data/download.py

The URLs are the ones linked from https://movingai.com/benchmarks/mapf/.
If a download fails, check that page: the links may have moved.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://movingai.com/benchmarks/mapf/"
ARCHIVES = {
    "mapf-map.zip": "maps",
    "mapf-scen-random.zip": "scen-random",
    "mapf-scen-even.zip": "scen-even",
}
TARGET = Path(__file__).parent / "movingai"


def main() -> None:
    for archive, folder in ARCHIVES.items():
        destination = TARGET / folder
        if destination.exists() and any(destination.iterdir()):
            print(f"{folder}: already present, skipping")
            continue
        print(f"{folder}: downloading {BASE + archive}")
        with urllib.request.urlopen(BASE + archive) as response:
            payload = response.read()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            bundle.extractall(destination)
        count = sum(1 for path in destination.rglob("*") if path.is_file())
        print(f"{folder}: extracted {count} files")


if __name__ == "__main__":
    main()
