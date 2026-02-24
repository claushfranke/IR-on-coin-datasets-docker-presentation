"""
Download-Script für ML-Modelle von Google Drive.
Wird beim ersten Container-Start automatisch ausgeführt.
"""

import os
import sys
import zipfile
import logging
import requests

logger = logging.getLogger(__name__)

MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Google Drive File-IDs
DOWNLOADS = [
    {
        "name": "Typ-Modell (vgg16_types_95.keras)",
        "file_id": "1m0N1pqY1mF50_XykAZsvk8lJkyXfKlhw",
        "dest": "vgg16_types.zip",
        "is_zip": True,
        "check_file": "vgg16_types_95.keras",
    },
    {
        "name": "Prägestätten-Modell (vgg16_mints_96.keras)",
        "file_id": "1LRhWPSIgWGcmrCOo2nZzpYfo6q7NC91e",
        "dest": "vgg16_mints.zip",
        "is_zip": True,
        "check_file": "vgg16_mints_96.keras",
    },
    {
        "name": "Typ-Dictionary (dict_types_95.txt)",
        "file_id": "1lJFaMwvTde_oxxF2FK5jKP4BaYf1Tgzi",
        "dest": "dict_types_95.txt",
        "is_zip": False,
        "check_file": "dict_types_95.txt",
    },
    {
        "name": "Prägestätten-Dictionary (dict_mints_96.txt)",
        "file_id": "1DwaiC5Ec_-rxhaJVzAyyhEX9yujfW3JM",
        "dest": "dict_mints_96.txt",
        "is_zip": False,
        "check_file": "dict_mints_96.txt",
    },
]


def download_from_gdrive(file_id: str, dest_path: str):
    """Lädt eine Datei von Google Drive herunter."""
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    logger.info(f"  Downloading from Google Drive: {file_id}")

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = (downloaded / total) * 100
                print(f"\r  Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)", end="", flush=True)

    print()  # Neue Zeile nach Progress
    logger.info(f"  Download abgeschlossen: {dest_path}")


def ensure_models():
    """
    Prüft ob alle Modelle vorhanden sind und lädt fehlende herunter.
    Returns True wenn alle Modelle verfügbar sind.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    all_present = True
    for item in DOWNLOADS:
        check_path = os.path.join(MODELS_DIR, item["check_file"])
        if not os.path.exists(check_path):
            all_present = False
            break

    if all_present:
        logger.info("Alle Modelle bereits vorhanden.")
        return True

    logger.info("Modelle werden heruntergeladen...")
    print("=" * 60)
    print("  MODELL-DOWNLOAD")
    print("  Dies geschieht nur beim ersten Start.")
    print("=" * 60)

    for item in DOWNLOADS:
        check_path = os.path.join(MODELS_DIR, item["check_file"])
        if os.path.exists(check_path):
            logger.info(f"  ✓ {item['name']} bereits vorhanden")
            continue

        print(f"\n⬇️  {item['name']}...")
        dest_path = os.path.join(MODELS_DIR, item["dest"])

        try:
            download_from_gdrive(item["file_id"], dest_path)

            if item["is_zip"]:
                logger.info(f"  Entpacke {dest_path}...")
                with zipfile.ZipFile(dest_path, "r") as z:
                    z.extractall(MODELS_DIR)
                os.remove(dest_path)
                logger.info(f"  ZIP entfernt: {dest_path}")

            if os.path.exists(check_path):
                print(f"  ✓ {item['name']} erfolgreich geladen")
            else:
                print(f"  ✗ {item['name']} – Datei nach Download nicht gefunden!")
                return False

        except Exception as e:
            logger.error(f"Fehler beim Download von {item['name']}: {e}")
            print(f"  ✗ FEHLER: {e}")
            return False

    print("\n" + "=" * 60)
    print("  ✓ Alle Modelle erfolgreich geladen!")
    print("=" * 60 + "\n")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = ensure_models()
    sys.exit(0 if success else 1)
