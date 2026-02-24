# IR-on-coin-datasets

This is an Image Recognition approach on the Corpus Nummorum dataset. You can predict coin types and mints on coin images (obverse and reverse).
If you have any problems or suggestions, please feel free to contact us.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Frankfurt-BigDataLab/IR-on-coin-datasets/blob/main/Colab/IR-for-types-and-mints.ipynb)

---

## Lokale Ausführung mit Docker

Die Anwendung besteht aus zwei Diensten:

| Dienst | Beschreibung | Port |
|--------|--------------|------|
| **Backend** | FastAPI + TensorFlow (Bildanalyse-API) | `8000` |
| **Frontend** | Nginx (statische Web-UI) | `3000` |

### Ports konfigurieren

Die Standard-Ports lassen sich in der Datei `.env` im Projektverzeichnis anpassen:

```ini
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

Einfach den gewünschten Wert eintragen und die Container neu starten. Die `:-`-Syntax in der `docker-compose.yml` sorgt dafür, dass die Standardwerte greifen, falls die Variablen nicht gesetzt sind.

---

### Voraussetzungen

Installiere **Docker Desktop** (enthält Docker Engine und Docker Compose):

- **Windows**: [docs.docker.com/desktop/install/windows-install](https://docs.docker.com/desktop/install/windows-install/)  
  > Empfohlen: WSL 2 als Backend aktivieren (wird vom Installer vorgeschlagen)
- **macOS**: [docs.docker.com/desktop/install/mac-install](https://docs.docker.com/desktop/install/mac-install/)  
  > Verfügbar für Intel (x86_64) und Apple Silicon (ARM64/M-Chips)
- **Linux**: [docs.docker.com/engine/install](https://docs.docker.com/engine/install/)  
  > Docker Engine + das Compose-Plugin installieren; kein Docker Desktop erforderlich

Überprüfe die Installation:

```bash
docker --version
docker compose version
```

---

### Anwendung starten

1. **Repository klonen (falls noch nicht geschehen)**

   ```bash
   git clone https://github.com/Frankfurt-BigDataLab/IR-on-coin-datasets.git
   cd IR-on-coin-datasets
   ```

2. **Container bauen und starten**

   ```bash
   docker compose up --build
   ```

   > Beim **ersten Start** werden die KI-Modelle automatisch heruntergeladen (~1–2 GB).  
   > Das Backend benötigt dafür bis zu **2 Minuten** – die Web-UI ist erst danach vollständig nutzbar.

3. **Anwendung öffnen**

   Öffne im Browser: [http://localhost:3000](http://localhost:3000)

   Die REST-API ist direkt erreichbar unter: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Im Hintergrund starten (ohne Log-Ausgabe)

```bash
docker compose up -d --build
```

Logs nachträglich anzeigen:

```bash
docker compose logs -f
```

---

### Anwendung stoppen

```bash
docker compose down
```

Zum vollständigen Entfernen **inklusive der heruntergeladenen Modelle** (Volume löschen):

```bash
docker compose down -v
```

> **Hinweis:** Nach `down -v` werden die Modelle beim nächsten Start erneut heruntergeladen.

---

### Hinweise für Windows

- Alle Befehle funktionieren in **PowerShell**, der **Windows-Eingabeaufforderung (CMD)** und im **WSL 2 Terminal** gleichermaßen.
- Stelle sicher, dass Docker Desktop läuft (Icon in der Taskleiste sichtbar), bevor du `docker compose` ausführst.
- Bei Problemen mit Dateipfaden unter WSL 2: das Repository am besten direkt im Linux-Dateisystem (`~/projekte/...`) klonen, nicht auf dem Windows-Laufwerk (`/mnt/c/...`).

### Hinweise für macOS

- Docker Desktop muss gestartet sein (Icon in der Menüleiste).
- Auf **Apple Silicon (M1/M2/M3/M4)**: Das TensorFlow-Image wird automatisch für `linux/arm64` gebaut – dies ist vollständig unterstützt.

### Hinweise für Linux

- Füge deinen Benutzer zur `docker`-Gruppe hinzu, um `sudo` zu vermeiden:
  ```bash
  sudo usermod -aG docker $USER
  # Danach neu einloggen oder: newgrp docker
  ```

---

### API-Gesundheitsstatus prüfen

```bash
curl http://localhost:8000/api/health
```

Erwartete Antwort (sobald Modelle geladen):

```json
{"status": "ok"}
```

