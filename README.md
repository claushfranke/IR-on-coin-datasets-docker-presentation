# IR-on-coin-datasets — Interactive Presentation

This is a Docker-based interactive presentation layer for the
[IR-on-coin-datasets](https://github.com/Frankfurt-BigDataLab/IR-on-coin-datasets) project by the
**Frankfurt Big Data Lab**. It lets museum visitors predict the **type** and **mint** of ancient
coins from the [Corpus Nummorum](https://www.corpus-nummorum.eu/) using a trained image-recognition
model.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Frankfurt-BigDataLab/IR-on-coin-datasets/blob/main/Colab/IR-for-types-and-mints.ipynb)

---

## Citation

If you use this work, please cite the original repository:

```bibtex
@misc{frankfurtbigdatalab2021coincognition,
  author       = {{Frankfurt Big Data Lab}},
  title        = {IR-on-coin-datasets: Image Recognition on the Corpus Nummorum Dataset},
  year         = {2021},
  howpublished = {\url{https://github.com/Frankfurt-BigDataLab/IR-on-coin-datasets}},
  note         = {GitHub repository}
}
```

> Original repository: <https://github.com/Frankfurt-BigDataLab/IR-on-coin-datasets>

---

## License

Content is licensed under a
[Creative Commons Attribution – NonCommercial – ShareAlike 3.0 Germany](http://creativecommons.org/licenses/by-nc-sa/3.0/de/)
license (CC BY-NC-SA 3.0 DE).

---

## Running Locally with Docker

The application consists of two services:

| Service | Description | Port |
|---------|-------------|------|
| **Backend** | FastAPI + TensorFlow (image analysis API) | `8000` |
| **Frontend** | Nginx (static web UI) | `3000` |

### Configure Ports

Default ports can be changed via an `.env` file in the project root:

```ini
BACKEND_PORT=8000
FRONTEND_PORT=3000
```

Edit the values and restart the containers. The `:-` syntax in `docker-compose.yml` ensures
default values are used if the variables are not set.

---

### Prerequisites

Install **Docker Desktop** (includes Docker Engine and Docker Compose):

- **Windows**: [docs.docker.com/desktop/install/windows-install](https://docs.docker.com/desktop/install/windows-install/)  
  > Recommended: enable WSL 2 as the backend (suggested by the installer)
- **macOS**: [docs.docker.com/desktop/install/mac-install](https://docs.docker.com/desktop/install/mac-install/)  
  > Available for Intel (x86_64) and Apple Silicon (ARM64 / M-series chips)
- **Linux**: [docs.docker.com/engine/install](https://docs.docker.com/engine/install/)  
  > Install Docker Engine + the Compose plugin; Docker Desktop is not required

Verify the installation:

```bash
docker --version
docker compose version
```

---

### Starting the Application

1. **Clone the repository (if you haven't already)**

   ```bash
   git clone https://github.com/Frankfurt-BigDataLab/IR-on-coin-datasets.git
   cd IR-on-coin-datasets
   ```

2. **Build and start the containers**

   ```bash
   docker compose up --build
   ```

   > On the **first start** the AI models are downloaded automatically (~1–2 GB).  
   > The backend needs up to **2 minutes** — the web UI is only fully usable after that.

3. **Open the application**

   Open in your browser: [http://localhost:3000](http://localhost:3000)

   The REST API is available at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Running in the Background (detached mode)

```bash
docker compose up -d --build
```

View logs later:

```bash
docker compose logs -f
```

---

### Stopping the Application

```bash
docker compose down
```

To fully remove everything **including the downloaded models** (delete volume):

```bash
docker compose down -v
```

> **Note:** After `down -v` the models will be re-downloaded on the next start.

---

### Notes for Windows

- All commands work in **PowerShell**, the **Windows Command Prompt (CMD)**, and the **WSL 2 terminal**.
- Make sure Docker Desktop is running (icon visible in the taskbar) before running `docker compose`.
- If you have path issues under WSL 2: clone the repository directly into the Linux filesystem
  (`~/projects/...`) rather than the Windows drive (`/mnt/c/...`).

### Notes for macOS

- Docker Desktop must be running (icon in the menu bar).
- On **Apple Silicon (M1/M2/M3/M4)**: the TensorFlow image is built automatically for `linux/arm64` — fully supported.

### Notes for Linux

- Add your user to the `docker` group to avoid `sudo`:
  ```bash
  sudo usermod -aG docker $USER
  # Then log out and back in, or run: newgrp docker
  ```

---

### Check API Health

```bash
curl http://localhost:8000/api/health
```

Expected response (once the models are loaded):

```json
{"status": "ok"}
```

