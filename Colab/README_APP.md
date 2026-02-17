# 🪙 Münz-Bildsuche Museum-App

Touch-optimierte Streamlit-App für die visuelle Suche nach ähnlichen antiken Münzen im Museum.

## 🚀 Installation

```bash
# 1. Dependencies installieren
pip install -r requirements.txt

# 2. Datenordner mit Münzbildern erstellen
mkdir data
# Kopieren Sie Ihre Münzbilder (.jpg, .png) in den data/ Ordner
```

## ▶️ Starten der App

```bash
streamlit run app.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`

## 🎨 Funktionen

- **Galerie-Modus**: Übersicht aller Münzen in einer Touch-freundlichen Grid-Ansicht
- **Bildsuche**: Klick auf eine Münze startet die Ähnlichkeitssuche
- **Ergebnisse**: Top 6 ähnlichste Münzen mit Ähnlichkeitswert (0-100%)
- **Feature-Caching**: Beim ersten Start werden alle Bilder analysiert und gecacht (für schnellere Suchen)

## 🖼️ Bildformat

- Unterstützte Formate: `.jpg`, `.png`, `.jpeg`
- Empfohlene Auflösung: mindestens 224x224 Pixel
- Die Bilder werden automatisch für das VGG16-Modell vorverarbeitet

## 🔧 Anpassungen

### Anzahl der Ergebnisse ändern
In `app.py`:
```python
TOP_K_RESULTS = 6  # Auf gewünschte Anzahl ändern
```

### Galerie-Spalten anpassen
```python
GALLERY_COLS = 3  # Für mehr/weniger Spalten
```

### Farben/Design anpassen
Die `<style>`-Sektion in `app.py` enthält alle CSS-Anpassungen.

## 📊 Technische Details

- **Feature-Extraktion**: VGG16 (ImageNet-Pretrained, ohne Top-Layer)
- **Ähnlichkeitsmetrik**: Cosine-Similarity
- **Caching**: Features werden in `features_cache.pkl` gespeichert
- **Touch-Optimierung**: Große Buttons (80px Höhe), klare Schrift (24px)

## 🗑️ Cache zurücksetzen

Wenn neue Bilder hinzugefügt wurden:
```bash
rm features_cache.pkl
# App neu starten - Cache wird automatisch neu erstellt
```

## 📱 Für Touchscreen-Display optimieren

### Vollbild-Modus im Browser
- Chrome/Edge: F11 drücken
- Firefox: F11 drücken

### Kiosk-Modus (für Production)
```bash
# Chrome im Kiosk-Modus starten
chromium-browser --kiosk --app=http://localhost:8501
```

### Streamlit-Config für Touchscreen
Erstellen Sie `.streamlit/config.toml`:
```toml
[browser]
gatherUsageStats = false

[server]
headless = true
enableCORS = false
```

## 🐛 Troubleshooting

**Keine Bilder gefunden:**
- Stellen Sie sicher, dass Bilder im `data/` Ordner liegen
- Prüfen Sie die Dateiformate (.jpg, .png, .jpeg)

**App lädt langsam:**
- Beim ersten Start werden Features berechnet (kann bei vielen Bildern dauern)
- Cache wird danach verwendet - nachfolgende Starts sind schnell

**Speicherprobleme:**
- Reduzieren Sie die Bildanzahl oder -auflösung
- Nutzen Sie eine GPU für schnellere Verarbeitung
