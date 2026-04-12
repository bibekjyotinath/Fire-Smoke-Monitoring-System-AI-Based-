# Fire & Smoke Monitoring System (AI-Based)

---

## Overview

An end-to-end **AI-powered safety monitoring system** that detects **fire, smoke, and human presence** in real-time video streams and generates **mission-level decisions** with alerts, severity scoring, and visual evidence.

---

## System Pipeline

Input (Video/Image)
↓
Detection (YOLO: Person, Fire, Smoke)
↓
Awareness (Spatial reasoning + hazard zones)
↓
Mission Manager (Rules / Scoring)
↓
Output (Alerts, Images, Video Clips, Dashboard)

---

## Features

- 🔥 Fire & Smoke Detection (Custom YOLO)
- 🧍 Person Detection with Tracking
- 📐 Spatial Awareness (distance + hazard zones)
- ⚠️ Mission-Level Decision Engine
  - Rule-based
  - Scoring-based
- 🎥 Automatic Video Clip Recording (H264)
- 🖼️ Frame Capture (High/Critical events)
- 📊 Interactive Dashboard (Streamlit)
- 🧾 Event Logging (JSON + CSV export)
- 🔄 Temporal Smoothing (stable predictions)

---

## Project Structure

.
├── src
├─├── main.py
├─├── utils/
│ ├── detection.py
│ ├── awareness.py
│ ├── image_process.py
│ ├── schema.py
│
├─├── manager/
│ ├── rule.py
│ ├── score.py
│
├─├── reader/
│ ├── reader_factory.py
│
├─├── dashboard/
│ ├── dashboard.py
│
├── Inference/
│ ├── images/
│ ├── videos/
│ ├── json/events.json

---

## Core Components

### Detection

- YOLO-based detection for:
  - Person
  - Fire
  - Smoke
- Tracking enabled for person IDs

---

### Awareness

- Bounding box normalization
- Distance computation
- Hazard zone generation
- Person–fire interaction logic

---

### Mission Manager

#### Rule-Based

- Deterministic safety logic
- Example:
  - Person inside fire → **CRITICAL**

#### Scoring-Based

- Weighted scoring system
- Example:
  - Fire (+3), Smoke (+2), Person (+1)

---

### Input Support

- Video files
- Images
- Webcam
- RTSP streams

---

### Dashboard

- Severity filtering (CRITICAL / HIGH / MEDIUM / LOW)
- Image & video viewer
- Event table (DataFrame)
- CSV export
- Auto-refresh

---

## Visualization

| Object | Color     |
| ------ | --------- |
| Person | 🟢 Green  |
| Fire   | 🔴 Red    |
| Smoke  | 🟠 Orange |

---

## Installation

```bash
git clone <your-repo-url>
cd <repo>
pip install -r requirements.txt

---

## Run Pipeline

python main.py \
  --source "<path to video, streaming, webcamera or image>" \
  --mode scoring

---

## Run Dashboard

streamlit run dashboard/dashboard.py
```
