# Fire & Smoke Monitoring System (AI-Based)

---

## Overview

An end-to-end **AI-powered safety monitoring system** that detects **fire, smoke, and human presence** in real-time video streams and generates **mission-level decisions** with alerts, severity scoring, and visual evidence.

## This system converts raw visual perception into **mission-aware decisions with contextual scene understanding**, not just object detection.

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

## Assumptions

### Robot Platform

- Assumed a quadruped robot (robodog) for inspection tasks
- Chosen due to stability in industrial environments and ability to navigate uneven terrain

### Target Environment

- Warehouse / factory inspection setting
- Includes zones such as conveyor belts, storage areas, and utility corridors

### Sensors

- RGB camera used as primary sensor
- No thermal or depth sensors used (can be extended in production)

### Simulation vs Real Deployment

- System uses prerecorded video (YouTube / sample footage)
- Simulates real-world inspection scenarios
- Designed to be deployable on live camera feeds

### Zone Definition

- Frame is divided into:
  - left_zone
  - center_zone
  - right_zone
- Used for generating scene-level contextual summaries

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
