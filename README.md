# 🧠😂 MEME-COMPARATOR

> **Real-time face-based meme matching using AI, emotion recognition, and vector similarity search.**

MEME-COMPARATOR is a full-stack, AI-powered web application that captures a live camera feed, detects faces in real time, generates emotional and visual embeddings, and instantly returns the **most similar meme** from a pre-indexed meme database.

It combines:
- 🧠 **Deep Learning** (CLIP, DeepFace, FER)
- 🔍 **Vector Search** (FAISS)
- 🎥 **Real-Time Face Detection** (InsightFace)
- ⚡ **FastAPI + WebSockets**
- 🐳 **Dockerized Deployment**

---

# 📸 Demo Flow

1. Open the web app
2. Allow camera access
3. App detects your face
4. AI generates emotional + visual embeddings
5. FAISS searches for the closest meme
6. Meme appears instantly on screen

---

# ✨ Features

| Feature | Description |
|---------|------------|
| 🎥 Live Camera Feed | Browser captures webcam frames in real-time |
| 🧠 Face Detection | Uses InsightFace to detect and crop faces |
| 😭 Emotion Recognition | DeepFace + FER for emotional embeddings |
| 🖼️ Visual Embeddings | OpenCLIP for visual similarity |
| 🔍 Vector Search | FAISS similarity search engine |
| ⚡ WebSocket Streaming | Low-latency communication |
| 🐳 Dockerized | One-command deployment |
| 📦 Portable | Self-contained image with memes + DB |

---

# 🏗️ Architecture Overview
```text
Browser (Camera)
│
▼
WebSocket (JSON frames)
│
▼
FastAPI Backend
│
├── Face Detection (InsightFace)
├── Emotion Embedding (DeepFace + FER)
├── Visual Embedding (CLIP)
└── FAISS Vector Search
│
▼
Matching Meme
│
▼
Browser (Live Display)
```
---

# 📁 Project Structure

```text
MEME-COMPARATOR/
├── meme_storage/ # Meme images + FAISS database
│ ├── db.faiss
│ ├── m0.jpeg
│ ├── m1.jpeg
│ └── ...
├── src/
│ ├── backend/
│ │ ├── api.py
│ │ ├── database.py
│ │ ├── embedding_generator.py
│ │ ├── face_detector.py
│ │ ├── prepare_db.py
│ │ └── utils.py
│ ├── frontend/
│ │ └── index.html
│ ├── main.py
│ └── settings.py
├── requirements.txt
└── Dockerfile
```

---

# ⚙️ Tech Stack

## Backend
- 🐍 Python 3.11
- ⚡ FastAPI
- 🔌 WebSockets
- 🧠 FAISS (Vector Search)
- 👁️ InsightFace (Face Detection)
- 😭 DeepFace + FER (Emotion Recognition)
- 🖼️ OpenCLIP (Visual Embeddings)
- 🧮 NumPy

## Frontend
- 🌐 HTML5
- 🎥 WebRTC Camera API
- 🔌 Native WebSocket API

## Deployment
- 🐳 Docker

---

# 🚀 Quick Start (Docker — Recommended)
From the project root:
```bash
docker build -t meme-comparator .
docker run -p 5050:5050 --name meme-comparator meme-comparator
```
⏳ This may take a few minutes due to ML dependencies

---

# 🗃️ Meme Database

- Stored in meme_storage/
- Each meme is renamed to match FAISS index ID
- db.faiss stores vector index

---

# 🛠️ Configuration

Settings are managed in:
```bash
src/settings.py
```

| Setting | Description |
|---------|------------|
| face_detector_device | CPU/GPU selection |
| faiss_k_neighbors | Number of nearest memes |
| meme_storage | Meme directory path |
| debug | Debug mode |

---

# 🛡️ Security Notes

- Camera access is browser-controlled
- No images are stored remotely
- All processing runs locally

---

# 🤝 Contributing

PRs are welcome!
1. Fork repo
2. Create feature branch
3. Commit changes
4. Open pull request

---

# 📜 License

MIT License — use it, break it, meme it 😈
