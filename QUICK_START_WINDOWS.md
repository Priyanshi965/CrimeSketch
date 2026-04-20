# CrimeSketch AI - Quick Start for Windows

**TL;DR**: 5 minutes to get running (after initial setup)

## ⚡ Super Quick Setup (First Time Only)

### 1. Install Prerequisites (10 minutes)

**Python 3.11+**
- Download: https://www.python.org/downloads/
- Run installer, check "Add Python to PATH"
- Verify: Open Command Prompt, type `python --version`

**Node.js 22+**
- Download: https://nodejs.org/ (LTS version)
- Run installer with defaults
- Verify: Open Command Prompt, type `node --version`

### 2. Extract Project

1. Download CrimeSketch AI ZIP from Manus
2. Extract to: `C:\Users\YourUsername\Desktop\crimesketch_ai_web`
3. Open Command Prompt in this folder

### 3. Install Dependencies (5 minutes)

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu opencv-python numpy pandas pillow scikit-learn scipy fastapi uvicorn
npm install -g pnpm
pnpm install
```

### 4. Generate Embeddings (30-60 minutes, one-time)

```cmd
python ml_backend/scripts/generate_embeddings.py
```

**Leave this running.** It will process 2,162+ images and create the search index.

### 5. Start Services

**Option A - Automatic (Easiest)**

Double-click: `start-all.bat`

**Option B - Manual (Two Command Prompts)**

Terminal 1:
```cmd
python -m uvicorn server.ml_api:app --host 0.0.0.0 --port 8000
```

Terminal 2:
```cmd
pnpm dev
```

### 6. Open in Browser

- **Web UI**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 🎯 Using CrimeSketch AI

### Draw a Sketch
1. Click "Start Search"
2. Use the canvas to draw a face
3. Click "Search"
4. See top-5 matches

### Upload a Sketch
1. Click "Start Search"
2. Drag and drop an image or click "Upload"
3. Click "Search"
4. See results

### View Admin Dashboard
1. Click "Dashboard" in top right
2. See system statistics
3. Click "Re-index" if needed

## 🆘 Common Issues

### "Python not found"
- Reinstall Python, **check "Add Python to PATH"**
- Restart Command Prompt after installing

### "Module not found"
```cmd
pip install --upgrade pip
pip install torch faiss-cpu opencv-python
```

### Port already in use
```cmd
netstat -ano | findstr :3000
taskkill /PID <number> /F
```

### Embedding generation is slow
- Normal! Takes 30-60 minutes
- Don't close the window
- Check progress bar

### "Cannot connect to ML API"
- Make sure ML backend is running in Terminal 1
- Check http://localhost:8000/health

## 📁 Project Structure

```
crimesketch_ai_web/
├── client/              # React frontend
├── server/              # Express backend
├── ml_backend/          # Python ML code
│   └── scripts/
│       └── generate_embeddings.py   # Run this first!
├── datasets/            # Your datasets
├── start-all.bat        # Click to start (Windows)
└── WINDOWS_SETUP_GUIDE.md
```

## 🚀 Full Setup Checklist

- [ ] Python 3.11+ installed
- [ ] Node.js 22+ installed
- [ ] Project extracted
- [ ] Dependencies installed (`pip install ...`)
- [ ] pnpm installed (`npm install -g pnpm`)
- [ ] Node packages installed (`pnpm install`)
- [ ] Embeddings generated (`python ml_backend/scripts/generate_embeddings.py`)
- [ ] ML Backend running (http://localhost:8000/health)
- [ ] Web Server running (http://localhost:3000)
- [ ] Can draw/upload sketch
- [ ] Can see search results

## 💡 Tips

- **First run takes longest**: Embedding generation is one-time setup
- **Keep terminals open**: Both services must run simultaneously
- **Use Chrome/Edge**: Best browser compatibility
- **Check logs**: `.logs/ml_backend.log` and `.logs/web_server.log`

## 📞 Need Help?

1. Check `WINDOWS_SETUP_GUIDE.md` for detailed troubleshooting
2. Review `CRIMESKETCH_README.md` for full documentation
3. Check API docs at http://localhost:8000/docs

---

**That's it!** You now have a forensic sketch recognition system running locally. 🎉
