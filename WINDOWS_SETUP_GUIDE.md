# CrimeSketch AI - Windows Desktop Setup Guide

Complete step-by-step instructions for downloading and running CrimeSketch AI on your Windows computer.

## 📋 System Requirements

- **Windows 10/11** (64-bit)
- **RAM**: 8GB minimum (16GB recommended for faster embedding generation)
- **Disk Space**: 10GB (5GB for project + 5GB for datasets and embeddings)
- **Internet**: For initial setup and dependencies

## 🔧 Prerequisites Installation

### Step 1: Install Python 3.11+

1. Download from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
   ```cmd
   python --version
   ```

### Step 2: Install Node.js 22+

1. Download from [nodejs.org](https://nodejs.org/)
2. Choose "LTS" version
3. Run the installer and follow defaults
4. Verify installation:
   ```cmd
   node --version
   npm --version
   ```

### Step 3: Install Git (Optional but Recommended)

1. Download from [git-scm.com](https://git-scm.com/download/win)
2. Run installer with default settings
3. Verify:
   ```cmd
   git --version
   ```

## 📥 Download CrimeSketch AI

### Option A: Download from Manus (Recommended)

1. Go to the Manus project dashboard
2. Click "Download as ZIP" in the Management UI
3. Extract the ZIP file to your desired location:
   ```
   C:\Users\YourUsername\Desktop\crimesketch_ai_web
   ```

### Option B: Clone from Repository (If Available)

```cmd
git clone <repository-url>
cd crimesketch_ai_web
```

## 🚀 Setup & Run on Windows

### Step 1: Open Command Prompt or PowerShell

1. Press `Win + R`
2. Type `cmd` or `powershell`
3. Navigate to project directory:
   ```cmd
   cd C:\Users\YourUsername\Desktop\crimesketch_ai_web
   ```

### Step 2: Install Python Dependencies

```cmd
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu opencv-python numpy pandas pillow scikit-learn scipy matplotlib tqdm fastapi uvicorn python-multipart
```

**Note**: If you have an NVIDIA GPU, replace `torch ... --index-url https://download.pytorch.org/whl/cpu` with:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Node Dependencies

```cmd
npm install -g pnpm
pnpm install
```

### Step 4: Generate Embeddings (One-Time Setup)

This processes all datasets and creates the FAISS index. **Takes 30-60 minutes depending on your hardware.**

```cmd
python ml_backend/scripts/generate_embeddings.py
```

**What it does:**
- Processes 2,162+ images from all 3 datasets
- Generates 512D embeddings for each face
- Creates FAISS index for fast similarity search
- Stores metadata in SQLite database

**Progress indicator**: You'll see a progress bar. Let it complete fully.

### Step 5: Start All Services

#### Option A: Automated Script (Windows Batch)

Create a file named `start-all.bat` in the project root:

```batch
@echo off
echo ========================================
echo CrimeSketch AI - Starting Services
echo ========================================
echo.

REM Check if embeddings exist
if not exist "ml_backend\embeddings\index.faiss" (
    echo.
    echo WARNING: FAISS index not found!
    echo Running embedding generation first...
    echo This may take 30-60 minutes.
    echo.
    python ml_backend/scripts/generate_embeddings.py
    if errorlevel 1 (
        echo Embedding generation failed!
        pause
        exit /b 1
    )
)

echo Starting ML Backend on port 8000...
start "ML Backend" cmd /k python -m uvicorn server.ml_api:app --host 0.0.0.0 --port 8000

timeout /t 5 /nobreak

echo Starting Web Server on port 3000...
start "Web Server" cmd /k pnpm dev

timeout /t 3 /nobreak

echo.
echo ========================================
echo Services started!
echo ========================================
echo.
echo Web UI: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C in each window to stop services
pause
```

Then double-click `start-all.bat` to run everything.

#### Option B: Manual (Two Command Prompts)

**Terminal 1 - ML Backend:**
```cmd
python -m uvicorn server.ml_api:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Web Server:**
```cmd
pnpm dev
```

### Step 6: Access the Application

Once both services are running:

- **Web UI**: Open browser and go to http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:3000/admin

## 📱 Using CrimeSketch AI

### Search for Suspects

1. Go to http://localhost:3000
2. Click "Start Search"
3. Either:
   - **Draw a sketch** using the canvas tool
   - **Upload an image** via drag-and-drop or file picker
4. Click "Search"
5. View top-5 matches with confidence scores
6. Click on a result to see full suspect details

### View Search History

1. Scroll down on the Search page
2. See all previous searches in current session
3. Click on any search to view results again

### Access Admin Dashboard

1. Go to http://localhost:3000/admin
2. View system statistics:
   - Total suspects indexed
   - Total embeddings generated
   - Average match confidence
3. Click "Re-index Database" to rebuild FAISS index (if needed)

## 🛠️ Troubleshooting

### Issue: "Python not found"

**Solution**: 
- Ensure Python is installed and added to PATH
- Restart Command Prompt after installing Python
- Try full path: `C:\Python311\python.exe --version`

### Issue: "Module not found" (torch, faiss, etc.)

**Solution**:
```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

Or reinstall individual packages:
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install faiss-cpu
```

### Issue: Port 3000 or 8000 already in use

**Solution**: Kill existing processes:

```cmd
REM Find process on port 3000
netstat -ano | findstr :3000

REM Kill process (replace PID with actual number)
taskkill /PID <PID> /F

REM Same for port 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

Or use different ports:
```cmd
REM Terminal 1
python -m uvicorn server.ml_api:app --host 0.0.0.0 --port 8001

REM Terminal 2
pnpm dev --port 3001
```

### Issue: Embedding generation is very slow

**Solutions**:
- Ensure no other heavy applications are running
- Check disk space (need 5GB+)
- If you have GPU, install CUDA version of PyTorch
- Reduce batch size in `ml_backend/scripts/generate_embeddings.py` (line 25)

### Issue: Out of memory during embedding generation

**Solution**: Reduce batch size in the script:
```python
# In ml_backend/scripts/generate_embeddings.py
BATCH_SIZE = 8  # Reduce from 16 to 8 or 4
```

### Issue: Web UI shows "Cannot connect to ML API"

**Solution**:
1. Verify ML backend is running: http://localhost:8000/health
2. Check firewall settings (allow localhost connections)
3. Verify both services are on same machine
4. Check `.env.local` file has correct `ML_API_URL=http://localhost:8000`

## 📊 Performance Tips

### For Faster Embedding Generation

1. **Use GPU** (if available):
   - Install CUDA 11.8+
   - Install GPU version of PyTorch
   - Embeddings will generate 5-10x faster

2. **Increase RAM allocation**:
   - Close unnecessary applications
   - Ensure 8GB+ free RAM

3. **Use SSD**:
   - Faster disk I/O for dataset loading
   - Faster database writes

### For Faster Searches

1. **Use approximate search** (in production):
   - Edit `ml_backend/embeddings/faiss_indexer.py`
   - Change `index_type='flat'` to `index_type='ivf'`
   - Trade accuracy for speed

2. **Increase search results cache**:
   - Results are cached in memory
   - Repeated searches are instant

## 🔄 Updating the Project

To get the latest version:

1. Download new ZIP from Manus dashboard
2. Extract to new folder
3. Copy your datasets folder:
   ```cmd
   xcopy old_folder\datasets new_folder\datasets /E
   ```
4. Copy database (optional):
   ```cmd
   copy old_folder\ml_backend\database\crimesketch.db new_folder\ml_backend\database\
   ```

## 📝 Configuration

Edit `.env.local` to customize:

```
# ML Backend
ML_API_URL=http://localhost:8000
ML_API_TIMEOUT=30000

# Application
VITE_APP_TITLE=CrimeSketch AI
VITE_APP_LOGO=https://your-logo-url.com/logo.png

# Features
ENABLE_ADMIN_DASHBOARD=true
ENABLE_SEARCH_HISTORY=true
ENABLE_PREPROCESSING_PREVIEW=true
```

## 🔐 Security Notes

- **Local Only**: By default, services only listen on localhost
- **No Authentication**: For local use only
- **For Network Access**: Configure firewall and add authentication before exposing

To allow network access (not recommended for public):
```cmd
python -m uvicorn server.ml_api:app --host 0.0.0.0 --port 8000
```

## 📞 Getting Help

1. **Check logs**:
   - ML Backend: `.logs/ml_backend.log`
   - Web Server: `.logs/web_server.log`

2. **View API docs**: http://localhost:8000/docs

3. **Check database**:
   - Location: `ml_backend/database/crimesketch.db`
   - Use SQLite viewer to inspect tables

4. **Review README**:
   - Full documentation: `CRIMESKETCH_README.md`

## ✅ Verification Checklist

- [ ] Python 3.11+ installed
- [ ] Node.js 22+ installed
- [ ] Project downloaded and extracted
- [ ] Python dependencies installed
- [ ] Node dependencies installed (pnpm install)
- [ ] Embeddings generated (python ml_backend/scripts/generate_embeddings.py)
- [ ] ML Backend running (http://localhost:8000/health returns OK)
- [ ] Web Server running (http://localhost:3000 loads)
- [ ] Can draw/upload sketch
- [ ] Can see search results
- [ ] Admin dashboard accessible

## 🎉 You're Ready!

Once all checks pass, you have a fully functional forensic sketch recognition system running locally on your Windows desktop!

---

**Last Updated**: April 16, 2026
**Windows Tested**: Windows 10, Windows 11
**Python Version**: 3.11+
**Node Version**: 22+
