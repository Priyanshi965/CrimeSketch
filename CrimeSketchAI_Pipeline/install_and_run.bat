@echo off
:: ════════════════════════════════════════════════════════════════════
::  CrimeSketch AI — IEEE FICV 2026
::  Install dependencies and run the full local evaluation pipeline
::  MIT World Peace University, Pune — Priyanshi & Prachi Gupta
:: ════════════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   CrimeSketch AI — Local Evaluation Pipeline        ║
echo  ║   IEEE FICV 2026 · All results computed locally     ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

:: ── Check Python ───────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo [OK] %%v found.

:: ── Install dependencies ───────────────────────────────────────────
echo.
echo [Step 1/2] Installing Python dependencies...
python -m pip install --upgrade pip -q
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
pip install facenet-pytorch faiss-cpu opencv-python scikit-learn scikit-image scipy tqdm Pillow pandas matplotlib requests -q
pip install deepface tf-keras -q
if errorlevel 1 (
    echo [WARNING] Some packages may have failed. Pipeline will run with available models.
)
echo [OK] Dependencies ready.

:: ── Run pipeline ──────────────────────────────────────────────────
echo.
echo [Step 2/2] Running evaluation pipeline on your dataset...
echo.
echo  Dataset : C:\Users\User\OneDrive\Desktop\ippr\datasets\organized
echo  Models  : All (PCA+SVM, Custom CNN, VGGFace1, DeepFace, FaceNet,
echo             CosFace, ArcFace, TransFace, MagFace, AdaFace, CrimeSketch AI)
echo  Output  : results\dashboard.html
echo.

python run_pipeline.py ^
    --data_dir "C:\Users\User\OneDrive\Desktop\ippr\datasets\organized" ^
    --output_dir results ^
    --max_rank 20 ^
    --latency_trials 50

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed. Check the output above for details.
    pause
    exit /b 1
)

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║  ✓ Pipeline Complete — All metrics computed locally  ║
echo  ╠══════════════════════════════════════════════════════╣
echo  ║  JSON    : results\metrics.json                     ║
echo  ║  Dashboard: results\dashboard.html                  ║
echo  ║  Figures : results\figures\  (10 PNG, 300 DPI)      ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
start "" "results\dashboard.html"
start "" "results\figures"
pause
