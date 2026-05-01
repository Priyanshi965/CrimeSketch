@echo off
:: ════════════════════════════════════════════════════════════════════
::  CrimeSketch AI — FAST MODE
::  Runs the full pipeline in ~5 minutes on CPU.
::
::  What fast mode does:
::    Custom CNN   → 3 epochs (vs 20 full)  saves ~25 min on CPU
::    Test slice   → 60 samples (vs 134)    saves time on 11 models
::    Latency      → 10 trials  (vs 50)     saves ~1 min
::    ROC/PR pairs → 500        (vs 5000)   saves ~30s per model
::    Max CMC rank → 10         (vs 20)
::
::  Results are fully real — computed from your actual dataset and models.
::  Run install_and_run.bat once first to install all packages.
:: ════════════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║   CrimeSketch AI — FAST MODE (~5 min on CPU)        ║
echo  ║   IEEE FICV 2026 · MIT WPU Pune                     ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    pause & exit /b 1
)

echo [INFO] Installing/verifying dependencies (quick check)...
pip install torch torchvision facenet-pytorch faiss-cpu opencv-python ^
    scikit-learn scikit-image scipy tqdm Pillow matplotlib deepface tf-keras -q
echo [OK] Dependencies ready.
echo.

echo [Fast Run] Starting pipeline...
echo.

python run_pipeline.py ^
    --data_dir "C:\Users\User\OneDrive\Desktop\ippr\datasets\organized" ^
    --output_dir results ^
    --fast ^
    --latency_trials 10 ^
    --max_rank 10

if errorlevel 1 (
    echo.
    echo [ERROR] Pipeline failed. Check output above.
    pause & exit /b 1
)

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║  ✓ Done! Results and 10 figure PNGs are ready.      ║
echo  ╚══════════════════════════════════════════════════════╝
echo.
echo  Figures  : results\figures\  (10 PNG files for IEEE paper)
echo  Dashboard: results\dashboard.html
echo  JSON     : results\metrics.json
echo.
start "" "results\figures"
start "" "results\dashboard.html"
pause
