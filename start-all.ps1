# CrimeSketch AI - PowerShell Startup Script for Windows
# Run with: powershell -ExecutionPolicy Bypass -File start-all.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CrimeSketch AI - Windows Startup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ProjectDir = Get-Location
$MLBackendPort = 8000
$WebServerPort = 3000

# Check if embeddings exist
if (-not (Test-Path "$ProjectDir\ml_backend\embeddings\index.faiss")) {
    Write-Host "WARNING: FAISS index not found!" -ForegroundColor Yellow
    Write-Host "Running embedding generation first..." -ForegroundColor Yellow
    Write-Host "This may take 30-60 minutes depending on your hardware." -ForegroundColor Yellow
    Write-Host ""
    
    python ml_backend/scripts/generate_embeddings.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Embedding generation failed!" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Green
Write-Host ""

# Start ML Backend
Write-Host "[1/2] Starting ML Backend on port $MLBackendPort..." -ForegroundColor Cyan
$MLProcess = Start-Process -FilePath "python" `
    -ArgumentList "-m uvicorn server.ml_api:app --host 0.0.0.0 --port $MLBackendPort" `
    -WindowStyle Normal `
    -PassThru

Write-Host "✓ ML Backend started (PID: $($MLProcess.Id))" -ForegroundColor Green

# Wait for ML backend to be ready
Write-Host "Waiting for ML Backend to initialize..." -ForegroundColor Yellow
$MLReady = $false
for ($i = 1; $i -le 30; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$MLBackendPort/health" -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ ML Backend is ready!" -ForegroundColor Green
            $MLReady = $true
            break
        }
    } catch {
        # Still waiting
    }
    
    if ($i -eq 30) {
        Write-Host "✗ ML Backend failed to start" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    Start-Sleep -Seconds 1
}

Write-Host ""

# Start Web Server
Write-Host "[2/2] Starting Web Server on port $WebServerPort..." -ForegroundColor Cyan
$WebProcess = Start-Process -FilePath "pnpm" `
    -ArgumentList "dev" `
    -WindowStyle Normal `
    -PassThru

Write-Host "✓ Web Server started (PID: $($WebProcess.Id))" -ForegroundColor Green

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "✓ All services started successfully!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access the application at:" -ForegroundColor Cyan
Write-Host "  Web UI: http://localhost:$WebServerPort" -ForegroundColor White
Write-Host "  ML API: http://localhost:$MLBackendPort" -ForegroundColor White
Write-Host "  API Docs: http://localhost:$MLBackendPort/docs" -ForegroundColor White
Write-Host ""
Write-Host "To stop all services:" -ForegroundColor Yellow
Write-Host "  Close both command windows" -ForegroundColor White
Write-Host "  Or run: Stop-Process -Id $($MLProcess.Id), $($WebProcess.Id)" -ForegroundColor White
Write-Host ""
Write-Host "View logs:" -ForegroundColor Yellow
Write-Host "  ML Backend: .logs\ml_backend.log" -ForegroundColor White
Write-Host "  Web Server: .logs\web_server.log" -ForegroundColor White
Write-Host ""

# Keep script running
Read-Host "Press Enter to exit (services will continue running)"
