#!/usr/bin/env python3
"""
CrimeSketch AI - Automated Windows Setup Script
Downloads, installs, and runs the complete system with one command.

Usage:
    python setup-windows.py
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from urllib.request import urlopen
import zipfile
import tempfile

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def check_python_version():
    """Verify Python 3.11+ is installed."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print_error(f"Python 3.11+ required. You have {version.major}.{version.minor}")
        print_info("Download from: https://www.python.org/downloads/")
        sys.exit(1)
    
    print_success(f"Python {version.major}.{version.minor}.{version.micro} detected")

def check_nodejs():
    """Verify Node.js 22+ is installed."""
    print_header("Checking Node.js")
    
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        version_str = result.stdout.strip().replace('v', '')
        major_version = int(version_str.split('.')[0])
        
        if major_version < 22:
            print_warning(f"Node.js 22+ recommended. You have {version_str}")
            print_info("Download from: https://nodejs.org/")
        else:
            print_success(f"Node.js {version_str} detected")
    except FileNotFoundError:
        print_error("Node.js not found")
        print_info("Download from: https://nodejs.org/")
        sys.exit(1)

def install_python_dependencies():
    """Install Python ML dependencies."""
    print_header("Installing Python Dependencies")
    
    dependencies = [
        ("torch torchvision torchaudio", "--index-url https://download.pytorch.org/whl/cpu"),
        ("faiss-cpu", ""),
        ("opencv-python", ""),
        ("numpy pandas pillow", ""),
        ("scikit-learn scipy", ""),
        ("fastapi uvicorn python-multipart", ""),
        ("tqdm requests", ""),
    ]
    
    for package, extra_args in dependencies:
        print_info(f"Installing {package}...")
        cmd = f"pip install {package}"
        if extra_args:
            cmd += f" {extra_args}"
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print_warning(f"Failed to install {package}. Continuing...")
        else:
            print_success(f"Installed {package}")

def install_node_dependencies():
    """Install Node.js dependencies."""
    print_header("Installing Node.js Dependencies")
    
    print_info("Installing pnpm globally...")
    subprocess.run("npm install -g pnpm", shell=True, capture_output=True)
    print_success("pnpm installed")
    
    print_info("Installing project dependencies...")
    result = subprocess.run("pnpm install", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print_error("Failed to install Node dependencies")
        print_info(result.stderr)
        sys.exit(1)
    
    print_success("Node dependencies installed")

def check_embeddings():
    """Check if embeddings have been generated."""
    embedding_path = Path("ml_backend/embeddings/index.faiss")
    return embedding_path.exists()

def generate_embeddings():
    """Generate embeddings for all datasets."""
    print_header("Generating Embeddings (One-Time Setup)")
    
    if check_embeddings():
        print_success("Embeddings already exist. Skipping generation.")
        return
    
    print_warning("This will take 30-60 minutes depending on your hardware")
    print_info("Processing 2,162+ images from all datasets...")
    
    result = subprocess.run(
        "python ml_backend/scripts/generate_embeddings.py",
        shell=True,
        capture_output=False,
        text=True
    )
    
    if result.returncode != 0:
        print_error("Embedding generation failed")
        sys.exit(1)
    
    print_success("Embeddings generated successfully")

def start_services():
    """Start ML backend and web server."""
    print_header("Starting Services")
    
    import time

    # Free ports if already in use from a previous run
    for port in [8000, 3000]:
        result = subprocess.run(
            f'netstat -ano | findstr ":{port} " | findstr "LISTENING"',
            shell=True, capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if parts:
                pid = parts[-1]
                subprocess.run(f'taskkill /PID {pid} /F', shell=True, capture_output=True)

    print_info("Starting ML Backend on port 8000...")
    ml_process = subprocess.Popen(
        "python -m uvicorn server.ml_api:app --host 0.0.0.0 --port 8000",
        shell=True,
    )
    print_success(f"ML Backend started (PID: {ml_process.pid})")

    # Wait for ML backend to be ready
    print_info("Waiting for ML Backend to initialize...")
    for i in range(30):
        if ml_process.poll() is not None:
            print_error("ML Backend process exited unexpectedly. Check the output above for errors.")
            sys.exit(1)
        try:
            response = urlopen("http://localhost:8000/health", timeout=2)
            if response.status == 200:
                print_success("ML Backend is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        if ml_process.poll() is not None:
            print_error("ML Backend process exited. Check the output above for errors.")
            sys.exit(1)
        print_warning("ML Backend health check timed out. Continuing anyway...")

    print_info("Starting Web Server on port 3000...")
    web_process = subprocess.Popen(
        "pnpm dev",
        shell=True,
    )
    print_success(f"Web Server started (PID: {web_process.pid})")

    print_header("Services Running")
    print_success("All services started successfully!")
    print(f"\n{Colors.CYAN}Access the application at:{Colors.ENDC}")
    print(f"  {Colors.BOLD}Web UI:{Colors.ENDC} http://localhost:3000")
    print(f"  {Colors.BOLD}API Docs:{Colors.ENDC} http://localhost:8000/docs")
    print(f"  {Colors.BOLD}Admin Dashboard:{Colors.ENDC} http://localhost:3000/admin")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop all services{Colors.ENDC}\n")

    try:
        while True:
            if ml_process.poll() is not None:
                print_error("ML Backend stopped unexpectedly.")
                web_process.terminate()
                sys.exit(1)
            if web_process.poll() is not None:
                print_error("Web Server stopped unexpectedly.")
                ml_process.terminate()
                sys.exit(1)
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n" + Colors.YELLOW + "Stopping services..." + Colors.ENDC)
        ml_process.terminate()
        web_process.terminate()
        try:
            ml_process.wait(timeout=5)
            web_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ml_process.kill()
            web_process.kill()
        print_success("Services stopped")

def main():
    """Main setup flow."""
    print_header("CrimeSketch AI - Automated Setup")
    
    # Change to project directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print_info(f"Project directory: {script_dir}")
    
    # Run checks
    check_python_version()
    check_nodejs()
    
    # Install dependencies
    install_python_dependencies()
    install_node_dependencies()
    
    # Generate embeddings if needed
    generate_embeddings()
    
    # Start services
    start_services()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print_error(f"Setup failed: {str(e)}")
        sys.exit(1)
