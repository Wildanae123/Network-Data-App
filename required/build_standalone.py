# build_standalone.py
import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_npm_available():
    """Check if npm is available in the system."""
    try:
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def check_node_available():
    """Check if Node.js is available in the system."""
    try:
        subprocess.run(["node", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def build_frontend():
    """Build the React frontend for production."""
    print("Checking Node.js and npm availability...")
    
    if not check_node_available():
        print("Error: Node.js is not installed or not in PATH!")
        print("Please install Node.js from https://nodejs.org/")
        return False
    
    if not check_npm_available():
        print("Error: npm is not installed or not in PATH!")
        print("Please install Node.js (which includes npm) from https://nodejs.org/")
        return False
    
    print("Building React frontend...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("Error: frontend directory not found!")
        print("Creating basic frontend structure...")
        return create_basic_frontend()
    
    try:
        print("Installing npm dependencies...")
        result = subprocess.run(["npm", "install"], cwd=frontend_dir, 
                              capture_output=True, text=True, check=True)
        print("Dependencies installed successfully!")
        
        print("Building production bundle...")
        result = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, 
                              capture_output=True, text=True, check=True)
        print("Frontend build successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Frontend build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error during frontend build: {e}")
        return False

def create_basic_frontend():
    """Create a basic frontend structure if React frontend is not available."""
    print("Creating basic HTML frontend as fallback...")
    
    frontend_dir = Path("frontend")
    dist_dir = frontend_dir / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Create basic index.html
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Data App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 800px; margin: 0 auto; }
        .error { color: red; padding: 20px; border: 1px solid red; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Network Data App</h1>
        <div class="error">
            <h3>Frontend Build Error</h3>
            <p>The React frontend could not be built. The application is running with a basic fallback interface.</p>
            <p>To use the full interface, please:</p>
            <ol>
                <li>Install Node.js from <a href="https://nodejs.org/">https://nodejs.org/</a></li>
                <li>Run the build script again</li>
            </ol>
            <p>You can still use the API endpoints directly at: <strong>/api/</strong></p>
        </div>
    </div>
</body>
</html>"""
    
    with open(dist_dir / "index.html", "w") as f:
        f.write(html_content)
    
    print("Basic frontend created successfully!")
    return True

def create_pyinstaller_spec():
    """Create PyInstaller spec file in output/ directory."""
    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['backend/server.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('frontend/dist', 'frontend/dist'),
        ('backend/commands.yaml', 'backend'),
    ],
    hiddenimports=[
        'flask',
        'flask_cors',
        'pandas',
        'plotly',
        'yaml',
        'jsonrpclib',
        'werkzeug',
        'openpyxl',
        'queue',
        'socket',
        'urllib3',
        'ssl',
        'json',
        'threading',
        'concurrent.futures',
        'difflib',
        'logging',
        'time',
        'uuid',
        'datetime',
        'pathlib',
        'typing',
        'dataclasses',
        'tempfile',
        'io',
        'base64',
        'collections',
        'importlib.metadata'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='NetworkDataApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    coerce_archive=True,
    cipher=block_cipher,
)
"""
    
    # Create the output/ directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the spec file in output/ directory
    spec_file_path = output_dir / "NetworkDataApp.spec"
    with open(spec_file_path, "w") as f:
        f.write(spec_content)
    
    print(f"PyInstaller spec file created at: {spec_file_path}")
    return spec_file_path

def install_dependencies():
    """Install required Python dependencies."""
    print("Installing Python dependencies...")
    
    dependencies = [
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0",
        "gunicorn>=21.2.0",
        "pandas>=2.1.0",
        "numpy>=1.25.0",
        "plotly>=5.17.0",
        "pyyaml>=6.0.1",
        "jsonrpclib-pelix>=0.4.3.2",
        "openpyxl>=3.1.2",
        "werkzeug>=3.0.0",
        "urllib3>=2.0.7",
        "pyinstaller>=5.13.0"
    ]
    
    try:
        for dep in dependencies:
            print(f"Installing {dep}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                  capture_output=True, text=True, check=True)
        
        print("All Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def build_executable(spec_file_path):
    """Build standalone executable using PyInstaller."""
    print("Building standalone executable...")
    
    try:
        # Check if PyInstaller is available
        subprocess.run(["pyinstaller", "--version"], capture_output=True, check=True)
        
        # Build executable using the spec file from App/output
        result = subprocess.run(["pyinstaller", "--clean", str(spec_file_path)], 
                              capture_output=True, text=True, check=True)
        
        print("Executable build successful!")
        print("Executable located at: dist/NetworkDataApp.exe")
        return True
    except FileNotFoundError:
        print("Error: PyInstaller not found. Installing...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
            result = subprocess.run(["pyinstaller", "--clean", str(spec_file_path)], 
                                  capture_output=True, text=True, check=True)
            print("Executable build successful!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to install PyInstaller or build executable: {e}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Executable build failed: {e}")
        if hasattr(e, 'stdout') and e.stdout:
            print(f"stdout: {e.stdout}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def create_requirements():
    """Create requirements.txt in output/ directory."""
    requirements = [
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "requests>=2.31.0",
        "gunicorn>=21.2.0",
        "pandas>=2.1.0",
        "numpy>=1.25.0",
        "plotly>=5.17.0",
        "pyyaml>=6.0.1",
        "jsonrpclib-pelix>=0.4.3.2",
        "openpyxl>=3.1.2",
        "werkzeug>=3.0.0",
        "urllib3>=2.0.7",
        "pyinstaller>=5.13.0"
    ]
    
    # Create the output/ directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the requirements.txt file in output/ directory
    requirements_file_path = output_dir / "requirements.txt"
    with open(requirements_file_path, "w") as f:
        f.write("\n".join(requirements))
    
    print(f"requirements.txt created at: {requirements_file_path}")

def check_backend_files():
    """Check if required backend files exist."""
    backend_dir = Path("backend")
    required_files = ["server.py", "commands.yaml"]
    
    missing_files = []
    for file in required_files:
        if not (backend_dir / file).exists():
            missing_files.append(str(backend_dir / file))
    
    if missing_files:
        print(f"Error: Missing required backend files: {missing_files}")
        return False
    
    print("All required backend files found!")
    return True

def main():
    print("=== Network Data App - Standalone Build ===")
    print("This script will create a standalone executable for the Network Data App")
    print()
    
    # Check backend files
    if not check_backend_files():
        print("Build cannot continue without required backend files.")
        return
    
    # Create requirements.txt in App/output
    create_requirements()
    
    # Install Python dependencies
    if not install_dependencies():
        print("Failed to install Python dependencies. Exiting.")
        return
    
    # Build frontend (with fallback)
    print("\n" + "="*50)
    frontend_success = build_frontend()
    if not frontend_success:
        print("Frontend build had issues, but continuing with fallback...")
    
    # Create PyInstaller spec in App/output
    print("\n" + "="*50)
    spec_file_path = create_pyinstaller_spec()
    
    # Build executable
    print("\n" + "="*50)
    if build_executable(spec_file_path):
        print("\n" + "="*60)
        print("=== Build Complete! ===")
        print("Your standalone executable is ready at: dist/NetworkDataApp.exe")
        print("You can distribute this single file - no installation required!")
        print("\nBuild files created in: output/")
        print("- NetworkDataApp.spec")
        print("- requirements.txt")
        print("\nNote: This app uses jsonrpclib for Arista eAPI compatibility")
        if not frontend_success:
            print("\nWarning: Frontend was built with fallback. Install Node.js for full functionality.")
        print("="*60)
    else:
        print("Build failed. Please check the errors above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBuild cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()