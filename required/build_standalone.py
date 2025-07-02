import os
import sys
import subprocess
import shutil
import stat
import time
from pathlib import Path

# Global variables to store the found paths of npm and node
_npm_path = None
_node_path = None

def check_npm_available():
    """Check if npm is available in the system and return its path."""
    global _npm_path
    try:
        # Try running directly, assuming it's in PATH
        subprocess.run(["npm", "--version"], capture_output=True, check=True)
        _npm_path = "npm" # It's in PATH, so just use "npm"
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try alternative npm locations
        npm_paths = [
            "npm.cmd",  # Windows
            "/usr/local/bin/npm",  # macOS/Linux common locations
            "/usr/bin/npm",
        ]
        
        for npm_candidate_path in npm_paths:
            try:
                subprocess.run([npm_candidate_path, "--version"], capture_output=True, check=True)
                print(f"Found npm at: {npm_candidate_path}")
                _npm_path = npm_candidate_path
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        _npm_path = None # Not found
        return False

def check_node_available():
    """Check if Node.js is available in the system and return its path."""
    global _node_path
    try:
        # Try running directly, assuming it's in PATH
        result = subprocess.run(["node", "--version"], capture_output=True, check=True, text=True)
        print(f"Node.js version: {result.stdout.strip()}")
        _node_path = "node" # It's in PATH, so just use "node"
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Try alternative node locations
        node_paths = [
            "node.exe",  # Windows
            "/usr/local/bin/node",  # macOS/Linux common locations
            "/usr/bin/node",
        ]
        
        for node_candidate_path in node_paths:
            try:
                result = subprocess.run([node_candidate_path, "--version"], capture_output=True, check=True, text=True)
                print(f"Found Node.js at: {node_candidate_path} - version: {result.stdout.strip()}")
                _node_path = node_candidate_path
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                continue
        
        _node_path = None # Not found
        return False

def build_frontend():
    """Build the React frontend for production to root dist/ folder."""
    print("Checking Node.js and npm availability...")
    
    node_available = check_node_available()
    npm_available = check_npm_available()
    
    if not node_available or not npm_available:
        print("Creating fallback frontend...")
        return create_basic_frontend()
    
    # Ensure npm and node paths are set, if not, fallback
    if _npm_path is None or _node_path is None:
        print("Could not determine full paths for Node.js or npm. Creating fallback frontend.")
        return create_basic_frontend()

    print("Building React frontend...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists() or not (frontend_dir / "package.json").exists():
        print("Frontend directory or package.json not found!")
        return create_basic_frontend()
    
    try:
        print("Installing npm dependencies...")
        # Use the stored npm path
        subprocess.run([_npm_path, "install"], cwd=frontend_dir, 
                       capture_output=True, text=True, check=True)
        print("Dependencies installed successfully!")
        
        print("Building production bundle to root dist/...")
        # Use the stored npm path
        subprocess.run([_npm_path, "run", "build"], cwd=frontend_dir, 
                       capture_output=True, text=True, check=True)
        print("Frontend build successful!")
        
        # Verify build went to root dist/
        root_dist = Path("dist")
        if root_dist.exists() and (root_dist / "index.html").exists():
            print(f"Frontend build verified at: {root_dist}")
            return True
        else:
            print("Frontend build failed - no dist/index.html found")
            return create_basic_frontend()
        
    except subprocess.CalledProcessError as e:
        print(f"Frontend build failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return create_basic_frontend()
    except FileNotFoundError as e:
        print(f"Error executing npm/node: {e}. Ensure Node.js and npm are installed and accessible.")
        return create_basic_frontend()


def create_basic_frontend():
    """Create a basic frontend structure in root dist/ folder."""
    print("Creating basic HTML frontend as fallback...")
    
    dist_dir = Path("dist")
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    # Enhanced HTML with auto-connect to backend
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Network Data App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status { padding: 15px; border-radius: 5px; margin: 20px 0; }
        .status.connected { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .status.disconnected { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .api-section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        button { padding: 10px 20px; margin: 10px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .response { background: #f8f9fa; border: 1px solid #e9ecef; padding: 15px; border-radius: 5px; margin: 10px 0; max-height: 300px; overflow-y: auto; }
        pre { margin: 0; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌐 Network Data App</h1>
            <p>Standalone Application - Frontend & Backend</p>
        </div>
        
        <div id="status" class="status disconnected">
            <strong>Status:</strong> Connecting to backend...
        </div>
        
        <div class="api-section">
            <h3>Quick Actions</h3>
            <button onclick="checkStatus()">Check System Status</button>
            <button onclick="listFiles()">List Output Files</button>
            <button onclick="getCommands()">Get Available Commands</button>
            <div id="response" class="response" style="display:none;"></div>
        </div>
        
        <div class="api-section">
            <h3>File Upload</h3>
            <input type="file" id="csvFile" accept=".csv" />
            <button onclick="uploadFile()">Upload CSV</button>
        </div>
        
        <div class="api-section">
            <h3>API Information</h3>
            <p><strong>Backend API:</strong> http://localhost:5000/api/</p>
            <p><strong>Health Check:</strong> <a href="/health" target="_blank">/health</a></p>
            <p><strong>System Info:</strong> <a href="/api/system_info" target="_blank">/api/system_info</a></p>
        </div>
    </div>

    <script>
        let backendConnected = false;
        
        // Check backend connection on load
        window.onload = function() {
            checkConnection();
            setInterval(checkConnection, 30000); // Check every 30 seconds
        };
        
        async function checkConnection() {
            try {
                const response = await fetch('/health');
                if (response.ok) {
                    backendConnected = true;
                    document.getElementById('status').className = 'status connected';
                    document.getElementById('status').innerHTML = '<strong>Status:</strong> ✅ Connected to backend';
                } else {
                    throw new Error('Backend not responding');
                }
            } catch (error) {
                backendConnected = false;
                document.getElementById('status').className = 'status disconnected';
                document.getElementById('status').innerHTML = '<strong>Status:</strong> ❌ Backend disconnected';
            }
        }
        
        async function apiCall(endpoint, options = {}) {
            if (!backendConnected) {
                showResponse('Error: Backend not connected');
                return;
            }
            
            try {
                const response = await fetch('/api/' + endpoint, options);
                const data = await response.json();
                showResponse(JSON.stringify(data, null, 2));
            } catch (error) {
                showResponse('Error: ' + error.message);
            }
        }
        
        function showResponse(text) {
            const responseDiv = document.getElementById('response');
            responseDiv.innerHTML = '<pre>' + text + '</pre>';
            responseDiv.style.display = 'block';
        }
        
        function checkStatus() {
            apiCall('system_info');
        }
        
        function listFiles() {
            apiCall('output_files');
        }
        
        function getCommands() {
            apiCall('comparison_commands');
        }
        
        async function uploadFile() {
            const fileInput = document.getElementById('csvFile');
            if (!fileInput.files[0]) {
                alert('Please select a CSV file first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                const response = await fetch('/api/upload_csv', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                showResponse(JSON.stringify(data, null, 2));
            } catch (error) {
                showResponse('Upload error: ' + error.message);
            }
        }
    </script>
</body>
</html>"""
    
    with open(dist_dir / "index.html", "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Enhanced frontend created at: {dist_dir}")
    return True

def verify_frontend_build():
    """Verify that the frontend build exists and is valid."""
    frontend_dist = Path("frontend") / "dist"
    
    if not frontend_dist.exists():
        print("Error: frontend/dist directory does not exist!")
        
        # Check if build went to root dist instead
        root_dist = Path("dist")
        if root_dist.exists():
            print("Found dist/ in root directory - this might be the frontend build")
            print("Attempting to move it to the correct location...")
            
            try:
                # Create frontend/dist directory
                frontend_dist.mkdir(parents=True, exist_ok=True)
                
                # Move contents from root dist to frontend/dist
                for item in root_dist.iterdir():
                    dest_path = frontend_dist / item.name
                    if item.is_dir():
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(item, dest_path)
                    else:
                        if dest_path.exists():
                            dest_path.unlink()
                        shutil.copy2(item, dest_path)
                
                print(f"Successfully moved frontend build to: {frontend_dist}")
                return True
                
            except Exception as e:
                print(f"Error moving frontend build: {e}")
                return False
        else:
            return False
    
    index_file = frontend_dist / "index.html"
    if not index_file.exists():
        print("Error: frontend/dist/index.html does not exist!")
        return False
    
    print(f"Frontend build verified: {frontend_dist} exists with required files")
    return True

def create_pyinstaller_spec():
    """Create PyInstaller spec file in output/ directory."""
    
    # Verify frontend build exists in root dist/
    root_dist = Path("dist")
    if not root_dist.exists() or not (root_dist / "index.html").exists():
        print("Creating fallback frontend before proceeding...")
        create_basic_frontend()
    
    # Get absolute paths for the backend components and frontend dist
    project_root = Path(os.getcwd()) # This should be the network-data-app/ directory
    server_script_abs_path = project_root / "backend" / "server.py"
    commands_yaml_abs_path = project_root / "backend" / "commands.yaml"
    dist_abs_path = project_root / "dist"

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{server_script_abs_path.as_posix()}'], # Use absolute path here
    pathex=[],
    binaries=[],
    datas=[
        ('{dist_abs_path.as_posix()}', 'dist'), # Use absolute path for source, target is 'dist' in exe
        ('{commands_yaml_abs_path.as_posix()}', 'backend'), # Use absolute path for source, target is 'backend' in exe
    ],
    hiddenimports=[
        'flask',
        'flask_cors',
        'requests',
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
        'importlib.metadata',
        'numpy',
        'jsonrpclib.jsonrpc',
        'webbrowser'
    ],
    hookspath=[],
    hooksconfig={{}},
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
    
    # Final verification before building
    if not verify_frontend_build():
        print("Error: Frontend build verification failed before executable build!")
        return False
    
    try:
        # Check if PyInstaller is available
        subprocess.run(["pyinstaller", "--version"], capture_output=True, check=True)
        
        # Print current working directory for debugging
        print(f"Current working directory for PyInstaller: {os.getcwd()}")

        # Build executable using the spec file from output/
        print(f"Using spec file: {spec_file_path}")
        result = subprocess.run(["pyinstaller", "--clean", str(spec_file_path)], 
                                 capture_output=True, text=True, check=True)
        
        print("Executable build successful!")
        
        # Check if executable was created
        exe_path = Path("dist") / "NetworkDataApp.exe"
        if exe_path.exists():
            print(f"Executable located at: {exe_path}")
            print(f"Executable size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        else:
            # Check for other platforms
            exe_path = Path("dist") / "NetworkDataApp"
            if exe_path.exists():
                print(f"Executable located at: {exe_path}")
                print(f"Executable size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            else:
                print("Warning: Executable file not found in expected location")
        
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
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"Executable build failed: Command: {' '.join(e.cmd)}")
        print(f"Return Code: {e.returncode}")
        print(f"stdout:\n{e.stdout}")
        print(f"stderr:\n{e.stderr}")
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

def force_remove_readonly(func, path, exc_info):
    """Error handler for removing read-only files on Windows."""
    if os.path.exists(path):
        # Change the file to be writable, then remove it
        os.chmod(path, stat.S_IWRITE)
        func(path)

def safe_remove_directory(dir_path, max_retries=3):
    """Safely remove a directory with retry logic for Windows."""
    if not dir_path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to remove {dir_path} (attempt {attempt + 1})")
            
            # First, try to make all files writable
            try:
                for root, dirs, files in os.walk(dir_path):
                    for dir_name in dirs:
                        dir_full_path = os.path.join(root, dir_name)
                        try:
                            os.chmod(dir_full_path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
                        except:
                            pass
                    for file_name in files:
                        file_full_path = os.path.join(root, file_name)
                        try:
                            os.chmod(file_full_path, stat.S_IWRITE | stat.S_IREAD)
                        except:
                            pass
            except:
                pass
            
            # Try to remove the directory
            shutil.rmtree(dir_path, onerror=force_remove_readonly)
            print(f"Successfully removed {dir_path}")
            return True
            
        except PermissionError as e:
            print(f"Permission error removing {dir_path}: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting 2 seconds before retry...")
                time.sleep(2)
            continue
        except Exception as e:
            print(f"Error removing {dir_path}: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting 2 seconds before retry...")
                time.sleep(2)
            continue
    
    print(f"Warning: Could not remove {dir_path} after {max_retries} attempts")
    return False

def clean_previous_builds():
    """Clean up previous build artifacts with improved error handling."""
    print("Cleaning previous build artifacts...")
    
    # Remove dist and build directories (but preserve frontend build if it exists)
    root_dist = Path("dist")
    if root_dist.exists():
        # Check if this might be a frontend build
        has_html = any(f.suffix == '.html' for f in root_dist.rglob('*') if f.is_file())
        has_js = any(f.suffix == '.js' for f in root_dist.rglob('*') if f.is_file())
        
        if has_html or has_js:
            print(f"Found potential frontend build in {root_dist}")
            # Move it to frontend/dist before cleaning
            frontend_dist = Path("frontend") / "dist"
            if not frontend_dist.exists():
                try:
                    frontend_dist.mkdir(parents=True, exist_ok=True)
                    for item in root_dist.iterdir():
                        dest_path = frontend_dist / item.name
                        if item.is_dir():
                            shutil.copytree(item, dest_path, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, dest_path)
                    print(f"Preserved frontend build in {frontend_dist}")
                except Exception as e:
                    print(f"Warning: Could not preserve frontend build: {e}")
        
        # Now remove the root dist
        success = safe_remove_directory(root_dist)
        if not success:
            print(f"Warning: Could not fully clean {root_dist}")
    
    # Remove build directory
    build_dir = Path("build")
    if build_dir.exists():
        success = safe_remove_directory(build_dir)
        if not success:
            print(f"Warning: Could not fully clean {build_dir}")
    
    # Remove any .spec files in root directory
    try:
        for spec_file in Path(".").glob("*.spec"):
            print(f"Removing {spec_file}")
            try:
                os.chmod(spec_file, stat.S_IWRITE)
                spec_file.unlink()
            except Exception as e:
                print(f"Warning: Could not remove {spec_file}: {e}")
    except Exception as e:
        print(f"Warning: Error during spec file cleanup: {e}")
    
    print("Cleanup completed")

def main():
    # Change current working directory to the project root
    # This assumes build_standalone.py is in network-data-app/required/
    # and the project root is network-data-app/
    os.chdir(Path(__file__).parent.parent)

    print("=== Network Data App - Standalone Build ===")
    print("This script will create a standalone executable for the Network Data App")
    print()
    
    # Clean previous builds
    clean_previous_builds()
    
    # Check backend files
    if not check_backend_files():
        print("Build cannot continue without required backend files.")
        return
    
    # Create requirements.txt in output/
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
    
    # Verify frontend build one more time
    if not verify_frontend_build():
        print("Final frontend verification failed!")
        return
    
    # Create PyInstaller spec in output/
    print("\n" + "="*50)
    spec_file_path = create_pyinstaller_spec()
    
    # Build executable
    print("\n" + "="*50)
    if build_executable(spec_file_path):
        print("\n" + "="*60)
        print("=== Build Complete! ===")
        
        # Check which executable was created
        exe_windows = Path("dist") / "NetworkDataApp.exe"
        exe_unix = Path("dist") / "NetworkDataApp"
        
        if exe_windows.exists():
            print(f"Your standalone executable is ready at: {exe_windows}")
        elif exe_unix.exists():
            print(f"Your standalone executable is ready at: {exe_unix}")
        else:
            print("Executable created but location may vary by platform")
        
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

