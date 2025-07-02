# Network Data App

A comprehensive network device management tool that automates data collection from network devices via vendor APIs. The application supports both development mode (React + Flask) and production mode (standalone executable) for maximum flexibility.

## Features

- 🔐 **API Authentication** - Secure connections using vendor APIs (eAPI, RESTCONF, etc.)
- 📊 **Real-time Progress Tracking** - Live progress bars and status updates
- 🔄 **Retry Mechanism** - Automatic retry for failed connections
- 📈 **Batch Processing** - Handle large numbers of devices efficiently
- 🎯 **Data Filtering** - Filter results by status, model, or custom criteria
- 📁 **Snapshot Comparison** - Compare configurations and track changes over time
- ⏹️ **Process Control** - Start, stop, and manage processing tasks
- 📤 **Export Capabilities** - Export results to Excel format
- 📊 **Visual Dashboard** - Charts and graphs for data visualization
- 🔧 **Command Selection** - Choose specific commands to execute
- 📋 **Real-time Logging** - Live log streaming and system monitoring
- 🌐 **Multi-vendor Support** - Arista eAPI, Cisco RESTCONF, and more

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Usage Guide](#usage-guide)
5. [Advanced Features](#advanced-features)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)
9. [API Documentation](#api-documentation)
10. [Building Standalone](#building-standalone)
11. [Support](#support)

## Requirements

### Software Requirements

- **Python 3.8+** - Required for backend processing
- **Node.js 16+** - Required for frontend development (development mode only)
- **npm or yarn** - Package manager for frontend dependencies (development mode only)

### System Requirements

- **Windows 10/11** (Primary support)
- **macOS** or **Linux** (Development mode)
- **4GB RAM minimum** (8GB+ recommended for large device lists)
- **Network access** to target devices via HTTPS/HTTP APIs

### Python Dependencies

```
flask>=2.3.0
flask-cors>=4.0.0
pandas>=2.0.0
plotly>=5.15.0
pyyaml>=6.0
jsonrpclib-pelix>=0.4.3
openpyxl>=3.1.0
werkzeug>=2.3.0
urllib3>=2.0.0
```

### Supported Vendor APIs

- **Arista EOS** - eAPI (JSON-RPC)
- **Cisco IOS-XE** - RESTCONF
- **Cisco NX-OS** - REST API
- **More vendors** - Extensible architecture

## Installation

### Quick Setup (Recommended)

1. **Clone or Download** the project to your local machine
2. **Run the build script** for a standalone executable:

   ```bash
   # Start bat file to install both frontend and backend dependencies
   INSTALL.bat
   ```

### Manual Development Setup

#### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. Frontend Setup (Development Mode Only)

```bash
cd frontend
npm install
```

#### 3. Configuration Setup

The application comes with a default commands.yaml file. You can customize it:

```yaml
arista_eos:
  mac_address_table:
    - "show mac address-table"
  ip_arp:
    - "show ip arp"
  interfaces_status:
    - "show interfaces status"
  mlag_interfaces:
    - "show mlag interfaces detail"
  system_info:
    - "show version"
    - "show hostname"
    - "show inventory"
```

## Getting Started

### Step 1: Prepare Your Device List

1. **Create a CSV file** with your device information:

   ```csv
   IP MGMT,Nama SW,SN,Model SW
   192.168.1.1,Switch-Core-01,ABC123456789,DCS-7050QX-32
   192.168.1.2,Switch-Access-01,DEF987654321,DCS-7048T-A
   192.168.1.3,Switch-Spine-01,GHI456789123,DCS-7280CR3-32P6
   10.0.0.1,Switch-Leaf-01,JKL789123456,DCS-7050CX3-32S
   172.16.1.10,Switch-Border-01,MNO345678901,DCS-7280CR3-32D4
   ```

2. **Required CSV Columns:**

   - `IP MGMT` or `ip_mgmt` - Management IP address
   - `Nama SW` or `nama_sw` - Device hostname/name
   - `SN` or `sn` - Serial number
   - `Model SW` or `model_sw` - Device model

3. **Supported Column Variations:**
   - IP: `ip_mgmt`, `ip`, `management ip`, `mgmt_ip`, `device_ip`
   - Name: `nama_sw`, `name`, `hostname`, `device_name`, `switch_name`
   - Serial: `sn`, `serial`, `serial_number`, `serial number`, `serial_no`
   - Model: `model_sw`, `model`, `device_model`, `switch_model`

### Step 2: Start the Application

#### Development Mode

```bash
# Start both frontend and backend
START-HERE.bat

# The application will open two windows:
# - Frontend: http://localhost:5173
# - Backend: http://localhost:5000
```

#### Production Mode (Standalone)

```bash
# Run the built executable
./dist/NetworkDataApp.exe
```

## Usage Guide

### Basic Workflow

#### 1. Upload Device List

1. **Click "Upload CSV File"** or drag and drop your device list
2. **Review warnings** if any column issues are detected
3. **Verify device count** matches your expectations

#### 2. Configure Authentication

1. **Enter API credentials:**
   - Username: Your device username
   - Password: Your device password
2. **These credentials** will be used for all devices in the list

#### 3. Select Commands

1. **Choose commands to execute** from the available options:

   - **MAC Address Table** - Layer 2 forwarding information
   - **IP ARP Table** - Layer 3 address resolution
   - **Interface Status** - Port status and configuration
   - **MLAG Interfaces** - Multi-chassis LAG details
   - **System Info** - Device version and inventory

2. **All commands selected by default** - uncheck to exclude specific commands

#### 4. Start Processing

1. **Click "Start API Collection"**
2. **Monitor real-time progress:**
   - Progress bar with percentage
   - Success/failure counts
   - Individual device status updates
   - Processing speed metrics

#### 5. Review Results

1. **Check the results table** for:

   - ✅ **Successful connections** - API data collected
   - ❌ **Failed connections** - Connection or authentication errors
   - 🔄 **Retry attempts** - Automatic retry status
   - ⏱️ **Processing times** - Performance metrics

2. **View detailed data** by clicking "View" for successful devices
3. **Export results** to Excel for further analysis

### Advanced Usage

#### Retry Failed Devices

**When devices fail:**

1. After processing completes, **"Retry Failed Devices"** button appears
2. Click to reprocess only failed devices
3. Results are merged with existing successful data
4. Failed devices get additional retry attempts

#### Real-time Monitoring

**Progress tracking includes:**

- Overall completion percentage
- Devices processed/remaining
- Success/failure rates
- API response times
- Connection status per device

#### Stop Processing

**To halt processing:**

1. Click **"Stop Processing"** button
2. System gracefully stops after current batch
3. Completed results are preserved
4. Partial results can be exported

## Advanced Features

### 1. Data Filtering and Search

**Filter results by:**

- **Status:** Success/Failed
- **Model:** Device model/type
- **Connection Status:** API connection state

**Search functionality:**

- Real-time search across IP, hostname, and model
- Case-insensitive matching
- Instant results filtering

### 2. Snapshot Comparison

**Compare device states over time:**

1. **Click "Compare Snapshots"**
2. **Select two data collection files**
3. **Choose command category** to compare
4. **Review detailed differences:**
   - Added entries (MAC addresses, ARP entries, etc.)
   - Removed entries
   - Status changes
   - Configuration drift

**Comparison features:**

- Excel export of comparison results
- Device-by-device change analysis
- Summary statistics
- Change categorization

### 3. File Management

**Output Files Manager:**

- View all collected data files
- Download individual files
- Delete old files
- File metadata (size, date, device count)
- Search and filter files

### 4. System Monitoring

**Real-time Logs:**

- Live log streaming during processing
- Filter by log level (INFO, WARNING, ERROR)
- Auto-scroll functionality
- Log clearing capability
- System health monitoring

### 5. Visual Analytics

**Dashboard features:**

- Interactive charts and graphs
- Device distribution by model
- Success/failure analysis
- Processing time metrics
- Customizable grouping options

### 6. Export and Reporting

**Multiple export formats:**

- **Excel exports** with structured data
- **JSON files** for programmatic access
- **Comparison reports** showing changes
- **Metadata inclusion** (timestamps, command selection)

## Configuration

### API Endpoints Configuration

The application automatically detects and configures API endpoints:

**Arista EOS (eAPI):**

- Endpoint: `/command-api`
- Protocol: HTTPS (default) / HTTP
- Port: 443 (HTTPS) / 80 (HTTP)
- Content-Type: `application/json-rpc`

**Cisco IOS-XE (RESTCONF):**

- Endpoint: `/restconf/data`
- Protocol: HTTPS
- Port: 443
- Content-Type: `application/yang-data+json`

### Custom Commands

Edit `backend/commands.yaml` to customize commands:

```yaml
arista_eos:
  # Custom command category
  custom_monitoring:
    - "show processes top"
    - "show version detail"
    - "show environment all"

  # Security commands
  security_status:
    - "show aaa"
    - "show ip access-lists"
    - "show management api http-commands"

  # Performance monitoring
  performance:
    - "show interfaces counters"
    - "show system memory"
    - "show processes cpu"
```

### Application Settings

**Default configuration values:**

- **Timeout:** 30 seconds per device
- **Max Workers:** 10 concurrent connections
- **Retry Attempts:** 3 attempts per failed device
- **Output Directory:** `backend/output/`
- **Protocol:** HTTPS (fallback to HTTP if needed)

## Troubleshooting

### Common Issues

#### "Backend Connection Failed"

**Symptoms:** Frontend shows disconnected status

**Solutions:**

```bash
# Check if Flask server is running
netstat -an | findstr :5000

# Restart backend server
cd backend
python server.py

# Check Windows Firewall settings
# Allow Python through Windows Firewall
```

#### "API Authentication Failed"

**Symptoms:** All devices show authentication errors

**Solutions:**

- Verify API credentials are correct
- Check if eAPI/RESTCONF is enabled on devices
- Ensure proper user privileges for API access
- Test manual API connection: `curl -k https://device_ip/command-api`

#### "Connection Timeout"

**Symptoms:** Devices show timeout errors

**Solutions:**

- Check network connectivity: `ping device_ip`
- Verify API port accessibility (443/80)
- Check device API configuration
- Increase timeout value if needed

#### "CSV Parsing Errors"

**Symptoms:** "Error reading CSV file"

**Solutions:**

- Check CSV file encoding (UTF-8 recommended)
- Verify column headers match requirements
- Remove special characters from device names
- Ensure all required columns are present

#### "Memory Issues with Large Files"

**Symptoms:** Application crashes with many devices

**Solutions:**

- Process in smaller batches (50-100 devices)
- Increase available RAM
- Close other applications
- Use 64-bit Python version

### Log Files and Debugging

**Log locations:**

- **Backend logs:** `backend/logs/network_fetcher_dev.log`
- **Frontend logs:** Browser Developer Console (F12)
- **Real-time logs:** Available in application UI

**Enable debug logging:**

```python
# In server.py, change log level
logging.basicConfig(level=logging.DEBUG)
```

### Network Requirements

**Required connectivity:**

- **HTTPS/HTTP access** to device management interfaces
- **DNS resolution** for device hostnames (if used)
- **Firewall rules** allowing outbound API calls
- **Sufficient bandwidth** for concurrent connections

**Test API connectivity:**

```bash
# Test Arista eAPI
curl -k -X POST https://device_ip/command-api \
  -H "Content-Type: application/json-rpc" \
  -d '{"jsonrpc":"2.0","method":"runCmds","params":{"version":1,"cmds":["show version"],"format":"json"},"id":"test"}' \
  -u username:password

# Test basic connectivity
telnet device_ip 443
nmap -p 443,80 device_ip
```

## Best Practices

### 1. Security Considerations

**Protect API credentials:**

- Use dedicated service accounts with minimal privileges
- Enable API access only on management interfaces
- Use HTTPS whenever possible
- Monitor API access logs on devices
- Implement proper certificate validation

### 2. Performance Optimization

**Speed up processing:**

- Use wired network connections for stability
- Adjust concurrent worker count based on system capacity
- Optimize API timeout values for your network
- Process during low-traffic periods
- Monitor system resources during large collections

### 3. Data Management

**Organize your data:**

- Use consistent device naming conventions
- Create dated backup folders for collections
- Export results regularly to avoid data loss
- Document configuration changes
- Maintain accurate device inventory

### 4. Operational Planning

**Before starting large collections:**

- Test with small device samples first
- Verify credentials on representative devices
- Schedule during maintenance windows
- Plan for processing time based on device count
- Prepare rollback procedures if needed

### 5. Monitoring and Maintenance

**Regular tasks:**

- Review failed device logs for patterns
- Update device credentials as needed
- Clean old export files to save space
- Monitor API rate limits on devices
- Update application dependencies

## API Documentation

### REST Endpoints

**System Information:**

```http
GET /api/system_info
```

**File Upload:**

```http
POST /api/upload_csv
Content-Type: multipart/form-data
```

**Start Processing:**

```json
POST /api/process_devices
Content-Type: application/json
{
  "username": "admin",
  "password": "password",
  "filepath": "/path/to/devices.csv",
  "selected_commands": ["mac_address_table", "ip_arp"]
}
```

**Processing Status:**

```http
GET /api/processing_status/{session_id}
```

**Compare Snapshots:**

```json
POST /api/compare_snapshots
Content-Type: application/json
{
  "first_file": "snapshot1.json",
  "second_file": "snapshot2.json",
  "command_category": "mac_address_table"
}
```

**Output Files:**

```http
GET /api/output_files
GET /api/output_files/{filename}
DELETE /api/output_files/{filename}
```

### WebSocket Events

**Real-time log streaming:**

```http
GET /api/logs/stream
Content-Type: text/event-stream
```

### Building Standalone

**Create Executable**

```bash
# Build standalone application
python build_standalone.py
```

**Build process includes:**

- **Frontend compilation** (if Node.js available)
- **Python dependency installation**
- **PyInstaller executable creation**
- **Asset bundling and optimization**

**Output:** `dist/NetworkDataApp.exe` - Single file executable

### Distribution

**The standalone executable:**

- No installation required
- Includes all dependencies
- Works on Windows without Python
- Self-contained with web interface
- Portable across similar systems

### Integration Examples

**1. Automated Monitoring**
**Schedule regular collections:**

```phyton
# scheduled_collection.py
from apscheduler.schedulers.blocking import BlockingScheduler
import requests
import json

def daily_collection():
    # Upload device list
    with open('devices.csv', 'rb') as f:
        files = {'file': f}
        upload_response = requests.post('http://localhost:5000/api/upload_csv', files=files)

    # Start processing
    process_data = {
        'username': 'admin',
        'password': 'password',
        'filepath': upload_response.json()['filepath'],
        'selected_commands': ['mac_address_table', 'interfaces_status']
    }
    requests.post('http://localhost:5000/api/process_devices', json=process_data)

scheduler = BlockingScheduler()
scheduler.add_job(daily_collection, 'cron', hour=2)  # 2 AM daily
scheduler.start()
```

**2. Change Detection**
**Monitor configuration changes:**

```python
# change_detector.py
import json
from datetime import datetime, timedelta

def detect_changes():
    # Get recent snapshots
    files_response = requests.get('http://localhost:5000/api/output_files')
    files = files_response.json()['data']

    # Compare last two collections
    if len(files) >= 2:
        recent_files = sorted(files, key=lambda x: x['modified'])[-2:]

        compare_data = {
            'first_file': recent_files[0]['filename'],
            'second_file': recent_files[1]['filename'],
            'command_category': 'mac_address_table'
        }

        comparison = requests.post('http://localhost:5000/api/compare_snapshots', json=compare_data)

        # Process changes and send alerts
        if comparison.json()['status'] == 'success':
            changes = [device for device in comparison.json()['data']
                      if device['compare_result']['status'] != 'no_changes']

            if changes:
                send_change_alert(changes)

def send_change_alert(changes):
    # Send email, ticket, or notification
    print(f"Detected {len(changes)} devices with configuration changes")
```

**3. Asset Management Integration**
**Update CMDB with collected data:**

```python
# cmdb_integration.py
def update_cmdb(collection_file):
    with open(collection_file) as f:
        data = json.load(f)

    for device in data['results']:
        if device['status'] == 'Success':
            cmdb_record = {
                'ci_name': device['nama_sw'],
                'ip_address': device['ip_mgmt'],
                'serial_number': device['sn'],
                'model': device['model_sw'],
                'last_discovered': device['last_attempt'],
                'management_status': 'reachable'
            }
            # Update CMDB via API
            update_cmdb_record(cmdb_record)
```

### Support

**Getting Help**

- **Check this documentation** for common solutions
- **Review log files** for detailed error information
- **Test with minimal configurations** to isolate issues
- **Verify network connectivity** independently
- **Check device API configurations** manually

### System Requirements Check

**Verify your environment:**

```bash
# Check Python version
python --version

# Check required packages
pip list | grep -E "(flask|pandas|plotly|jsonrpclib)"

# Test network connectivity
ping target_device_ip
curl -k https://target_device_ip

# Check available memory
# Windows: wmic computersystem get TotalPhysicalMemory
# Linux: free -h
```

### Performance Monitoring

**Monitor during large collections:**

```bash
# Windows Task Manager
# Monitor Python memory usage and CPU

# Linux system monitoring
top -p $(pgrep python)

# Network interface statistics
netstat -i
```

### Community Resources

- **GitHub Repository:** Latest updates and issue tracking
- **API Documentation:** Swagger/OpenAPI specifications
- **Video Tutorials:** Step-by-step usage guides
- **Vendor Documentation:** Device API configuration guides

**Note:** This application connects to network devices via their management APIs (eAPI, RESTCONF, etc.) and requires proper API access configuration. Always follow your organization's security policies and ensure appropriate network access controls are in place.

## File Structure

```
network-data-app/
├── backend/                    # Python backend application
│   ├── commands.yaml           # Configuration file defining network device commands to be executed.
│   ├── server.py               # The core Flask server application, providing RESTful API endpoints for frontend interaction and data processing.
│   └── test_connection.py      # A utility script used for testing SSL-bypassed network connections, likely for debugging or specific network setups.
├── dist/                       # Contains the built, production-ready output of the application.
│   ├── index.html              # The main HTML file for the deployed frontend application.
│   └── NetworkDataApp.exe      # The standalone Windows executable for the entire application (frontend and backend bundled).
├── frontend/                   # The React-based user interface application.
│   ├── assets/                 # Stores static assets like images, fonts, or other media used by the frontend.
│   ├── src/                    # Source code for the React application.
│   │   ├── App.css             # Contains CSS rules for styling the main application components.
│   │   ├── App.jsx             # The primary React component that orchestrates the entire user interface.
│   │   └── main.jsx            # The entry point for the React application, responsible for rendering the `App` component.
│   ├── eslint.config.js        # Configuration file for ESLint, used for static code analysis and enforcing coding standards.
│   ├── index.html              # The base HTML file for the frontend development environment.
│   ├── package-lock.json       # Automatically generated file detailing the exact versions of all frontend dependencies.
│   ├── package.json            # Defines project metadata and lists all direct frontend dependencies.
│   └── vite.config.js          # Configuration file for Vite, a fast build tool for modern web projects.
├── logs/                       # Stores application logs. (gitignored)
├── output/                     # Contains generated output files. (gitignored)
│   ├── network_fetcher_dev.log # A log file specifically for the network data fetching process during development.
│   ├── NetworkDataApp.spec     # A PyInstaller specification file, used to configure how the Python application is bundled into an executable.
│   ├── requirements.txt        # Lists all Python dependencies required for the backend application and the build process.
│   ├── *.json                  # Placeholder for JSON files, which represent snapshots of fetched network device data.
│   └── *.xlsx                  # Placeholder for Excel files, likely containing reports from comparisons or analysis of network data.
├── required/                   # Essential scripts for building and starting the application.
│   ├── build_standalone.py     # A Python script responsible for initiating the process of building the standalone executable.
│   └── start.sh                # A shell script for starting the application, typically used in Linux/macOS environments.
├── uploads/                    # Directory for storing user-uploaded files, such as CSV lists of network devices. (gitignored)
│   └── *.csv                   # Placeholder for CSV files, which likely contain lists of network devices for the application to process.
├── .gitignore                  # Specifies intentionally untracked files and directories that Git should ignore.
├── .railwayignore              # Similar to `.gitignore`, but specifically for Railway deployment to exclude certain files.
├── BUILD.bat                   # A Windows batch script to trigger the build process for the application.
├── Dockerfile                  # Defines the steps to build a Docker image for containerizing the application (optional for deployment).
├── INSTALL.bat                 # A Windows batch script for installing necessary dependencies or setting up the application.
├── LICENSE                     # Contains the licensing information for the project.
├── railway.json                # Configuration file for deploying the application on Railway, a platform as a service.
├── README.md                   # Provides general information about the project, setup instructions, and usage details.
├── RUN.bat                     # A Windows batch script to run the application after it has been built or installed.
└── START-HERE.bat              # A convenience script for new users to quickly get started with the application.
```
