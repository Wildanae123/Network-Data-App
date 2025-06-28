# Network Data App

A comprehensive network device management tool that automates data collection from network devices via SSH. The application supports both development mode (React + Flask) and production mode (PyWebView) for maximum flexibility.

## Features

- 🔐 **Device Authentication** - Secure SSH connections with credential management
- 📊 **Real-time Progress Tracking** - Live progress bars and status updates
- 🔄 **Retry Mechanism** - Automatic retry for failed connections
- 📈 **Batch Processing** - Handle large numbers of devices efficiently
- 🎯 **Data Filtering** - Filter results by status, model, or custom criteria
- 📁 **File Comparison** - Compare configurations and track changes
- ⏹️ **Process Control** - Start, stop, and manage processing tasks
- 📤 **Export Capabilities** - Export results to Excel format
- 📊 **Visual Dashboard** - Charts and graphs for data visualization

## Requirements

### Software Requirements

- **Python 3.8+** - Required for backend processing
- **Node.js 16+** - Required for frontend development
- **npm or yarn** - Package manager for frontend dependencies

### System Requirements

- **Windows 10/11** (Primary support)
- **macOS** or **Linux** (Development mode)
- **4GB RAM minimum** (8GB+ recommended for large device lists)
- **Network access** to target devices via SSH

### Python Dependencies

```
flask==3.0.0
flask-cors==4.0.0
netmiko>=4.0.0
pandas>=1.5.0
plotly>=5.0.0
pyyaml>=6.0
openpyxl>=3.1.0
pywebview>=4.0.0
```

### Node.js Dependencies

```
react>=18.0.0
vite>=4.0.0
plotly.js
lucide-react
```

## Installation

### Quick Setup (Recommended)

1. **Clone or Download** the project to your local machine
2. **Run the setup script** from the project root:

   ```bash
   # Windows
   INSTALL.bat

   # Linux/Mac
   chmod +x INSTALL.sh && ./INSTALL.sh
   ```

### Manual Setup

#### 1. Frontend Setup

```bash
cd frontend
npm install
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/Scripts/activate

# Install dependencies
pip install flask flask-cors netmiko pandas plotly pyyaml openpyxl pywebview
```

#### 3. Configuration Setup

Create a `commands.yaml` file in the `backend` directory:

```yaml
cisco_ios:
  system:
    - "show version"
    - "show running-config | section hostname"
  interfaces:
    - "show ip interface brief"
    - "show interface status"
  routing:
    - "show ip route summary"

cisco_xe:
  system:
    - "show version"
    - "show running-config | section hostname"
  interfaces:
    - "show ip interface brief"
    - "show interface status"
```

## Usage

### Starting the Application

#### Development Mode (Recommended for development)

```bash
# Start both frontend and backend
START-HERE.bat

# Or manually:
# Terminal 1 - Frontend
cd frontend && npm run dev

# Terminal 2 - Backend
cd backend && python dev_server.py
```

#### Production Mode (Standalone application)

```bash
cd backend
python main.py
```

### Preparing Device Data

1. **Create a CSV file** with your device information:

   ```csv
   IP MGMT,Nama SW,SN,Model SW
   192.168.1.1,Switch-Core-01,ABC123456,Catalyst 9300
   192.168.1.2,Switch-Access-01,DEF789012,Catalyst 2960
   ```

2. **Required CSV Columns:**
   - `IP MGMT` or `ip_mgmt` - Management IP address
   - `Nama SW` or `nama_sw` - Device hostname/name
   - `SN` or `sn` - Serial number
   - `Model SW` or `model_sw` - Device model

### Step-by-Step Operation

#### 1. Device Credentials

- Enter SSH username and password
- Credentials are used for all devices in the list
- Leave empty if devices don't require authentication

#### 2. Start Processing

- Click "Start & Select Device File"
- Choose your CSV file from the file dialog
- Processing will begin automatically

#### 3. Monitor Progress

- Watch the real-time progress bar
- View individual device status updates
- Check success/failure counts

#### 4. Review Results

- Examine the results table
- View detailed device information
- Check error messages for failed devices

#### 5. Export Data

- Click "Export to Excel" to save results
- Choose location and filename
- Excel file includes all collected data

### Advanced Features

#### Retry Failed Devices

1. After initial processing completes
2. Click "Retry Failed Devices" button
3. Only failed devices will be reprocessed
4. Results are merged with existing data

#### Filter Results

- Use the filter dropdown to sort by:
  - Device status (Success/Failed)
  - Device model
  - Custom criteria
- Real-time filtering of results table

#### Compare Configurations

1. Export results from different time periods
2. Click "Compare Files" button
3. Select two JSON files to compare
4. View detailed diff showing:
   - Added configurations
   - Removed configurations
   - Modified settings

#### Stop Processing

- Click the "Stop Processing" button anytime
- Gracefully stops all running tasks
- Preserves completed results

### Supported Device Types

The application supports various network device types through Netmiko:

- Cisco IOS/IOS-XE
- Cisco NX-OS
- Arista EOS
- Juniper Junos
- And many more...

Device type is auto-detected during connection.

## Configuration

### Custom Commands

Edit `backend/commands.yaml` to customize commands for each device type:

```yaml
device_type:
  category_name:
    - "command 1"
    - "command 2"
```

### Application Settings

- **Timeout**: Default 20 seconds per device
- **Max Workers**: Default 10 concurrent connections
- **Retry Attempts**: Default 3 attempts per failed device
- **Output Directory**: `backend/output/`

## Troubleshooting

### Common Issues

#### "Backend connection failed"

- Ensure Flask server is running on port 5000
- Check Windows Firewall settings
- Verify virtual environment is activated

#### "No data to export"

- Ensure device processing completed successfully
- Check that CSV file contains valid device entries
- Verify network connectivity to devices

#### "Authentication failed"

- Verify SSH credentials are correct
- Check if devices require enable password
- Ensure SSH is enabled on target devices

#### "Import/Export errors"

- Check file permissions
- Ensure sufficient disk space
- Verify file format (CSV for import, Excel for export)

### Log Files

- Frontend: Browser developer console
- Backend: `network_fetcher.log` and `network_fetcher_dev.log`

### Network Requirements

- SSH access (port 22) to all target devices
- Devices must be reachable from the application host
- Proper network routing and firewall rules

## File Structure

```
network-data-app/
├── frontend/                # React frontend application
│   ├── src/
│   │   ├── App.jsx          # Main application component
│   │   ├── main.jsx         # Entry point
│   │   └── App.css          # Styling
│   ├── package.json         # Frontend dependencies
│   └── dist/                # Built files (after npm run build)
├── backend/                 # Python backend application
│   ├── production_server.py # Production PyWebView app
│   ├── dev_server.py        # Development Flask server
│   ├── commands.yaml        # Device command configuration
│   ├── venv/                # Python virtual environment
│   └── uploads/             # Created automatically
├── output/                  # Generated reports and exports
├── requirements.txt          # Fallback/default requirements
├── requirements-dev.txt      # Development dependencies
├── requirements-prod.txt     # Production dependencies
├── Procfile                  # Railway start command
├── nixpacks.toml            # Railway build configuration
├── railway.json             # Railway deployment config (optional)
├── START-HERE.bat           # Development startup script
├── INSTALL.bat              # Setup script
└── README.md                # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review log files for error details
3. Ensure all requirements are met
4. Create an issue with detailed error information

---

**Note**: This application is designed for network administrators and requires proper SSH access to network devices. Always follow your organization's security policies when using this tool.
