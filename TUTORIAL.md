# Network Data App - Complete Tutorial

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Troubleshooting](#troubleshooting)
5. [Best Practices](#best-practices)
6. [Integration Examples](#integration-examples)
7. [Support and Resources](#support-and-resources)

## Getting Started

### Step 1: Installation

#### Quick Installation (Recommended)

```bash
# Run the setup script
./INSTALL.bat  # Windows
# or
chmod +x INSTALL.sh && ./INSTALL.sh  # Linux/Mac
```

#### Manual Installation

```bash
# 1. Setup Frontend
cd frontend
npm install

# 2. Setup Backend
cd ../backend
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2: Prepare Your Device List

Create a CSV file with your network devices:

**Example: devices.csv**

```csv
IP MGMT,Nama SW,SN,Model SW
192.168.1.1,Core-Switch-01,ABC123456789,Cisco Catalyst 9300
192.168.1.2,Access-Switch-01,DEF987654321,Cisco Catalyst 2960
192.168.1.3,Distribution-01,GHI456789123,Cisco Catalyst 3850
10.0.0.1,Router-WAN-01,JKL789123456,Cisco ISR 4321
172.16.1.10,Firewall-01,MNO345678901,Cisco ASA 5516
```

**Required Columns:**

- `IP MGMT` - Management IP address
- `Nama SW` - Device hostname/name
- `SN` - Serial number
- `Model SW` - Device model

**Supported Column Variations:**

- IP: `ip_mgmt`, `ip`, `management ip`
- Name: `nama_sw`, `name`, `hostname`, `device_name`
- Serial: `sn`, `serial`, `serial_number`
- Model: `model_sw`, `model`, `device_model`

## Basic Usage

### Step 1: Start the Application

```bash
# Start both frontend and backend
START-HERE.bat

# The application will open two windows:
# - Frontend: http://localhost:5173
# - Backend: http://localhost:5000
```

### Step 2: Configure Device Credentials

1. **Enter SSH Credentials:**

   - Username: Your SSH username
   - Password: Your SSH password
   - These will be used for all devices

2. **Note:** You can leave credentials empty if:
   - Devices don't require authentication
   - You'll provide them later during processing

### Step 3: Start Processing

1. **Click "Start & Select Device File"**
2. **Select your CSV file** from the file dialog
3. **Monitor the progress:**
   - Real-time progress bar
   - Live device count updates
   - Success/failure statistics

### Step 4: Review Results

1. **Check the results table** for:

   - ✅ Successful connections
   - ❌ Failed connections
   - 🔄 Retry attempts
   - ⏱️ Processing times

2. **View detailed data** by clicking "View" for successful devices

3. **Export results** to Excel for further analysis

## Advanced Features

### 1. Retry Mechanism

**When devices fail:**

1. The system automatically retries 3 times
2. After processing completes, click **"Retry Failed Devices"**
3. Only failed devices are reprocessed
4. Results are merged with existing data

**Retry Process:**

- Automatic exponential backoff
- Individual retry counters per device
- Preserve successful results

### 2. Progress Tracking

**Real-time monitoring includes:**

- Overall progress percentage
- Devices completed/remaining
- Success/failure counts
- Processing speed
- Estimated completion time

**Progress visualization:**

- Animated progress bar
- Live statistics
- Success rate calculation

### 3. Data Filtering

**Filter by:**

- **Status:** Success/Failed
- **Model:** Device model/type
- **Connection Status:** success/failed/connecting/retrying

**Steps:**

1. Select filter type from dropdown
2. Enter filter value
3. Results update immediately
4. Use search box for additional filtering

### 4. Stop Processing

**To stop processing:**

1. Click **"Stop Processing"** button
2. System gracefully stops after current device
3. Completed results are preserved
4. You can export partial results

### 5. File Comparison

**Compare configurations from different time periods:**

1. **Click "Compare Files"**
2. **Select two JSON export files** to compare
3. **Review comparison results:**
   - **No Changes:** Configuration identical
   - **Changed:** Modified settings highlighted
   - **Added:** New configuration lines
   - **Removed:** Deleted configuration lines

**Comparison Features:**

- Line-by-line configuration diff
- Color-coded changes (green=added, red=removed, yellow=modified)
- Search within comparison results
- Export comparison report

### 6. Batch Processing

**For large device lists (100+ devices):**

- Automatic parallel processing (10 concurrent connections)
- Memory-efficient streaming
- Progress checkpoints
- Resume capability after interruption

**Best practices:**

- Process during maintenance windows
- Monitor network bandwidth
- Use appropriate timeout values
- Test with small batches first

### 7. Status Tracking

**Device connection statuses:**

- 🔵 **Pending:** Waiting to be processed
- 🟡 **Connecting:** Establishing SSH connection
- 🟢 **Success:** Data collected successfully
- 🔴 **Failed:** Connection/collection failed
- 🟠 **Retrying:** Attempting reconnection
- ⚫ **Stopped:** Processing halted by user

### 8. Custom Commands Configuration

**Edit `backend/commands.yaml` to customize:**

```yaml
cisco_ios:
  system:
    - "show version"
    - "show running-config | section hostname"
    - "show inventory"
  interfaces:
    - "show ip interface brief"
    - "show interface status"
    - "show interface description"
  routing:
    - "show ip route summary"
    - "show ip protocols"
  security:
    - "show access-lists"
    - "show ip access-lists"

cisco_xe:
  system:
    - "show version"
    - "show license summary"
  monitoring:
    - "show processes cpu"
    - "show memory statistics"

arista_eos:
  system:
    - "show version"
    - "show hostname"
  interfaces:
    - "show interfaces status"
    - "show ip interface brief"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "Backend connection failed"

**Symptoms:** Frontend shows disconnected status
**Solutions:**

```bash
# Check if Flask server is running
netstat -an | findstr :5000

# Restart backend server
cd backend
python dev_server.py

# Check firewall settings
# Windows: Allow Python through Windows Firewall
# Linux: Check iptables rules
```

#### 2. "Authentication failed"

**Symptoms:** All devices show authentication errors
**Solutions:**

- Verify SSH credentials are correct
- Check if devices require enable password
- Ensure SSH is enabled on target devices
- Test manual SSH connection: `ssh username@device_ip`

#### 3. "Connection timeout"

**Symptoms:** Devices show timeout errors
**Solutions:**

- Increase timeout value in configuration
- Check network connectivity: `ping device_ip`
- Verify SSH port (default 22) is accessible
- Check for network firewalls/ACLs

#### 4. "CSV parsing errors"

**Symptoms:** "Error reading CSV file"
**Solutions:**

- Check CSV file encoding (UTF-8 recommended)
- Verify column headers match requirements
- Remove special characters from device names
- Check for missing required columns

#### 5. "Memory issues with large files"

**Symptoms:** Application crashes with many devices
**Solutions:**

- Process in smaller batches (50-100 devices)
- Increase available RAM
- Close other applications
- Use 64-bit Python version

### Log Files

**Check these files for detailed error information:**

- Frontend: Browser Developer Console (F12)
- Backend: `backend/network_fetcher.log`
- System: Windows Event Viewer / Linux syslog

### Network Requirements

**Ensure the following connectivity:**

- SSH access (port 22) to all target devices
- DNS resolution for device hostnames
- Proper routing between application host and devices
- Sufficient bandwidth for concurrent connections

## Best Practices

### 1. Preparation

**Before starting:**

- Test with 2-3 devices first
- Verify credentials on sample devices
- Check network connectivity
- Prepare device list in correct format
- Schedule during maintenance windows

### 2. Processing Strategy

**For optimal results:**

- Start with smaller batches (10-20 devices)
- Gradually increase batch size
- Monitor system resources
- Use appropriate timeout values
- Save intermediate results

### 3. Security Considerations

**Protect credentials:**

- Use dedicated service accounts
- Implement least-privilege access
- Avoid storing passwords in files
- Use SSH keys where possible
- Monitor access logs

### 4. Performance Optimization

**Speed up processing:**

- Use wired network connections
- Increase concurrent worker threads (if system allows)
- Optimize SSH timeout values
- Use local DNS servers
- Close unnecessary applications

### 5. Data Management

**Organize your data:**

- Use consistent naming conventions
- Create dated backup folders
- Export results regularly
- Document configuration changes
- Maintain device inventory

### 6. Monitoring and Maintenance

**Regular tasks:**

- Review failed device logs
- Update device credentials
- Clean old export files
- Monitor disk space
- Update application dependencies

## Advanced Configuration

### Custom Device Types

**Add support for new device types in `commands.yaml`:**

```yaml
fortinet_fortios:
  system:
    - "get system status"
    - "get system interface"
  security:
    - "show firewall policy"
    - "show vpn ipsec tunnel"

paloalto_panos:
  system:
    - "show system info"
    - "show interface all"
  security:
    - "show security-policy-match"
    - "show vpn tunnel"

juniper_junos:
  system:
    - "show version"
    - "show chassis hardware"
  configuration:
    - "show configuration | display set"
  interfaces:
    - "show interfaces terse"
```

### Environment Variables

**Configure via environment variables:**

```bash
# Windows
set NETWORK_FETCHER_TIMEOUT=30
set NETWORK_FETCHER_MAX_WORKERS=15
set NETWORK_FETCHER_LOG_LEVEL=DEBUG

# Linux/Mac
export NETWORK_FETCHER_TIMEOUT=30
export NETWORK_FETCHER_MAX_WORKERS=15
export NETWORK_FETCHER_LOG_LEVEL=DEBUG
```

### Automation Scripts

**Create batch processing scripts:**

```bash
#!/bin/bash
# automated_collection.sh

# Set variables
CSV_FILE="/path/to/devices.csv"
USERNAME="admin"
PASSWORD="password"
OUTPUT_DIR="/path/to/outputs"

# Start processing
python dev_server.py \
  --csv-file "$CSV_FILE" \
  --username "$USERNAME" \
  --password "$PASSWORD" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 20 \
  --timeout 30
```

## Integration Examples

### 1. Integration with Network Monitoring

**Export to monitoring systems:**

```python
# Export to CSV for monitoring tool import
import pandas as pd

# Load results
df = pd.read_json('collected_data.json')

# Transform for monitoring system
monitoring_data = df[['ip_mgmt', 'nama_sw', 'status', 'processing_time']]
monitoring_data.to_csv('monitoring_import.csv', index=False)
```

### 2. Integration with Configuration Management

**Compare with baseline configurations:**

```python
# Compare current config with baseline
baseline_file = 'baseline_configs.json'
current_file = 'current_configs.json'

# Use built-in comparison feature
# Results show configuration drift
```

### 3. Integration with Ticketing Systems

**Create tickets for failed devices:**

```python
# Generate ticket data for failed devices
failed_devices = df[df['status'] == 'Failed']
for device in failed_devices:
    ticket_data = {
        'title': f"SSH Access Issue - {device['nama_sw']}",
        'description': f"Failed to connect to {device['ip_mgmt']}: {device['error']}",
        'priority': 'Medium',
        'category': 'Network'
    }
    # Submit to ticketing API
```

### 4. Integration with SIEM Systems

**Export security events:**

```python
# Export authentication failures to SIEM
auth_failures = df[df['error'].str.contains('Authentication', na=False)]
siem_events = []

for device in auth_failures:
    event = {
        'timestamp': device['last_attempt'],
        'source_ip': device['ip_mgmt'],
        'event_type': 'authentication_failure',
        'severity': 'medium',
        'description': device['error']
    }
    siem_events.append(event)

# Send to SIEM via API or syslog
```

### 5. Integration with Asset Management

**Update CMDB with device information:**

```python
# Update Configuration Management Database
for device in successful_devices:
    cmdb_record = {
        'ci_name': device['nama_sw'],
        'ip_address': device['ip_mgmt'],
        'serial_number': device['sn'],
        'model': device['model_sw'],
        'last_discovered': device['last_attempt'],
        'status': 'active' if device['status'] == 'Success' else 'unreachable'
    }
    # Update CMDB via API
```

## Scheduled Operations

### Setting up Automated Collections

**Create scheduled tasks for regular data collection:**

```python
# scheduled_collection.py
from apscheduler.schedulers.blocking import BlockingScheduler
import subprocess
import datetime

def run_daily_collection():
    """Run daily device collection"""
    print(f"Starting daily collection at {datetime.datetime.now()}")

    # Run collection script
    result = subprocess.run([
        'python', 'dev_server.py',
        '--csv-file', 'devices_daily.csv',
        '--username', 'admin',
        '--password', 'password123',
        '--output-dir', f'daily_reports/{datetime.date.today()}'
    ])

    if result.returncode == 0:
        print("Daily collection completed successfully")
        # Send success notification
    else:
        print("Daily collection failed")
        # Send failure alert

# Setup scheduler
scheduler = BlockingScheduler()
scheduler.add_job(run_daily_collection, 'cron', hour=2, minute=0)  # Run at 2 AM daily

try:
    scheduler.start()
except KeyboardInterrupt:
    print("Scheduler stopped")
```

### Windows Task Scheduler Integration

**Create Windows scheduled task:**

```batch
# create_scheduled_task.bat
schtasks /create /tn "Network Data Collection" ^
  /tr "C:\path\to\backend\venv\Scripts\python.exe C:\path\to\scheduled_collection.py" ^
  /sc daily /st 02:00 /ru SYSTEM
```

### Linux Cron Integration

**Add to crontab:**

```bash
# Edit crontab
crontab -e

# Add daily collection at 2 AM
0 2 * * * /path/to/backend/venv/bin/python /path/to/scheduled_collection.py >> /var/log/network_collection.log 2>&1
```

## Performance Tuning

### Hardware Recommendations

**Minimum Requirements:**

- CPU: 2 cores, 2.0 GHz
- RAM: 4 GB
- Storage: 10 GB free space
- Network: 100 Mbps

**Recommended for Large Deployments:**

- CPU: 8 cores, 3.0 GHz
- RAM: 16 GB
- Storage: 100 GB SSD
- Network: 1 Gbps

### Tuning Parameters

**Optimize for your environment:**

```python
# backend/config.py
PERFORMANCE_CONFIG = {
    'MAX_WORKERS': 20,          # Increase for powerful systems
    'TIMEOUT': 15,              # Reduce for fast networks
    'RETRY_ATTEMPTS': 2,        # Reduce for reliable networks
    'BATCH_SIZE': 100,          # Increase for large deployments
    'MEMORY_LIMIT': '8GB',      # Set based on available RAM
    'LOG_LEVEL': 'INFO'         # Use ERROR for production
}
```

### Database Backend (Advanced)

**For large-scale deployments, consider database backend:**

```python
# database_backend.py
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

# Database connection
engine = sa.create_engine('postgresql://user:pass@localhost/netfetcher')
Session = sessionmaker(bind=engine)

class DeviceResult(Base):
    __tablename__ = 'device_results'

    id = sa.Column(sa.Integer, primary_key=True)
    ip_mgmt = sa.Column(sa.String(15), nullable=False)
    nama_sw = sa.Column(sa.String(255))
    status = sa.Column(sa.String(20))
    data = sa.Column(sa.JSON)
    created_at = sa.Column(sa.DateTime, default=datetime.utcnow)

# Store results in database instead of JSON files
def store_results(results):
    session = Session()
    for result in results:
        db_result = DeviceResult(**result)
        session.add(db_result)
    session.commit()
```

## Support and Resources

### Getting Help

1. **Check the troubleshooting section** above
2. **Review log files** for detailed error messages
3. **Test network connectivity** manually
4. **Verify device configurations** independently
5. **Create minimal test cases** to isolate issues

### Useful Commands

**Test SSH connectivity:**

```bash
# Manual SSH test
ssh -v username@device_ip

# Test with timeout
timeout 10 ssh username@device_ip "show version"

# Test multiple devices
for ip in 192.168.1.{1..10}; do
  echo "Testing $ip"
  timeout 5 ssh username@$ip "show version" && echo "OK" || echo "FAILED"
done
```

**Monitor network traffic:**

```bash
# Monitor SSH connections
netstat -an | grep :22

# Watch network traffic (Linux)
tcpdump -i eth0 port 22

# Check DNS resolution
nslookup device_hostname
dig device_hostname
```

**System resource monitoring:**

```bash
# Windows
tasklist | findstr python
perfmon

# Linux
top -p $(pgrep python)
htop
iotop
iostat 1
```

### Community and Documentation

**Additional Resources:**

- GitHub repository with latest updates
- Community forum for questions and discussions
- Video tutorials on YouTube
- API documentation (Swagger/OpenAPI)
- Netmiko documentation for device support
- Plotly documentation for customizing charts

### Reporting Issues

**When reporting bugs or issues:**

1. Include log files (`network_fetcher.log`)
2. Provide CSV sample (anonymized)
3. Specify operating system and Python version
4. Include error messages and stack traces
5. Describe steps to reproduce the issue

### Feature Requests

**To request new features:**

1. Check existing feature list
2. Describe the use case
3. Provide implementation suggestions
4. Consider contributing code

This comprehensive tutorial covers all aspects of using the Network Data App effectively, from basic usage to advanced enterprise deployments. Start with the basic features and gradually explore advanced capabilities as your needs grow.
