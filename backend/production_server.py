# backend/production_server.py

import os
import json
import threading
import concurrent.futures
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import difflib
import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from netmiko import ConnectHandler
from netmiko.ssh_autodetect import SSHDetect
from netmiko.ssh_dispatcher import CLASS_MAPPER
from netmiko.exceptions import NetmikoBaseException, NetmikoAuthenticationException, NetmikoTimeoutException
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import tempfile
import io
import base64
from werkzeug.utils import secure_filename
from collections import defaultdict

# Constants
DEFAULT_TIMEOUT = 20
MAX_WORKERS = 10
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
SUPPORTED_FILE_TYPES = ("CSV Files (*.csv)",)
SUPPORTED_EXPORT_TYPES = ("JSON Files (*.json)",)
EXCEL_ENGINE = "openpyxl"

# Vendor detection mapping based on Model SW
VENDOR_DETECTION_MAP = {
    'cisco_ios': ['ISR', 'C1100', 'C1000', 'C2900', 'C3900', 'CAT', 'WS-C', 'C9200', 'C9300', 'C9400', 'C9500', 'C3850', 'C3750'],
    'cisco_nxos': ['N9K', 'N7K', 'N5K', 'N3K', 'N2K', 'Nexus'],
    'cisco_xe': ['ASR', 'CSR', 'ISR4', 'C8000'],
    'cisco_xr': ['ASR9', 'NCS', 'CRS'],
    'arista_eos': ['DCS', 'CCS', 'Arista'],
    'juniper_junos': ['EX', 'QFX', 'MX', 'SRX', 'ACX'],
    'hp_procurve': ['ProCurve', 'Aruba', 'HP'],
    'dell_force10': ['S4048', 'S5048', 'S6010', 'Z9100'],
    'huawei': ['S5700', 'S6700', 'S7700', 'S9700', 'CE'],
    'fortinet': ['FGT', 'FortiGate'],
}

# Configure logging
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Create log file handler that stores logs in memory
class InMemoryLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.max_logs = 1000  # Keep last 1000 log entries
    
    def emit(self, record):
        self.logs.append({
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': self.format(record),
            'module': record.module
        })
        # Keep only the last max_logs entries
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def get_logs(self):
        return self.logs
    
    def clear_logs(self):
        self.logs = []

# Create in-memory log handler
in_memory_handler = InMemoryLogHandler()
in_memory_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'network_fetcher.log')),
        logging.StreamHandler(),
        in_memory_handler
    ]
)
logger = logging.getLogger(__name__)

# Device Status Constants
DEVICE_STATUS = {
    "PENDING": "pending",
    "CONNECTING": "connecting", 
    "SUCCESS": "success",
    "FAILED": "failed",
    "RETRYING": "retrying",
    "STOPPED": "stopped"
}

@dataclass
class DeviceInfo:
    """Data class for device information."""
    host: str
    device_type: str = "autodetect"
    username: str = ""
    password: str = ""
    conn_timeout: int = DEFAULT_TIMEOUT
    
@dataclass
class DeviceMetadata:
    """Data class for device metadata."""
    ip_mgmt: str
    nama_sw: str
    sn: str
    model_sw: str
    
@dataclass
class ProcessingResult:
    """Data class for processing results."""
    ip_mgmt: str
    nama_sw: str
    sn: str
    model_sw: str
    status: str
    data: Optional[Dict] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    retry_count: int = 0
    last_attempt: Optional[str] = None
    connection_status: str = DEVICE_STATUS["PENDING"]
    detected_device_type: Optional[str] = None

@dataclass
class ProcessingSession:
    """Data class for tracking processing sessions."""
    session_id: str
    total_devices: int
    completed: int = 0
    successful: int = 0
    failed: int = 0
    retrying: int = 0
    is_stopped: bool = False
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    output_file: Optional[str] = None

class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_file: str = "commands.yaml"):
        self.config_file = config_file
        self.config_data = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            script_dir = Path(__file__).parent
            file_path = script_dir / self.config_file
            
            if not file_path.exists():
                logger.warning(f"Configuration file not found: {file_path}, creating default config")
                self._create_default_config(file_path)
            
            with open(file_path, "r", encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            self.config_data = self._transform_config(raw_config)
            
            logger.info(f"Configuration loaded successfully from {file_path}")
            logger.debug(f"Loaded device types: {list(self.config_data.keys())}")
            return self.config_data
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def _create_default_config(self, file_path):
        """Create a default configuration file if none exists."""
        default_config = {
            "cisco_ios": {
                "system_info": [
                    "show version",
                    "show running-config | include hostname",
                    "show inventory"
                ],
                "interface_status": [
                    "show ip interface brief",
                    "show interfaces status"
                ],
                "routing": [
                    "show ip route summary",
                    "show ip route"
                ]
            },
            "cisco_nxos": {
                "system_info": [
                    "show version",
                    "show hostname",
                    "show inventory"
                ],
                "interface_status": [
                    "show ip interface brief",
                    "show interface status"
                ],
                "routing": [
                    "show ip route summary",
                    "show ip route"
                ]
            },
            "arista_eos": {
                "system_info": [
                    "show version",
                    "show hostname",
                    "show inventory"
                ],
                "interface_status": [
                    "show ip interface brief",
                    "show interface status"
                ],
                "routing": [
                    "show ip route summary",
                    "show ip route"
                ]
            }
        }
        
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created default configuration file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to create default config: {e}")
    
    def _transform_config(self, raw_config):
        """Transform flat YAML config to nested structure expected by the application."""
        if not raw_config:
            return {}
            
        transformed = {}
        
        for device_type, categories in raw_config.items():
            if isinstance(categories, dict):
                transformed[device_type] = categories
                logger.debug(f"Device type '{device_type}' has categories: {list(categories.keys())}")
            else:
                logger.warning(f"Invalid configuration structure for device type: {device_type}")
                
        return transformed
    
    def get_commands_for_device(self, device_type: str):
        """Get commands for specific device type."""
        commands = self.config_data.get(device_type)
        if commands:
            logger.debug(f"Found {len(commands)} categories for device type '{device_type}'")
        else:
            logger.warning(f"No commands found for device type '{device_type}'")
            logger.debug(f"Available device types: {list(self.config_data.keys())}")
        return commands
    
    def get_supported_device_types(self):
        """Get list of supported device types."""
        return list(self.config_data.keys())


class NetworkDeviceManager:
    """Network device manager with retry and progress tracking."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def detect_device_type_by_model(self, model_sw: str) -> str:
        """Detect device type based on model string."""
        model_upper = model_sw.upper()
        
        for device_type, patterns in VENDOR_DETECTION_MAP.items():
            for pattern in patterns:
                if pattern.upper() in model_upper:
                    logger.info(f"Detected device type '{device_type}' for model '{model_sw}'")
                    return device_type
        
        logger.warning(f"Could not detect device type for model '{model_sw}', using autodetect")
        return "autodetect"
    
    def connect_and_collect_data(self, device_info: DeviceInfo, model_sw: str = None, retry_count: int = 0, session: ProcessingSession = None):
        """Connect to device and collect data with error handling."""
        collected_data = {}
        start_time = datetime.now()
        detected_device_type = None
        
        try:
            # First try to detect device type from model
            if model_sw and device_info.device_type == "autodetect":
                detected_type = self.detect_device_type_by_model(model_sw)
                if detected_type != "autodetect":
                    device_info.device_type = detected_type
            
            logger.info(f"Connecting to device: {device_info.host} with type: {device_info.device_type} (attempt {retry_count + 1})")
            
            # Create connection parameters
            connection_params = asdict(device_info)
            
            # If still autodetect, use SSH autodetect
            if device_info.device_type == "autodetect":
                try:
                    logger.info(f"Running SSH autodetect for {device_info.host}")
                    guesser = SSHDetect(**connection_params)
                    best_match = guesser.autodetect()
                    if best_match:
                        device_info.device_type = best_match
                        connection_params['device_type'] = best_match
                        logger.info(f"Autodetected device type: {best_match}")
                    guesser.disconnect()
                except Exception as e:
                    logger.warning(f"Autodetect failed for {device_info.host}: {e}")
                    # Try common types
                    device_info.device_type = "cisco_ios"  # Default fallback
                    connection_params['device_type'] = "cisco_ios"
            
            detected_device_type = connection_params['device_type']
            
            with ConnectHandler(**connection_params) as connection:
                logger.info(f"Successfully connected to {device_info.host} as {detected_device_type}")
                
                commands_by_category = self.config_manager.get_commands_for_device(detected_device_type)
                
                if not commands_by_category:
                    error_msg = f"Device type '{detected_device_type}' not supported in configuration"
                    logger.warning(error_msg)
                    return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type
                
                # Execute commands by category
                for category, commands in commands_by_category.items():
                    logger.debug(f"Processing category '{category}' with {len(commands)} commands")
                    collected_data[category] = {}
                    
                    for command in commands:
                        # Check if session is stopped
                        if session and session.is_stopped:
                            return None, "Processing stopped by user", DEVICE_STATUS["STOPPED"], detected_device_type
                            
                        try:
                            logger.debug(f"Executing command on {device_info.host}: '{command}'")
                            
                            # Execute command with textfsm parsing where possible
                            output = connection.send_command(command, use_textfsm=True)
                            
                            # Store the output
                            collected_data[category][command] = output
                            
                            logger.debug(f"Command '{command}' executed successfully")
                            
                        except Exception as cmd_error:
                            error_msg = f"Command '{command}' failed: {str(cmd_error)}"
                            logger.warning(f"{error_msg} on {device_info.host}")
                            collected_data[category][command] = f"Error: {error_msg}"
                
                processing_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Data collection completed for {device_info.host} in {processing_time:.2f}s")
                
                return collected_data, None, DEVICE_STATUS["SUCCESS"], detected_device_type
                
        except NetmikoAuthenticationException as e:
            error_msg = f"Authentication failed for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type
            
        except NetmikoTimeoutException as e:
            error_msg = f"Connection timeout for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type
            
        except NetmikoBaseException as e:
            error_msg = f"Netmiko error for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type
            
        except Exception as e:
            error_msg = f"Unexpected error for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type


class DataProcessor:
    """Data processor with filtering and comparison capabilities."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_results(self, results, session_id: str = None):
        """Save processing results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"collected_data_{session_id}_{timestamp}.json" if session_id else f"collected_data_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        try:
            # Convert dataclass objects to dictionaries
            serializable_results = [asdict(result) for result in results]
            
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Results saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def load_results(self, filepath: str):
        """Load results from JSON file."""
        try:
            with open(filepath, "r", encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading results from {filepath}: {e}")
            raise
    
    def list_output_files(self):
        """List all output files in the output directory."""
        try:
            files = []
            for file_path in self.output_dir.glob("*.json"):
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
            
            # Sort by modified time, newest first
            files.sort(key=lambda x: x["modified"], reverse=True)
            return files
            
        except Exception as e:
            logger.error(f"Error listing output files: {e}")
            return []
    
    def delete_output_file(self, filename: str):
        """Delete an output file."""
        try:
            filepath = self.output_dir / filename
            if filepath.exists() and filepath.is_file():
                filepath.unlink()
                logger.info(f"Deleted file: {filename}")
                return True
            else:
                logger.warning(f"File not found: {filename}")
                return False
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {e}")
            return False
    
    def export_to_excel(self, data, filepath: str):
        """Excel export with additional columns."""
        try:
            # Flatten data for Excel export
            flat_data = []
            for item in data:
                flat_item = {
                    "IP_MGMT": item.get("ip_mgmt", "N/A"),
                    "Nama_SW": item.get("nama_sw", "N/A"),
                    "SN": item.get("sn", "N/A"),
                    "Model_SW": item.get("model_sw", "N/A"),
                    "Status": item.get("status", "N/A"),
                    "Connection_Status": item.get("connection_status", "N/A"),
                    "Detected_Device_Type": item.get("detected_device_type", "N/A"),
                    "Processing_Time": item.get("processing_time", "N/A"),
                    "Retry_Count": item.get("retry_count", 0),
                    "Last_Attempt": item.get("last_attempt", "N/A"),
                    "Details": str(item.get("error") or "Data collected successfully")
                }
                flat_data.append(flat_item)
            
            df = pd.DataFrame(flat_data)
            df.to_excel(filepath, index=False, engine=EXCEL_ENGINE)
            
            logger.info(f"Data exported to Excel: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")
            raise
    
    def filter_results(self, data, filter_type: str, filter_value: str):
        """Filter results based on criteria."""
        try:
            if not data:
                return []
            
            filtered_data = []
            
            for item in data:
                if filter_type == "status":
                    if item.get("status", "").lower() == filter_value.lower():
                        filtered_data.append(item)
                elif filter_type == "model_sw":
                    if filter_value.lower() in item.get("model_sw", "").lower():
                        filtered_data.append(item)
                elif filter_type == "connection_status":
                    if item.get("connection_status", "").lower() == filter_value.lower():
                        filtered_data.append(item)
                elif filter_type == "device_type":
                    if filter_value.lower() in item.get("detected_device_type", "").lower():
                        filtered_data.append(item)
                elif filter_type == "all":
                    filtered_data.append(item)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            return data
    
    def compare_configurations(self, file1_path: str, file2_path: str):
        """Configuration comparison with detailed diff analysis."""
        try:
            # Load data from files
            if file1_path.startswith(str(self.output_dir)):
                data1 = self.load_results(file1_path)
            else:
                data1 = self.load_results(os.path.join(self.output_dir, file1_path))
                
            if file2_path.startswith(str(self.output_dir)):
                data2 = self.load_results(file2_path)
            else:
                data2 = self.load_results(os.path.join(self.output_dir, file2_path))
            
            # Create device mapping for comparison
            devices1 = {item["ip_mgmt"]: item for item in data1}
            devices2 = {item["ip_mgmt"]: item for item in data2}
            
            comparison_results = []
            
            # Compare devices that exist in both files
            for ip, device1 in devices1.items():
                if ip in devices2:
                    device2 = devices2[ip]
                    diff_result = self._compare_device_configs(device1, device2)
                    comparison_results.append(diff_result)
                else:
                    # Device only in file1
                    comparison_results.append({
                        "ip_mgmt": ip,
                        "nama_sw": device1.get("nama_sw", "N/A"),
                        "status": "Removed",
                        "changes": ["Device no longer exists in second file"],
                        "details": "Device was removed"
                    })
            
            # Devices only in file2
            for ip, device2 in devices2.items():
                if ip not in devices1:
                    comparison_results.append({
                        "ip_mgmt": ip,
                        "nama_sw": device2.get("nama_sw", "N/A"),
                        "status": "Added",
                        "changes": ["New device in second file"],
                        "details": "Device was added"
                    })
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing configurations: {e}")
            raise
    
    def _compare_device_configs(self, device1, device2):
        """Compare configurations between two devices."""
        try:
            changes = []
            
            # Get configuration data
            config1 = device1.get("data", {})
            config2 = device2.get("data", {})
            
            if not config1 or not config2:
                return {
                    "ip_mgmt": device1.get("ip_mgmt", "N/A"),
                    "nama_sw": device1.get("nama_sw", "N/A"),
                    "status": "No Configuration",
                    "changes": ["No configuration data available for comparison"],
                    "details": "Configuration data missing"
                }
            
            # Compare all command outputs
            for category in set(list(config1.keys()) + list(config2.keys())):
                cat_config1 = config1.get(category, {})
                cat_config2 = config2.get(category, {})
                
                for command in set(list(cat_config1.keys()) + list(cat_config2.keys())):
                    output1 = str(cat_config1.get(command, ""))
                    output2 = str(cat_config2.get(command, ""))
                    
                    if output1 != output2:
                        if command not in cat_config1:
                            changes.append(f"ADDED: Command '{command}' in category '{category}'")
                        elif command not in cat_config2:
                            changes.append(f"REMOVED: Command '{command}' in category '{category}'")
                        else:
                            # Generate detailed diff
                            diff = self._generate_config_diff(output1, output2, command)
                            changes.extend(diff)
                
            # If no changes found
            if not changes:
                return {
                    "ip_mgmt": device1.get("ip_mgmt", "N/A"),
                    "nama_sw": device1.get("nama_sw", "N/A"),
                    "status": "No Changes",
                    "changes": [],
                    "details": "No configuration changes detected"
                }
            
            return {
                "ip_mgmt": device1.get("ip_mgmt", "N/A"),
                "nama_sw": device1.get("nama_sw", "N/A"),
                "status": "Changed",
                "changes": changes,
                "details": f"Found {len(changes)} configuration changes"
            }
            
        except Exception as e:
            logger.error(f"Error comparing device configs: {e}")
            return {
                "ip_mgmt": device1.get("ip_mgmt", "N/A"),
                "nama_sw": device1.get("nama_sw", "N/A"),
                "status": "Error",
                "changes": [f"Error during comparison: {str(e)}"],
                "details": "Comparison failed"
            }
    
    def _generate_config_diff(self, config1: str, config2: str, command: str = ""):
        """Generate detailed configuration diff."""
        try:
            lines1 = config1.splitlines()
            lines2 = config2.splitlines()
            
            changes = []
            
            # Generate unified diff
            diff = difflib.unified_diff(
                lines1, lines2,
                fromfile="Previous",
                tofile="Current",
                lineterm=""
            )
            
            cmd_prefix = f"[{command}] " if command else ""
            
            for line in diff:
                if line.startswith('@@'):
                    continue
                elif line.startswith('---') or line.startswith('+++'):
                    continue
                elif line.startswith('-'):
                    changes.append(f"{cmd_prefix}REMOVED: {line[1:].strip()}")
                elif line.startswith('+'):
                    changes.append(f"{cmd_prefix}ADDED: {line[1:].strip()}")
            
            return changes[:20]  # Limit to first 20 changes per command
            
        except Exception as e:
            logger.error(f"Error generating config diff: {e}")
            return [f"Error generating diff: {str(e)}"]


class ChartGenerator:
    """Chart generator with new visualizations."""
    
    @staticmethod
    def generate_bar_chart(data, filter_by: str = "model_sw"):
        """Generate bar chart data."""
        try:
            if not data:
                raise ValueError("No data provided for chart generation")
            
            df = pd.DataFrame(data)
            
            if filter_by not in df.columns:
                available_columns = list(df.columns)
                raise ValueError(f"Filter '{filter_by}' not found. Available: {available_columns}")
            
            # Count occurrences
            counts = df.groupby(filter_by).size().reset_index(name="count")
            
            # Create bar chart
            fig = px.bar(
                counts,
                x=filter_by,
                y="count",
                title=f"Device Distribution by {filter_by.replace('_', ' ').title()}",
                labels={filter_by: filter_by.replace('_', ' ').title(), "count": "Number of Devices"},
                color="count",
                color_continuous_scale="viridis"
            )
            
            # Update layout
            fig.update_layout(
                showlegend=False,
                xaxis_tickangle=-45,
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            raise
    
    @staticmethod
    def generate_status_pie_chart(data):
        """Generate pie chart for status distribution."""
        try:
            df = pd.DataFrame(data)
            status_counts = df['status'].value_counts()
            
            # Custom colors for different statuses
            colors = {
                'Success': '#28a745',
                'Failed': '#dc3545', 
                'Retrying': '#ffc107',
                'Pending': '#6c757d',
                'Stopped': '#fd7e14'
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                hole=0.3,
                marker_colors=[colors.get(status, '#17a2b8') for status in status_counts.index]
            )])
            
            fig.update_layout(
                title="Device Status Distribution",
                height=400,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error generating pie chart: {e}")
            raise
    
    @staticmethod
    def generate_progress_chart(session: ProcessingSession):
        """Generate progress tracking chart."""
        try:
            categories = ['Completed', 'Failed', 'Pending']
            values = [
                session.successful,
                session.failed,
                session.total_devices - session.completed
            ]
            colors = ['#28a745', '#dc3545', '#6c757d']
            
            fig = go.Figure(data=[go.Bar(
                x=categories,
                y=values,
                marker_color=colors
            )])
            
            fig.update_layout(
                title="Processing Progress",
                yaxis_title="Number of Devices",
                height=300,
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            return json.loads(fig.to_json())
            
        except Exception as e:
            logger.error(f"Error generating progress chart: {e}")
            raise


# Flask Production Server
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')
CORS(app)

# File upload configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for the Flask app
config_manager = ConfigManager()
device_manager = NetworkDeviceManager(config_manager)
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Global processing state
processing_sessions = {}
current_session_id = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_csv_columns(df):
    """Validate CSV columns and provide detailed error messages."""
    errors = []
    warnings = []
    
    # Required columns and their aliases
    required_columns = {
        'IP MGMT': ['ip mgmt', 'ip_mgmt', 'ip', 'management ip', 'mgmt_ip', 'device_ip'],
        'Nama SW': ['nama sw', 'nama_sw', 'name', 'hostname', 'device_name', 'switch_name'],
        'SN': ['sn', 'serial', 'serial_number', 'serial number', 'serial_no'],
        'Model SW': ['model sw', 'model_sw', 'model', 'device_model', 'switch_model']
    }
    
    # Check for empty dataframe
    if df.empty:
        errors.append("CSV file is empty. Please provide a file with device information.")
        return errors, warnings, {}
    
    # Map columns
    column_mapping = {}
    df_cols_lower = {col.lower(): col for col in df.columns}
    
    for req_col, aliases in required_columns.items():
        found = False
        for alias in [req_col.lower()] + aliases:
            if alias in df_cols_lower:
                column_mapping[req_col] = df_cols_lower[alias]
                found = True
                break
        
        if not found:
            errors.append(f"Missing required column '{req_col}'. Acceptable column names: {', '.join([req_col] + [a.upper() for a in aliases])}")
    
    # If we have all required columns, check data quality
    if not errors:
        for idx, row in df.iterrows():
            row_num = idx + 2  # Account for header row
            
            # Check IP address
            ip_val = str(row[column_mapping['IP MGMT']]).strip()
            if not ip_val or ip_val.lower() in ['nan', 'none', 'null', '']:
                errors.append(f"Row {row_num}: Missing IP address")
            elif not validate_ip_address(ip_val):
                errors.append(f"Row {row_num}: Invalid IP address format '{ip_val}'")
            
            # Check device name
            name_val = str(row[column_mapping['Nama SW']]).strip()
            if not name_val or name_val.lower() in ['nan', 'none', 'null', '']:
                warnings.append(f"Row {row_num}: Missing device name")
            
            # Check model
            model_val = str(row[column_mapping['Model SW']]).strip()
            if not model_val or model_val.lower() in ['nan', 'none', 'null', '']:
                warnings.append(f"Row {row_num}: Missing device model - will use SSH autodetect")
            
            # Check serial number
            sn_val = str(row[column_mapping['SN']]).strip()
            if not sn_val or sn_val.lower() in ['nan', 'none', 'null', '']:
                warnings.append(f"Row {row_num}: Missing serial number")
    
    return errors, warnings, column_mapping

def validate_ip_address(ip_str):
    """Validate IP address format."""
    parts = ip_str.split('.')
    if len(parts) != 4:
        return False
    
    for part in parts:
        try:
            num = int(part)
            if num < 0 or num > 255:
                return False
        except ValueError:
            return False
    
    return True

def get_csv_separator(file_path: str):
    """Detects the separator of a CSV file by checking the header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            # Count occurrences of common separators
            separators = {',': header.count(','), ';': header.count(';'), '\t': header.count('\t'), '|': header.count('|')}
            # Return the separator with most occurrences
            separator = max(separators.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected CSV separator: '{separator}'")
            return separator
    except Exception as e:
        logger.warning(f"Could not detect separator for {file_path}, defaulting to comma. Error: {e}")
        return ','

def threaded_process_devices(username: str, password: str, file_path: str, session_id: str, retry_failed_only: bool = False):
    """Threaded processing with retry mechanism and progress tracking."""
    try:
        session = processing_sessions[session_id]
        session.start_time = datetime.now()
        
        logger.info(f"Starting threaded processing for session {session_id}")
        
        if retry_failed_only:
            # Load previous results and retry only failed devices
            logger.info("Retrying failed devices only")
            # This would be implemented to load previous session data
            # For now, we'll process the entire file
        
        # Detect separator before reading with pandas
        separator = get_csv_separator(file_path)
        df = pd.read_csv(file_path, sep=separator)
        
        # Strip whitespace from column names
        df.columns = [col.strip() for col in df.columns]

        # Validate CSV columns and data
        errors, warnings, column_mapping = validate_csv_columns(df)
        
        if errors:
            error_message = "CSV validation failed:\n" + "\n".join(errors)
            if warnings:
                error_message += "\n\nWarnings:\n" + "\n".join(warnings[:5])  # Show first 5 warnings
                if len(warnings) > 5:
                    error_message += f"\n... and {len(warnings) - 5} more warnings"
            
            raise ValueError(error_message)
        
        # Log warnings
        if warnings:
            logger.warning(f"CSV validation warnings: {len(warnings)} issues found")
            for warning in warnings[:10]:  # Log first 10 warnings
                logger.warning(warning)

        # Update session with total devices
        session.total_devices = len(df)
        logger.info(f"Processing {len(df)} devices from CSV file")

        results = []
        
        # Process devices with retry mechanism
        for idx, row in df.iterrows():
            if session.is_stopped:
                logger.info("Processing stopped by user")
                break
                
            device_info = DeviceInfo(
                host=str(row[column_mapping['IP MGMT']]).strip(),
                username=username,
                password=password,
                conn_timeout=DEFAULT_TIMEOUT
            )
            
            metadata = DeviceMetadata(
                ip_mgmt=str(row[column_mapping['IP MGMT']]).strip(),
                nama_sw=str(row[column_mapping['Nama SW']]).strip(),
                sn=str(row[column_mapping['SN']]).strip(),
                model_sw=str(row[column_mapping['Model SW']]).strip()
            )
            
            # Process single device with retry logic
            result = process_single_device_with_retry(device_info, metadata, session)
            results.append(result)
            
            # Update session progress
            session.completed += 1
            if result.status == "Success":
                session.successful += 1
            else:
                session.failed += 1
            
            logger.info(f"Progress: {session.completed}/{session.total_devices} - {metadata.ip_mgmt}: {result.status}")
        
        # Save results
        session.end_time = datetime.now()
        output_filename = data_processor.save_results(results, session_id)
        session.output_file = output_filename
        
        # Prepare final response
        response = {
            "status": "success" if not session.is_stopped else "stopped",
            "message": f"Processing complete: {session.successful} successful, {session.failed} failed. Results saved to {Path(output_filename).name}",
            "data": [asdict(result) for result in results],
            "session_id": session_id,
            "output_file": Path(output_filename).name,
            "summary": {
                "total": session.total_devices,
                "successful": session.successful,
                "failed": session.failed,
                "completed": session.completed,
                "duration": (session.end_time - session.start_time).total_seconds() if session.end_time else 0
            }
        }
        
        # Store final results in session
        processing_sessions[session_id] = session
        processing_sessions[f"{session_id}_results"] = response
        
        logger.info(f"Processing completed for session {session_id}: {len(results)} devices processed")
        
    except Exception as e:
        logger.error(f"Error in threaded processing: {e}", exc_info=True)
        session.end_time = datetime.now()
        response = {
            "status": "error",
            "message": f"Processing failed: {str(e)}",
            "session_id": session_id
        }
        processing_sessions[f"{session_id}_results"] = response

def process_single_device_with_retry(device_info: DeviceInfo, metadata: DeviceMetadata, session: ProcessingSession):
    """Process a single device with retry mechanism."""
    retry_count = 0
    max_retries = DEFAULT_RETRY_ATTEMPTS
    
    while retry_count < max_retries:
        if session.is_stopped:
            return ProcessingResult(
                ip_mgmt=metadata.ip_mgmt,
                nama_sw=metadata.nama_sw,
                sn=metadata.sn,
                model_sw=metadata.model_sw,
                status="Stopped",
                connection_status=DEVICE_STATUS["STOPPED"],
                retry_count=retry_count,
                last_attempt=datetime.now().isoformat(),
                error="Processing stopped by user"
            )
        
        start_time = datetime.now()
        
        try:
            # Update connection status
            connection_status = DEVICE_STATUS["CONNECTING"] if retry_count == 0 else DEVICE_STATUS["RETRYING"]
            
            device_data, error, final_status, detected_type = device_manager.connect_and_collect_data(
                device_info, metadata.model_sw, retry_count, session
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if error is None:
                # Success
                return ProcessingResult(
                    ip_mgmt=metadata.ip_mgmt,
                    nama_sw=metadata.nama_sw,
                    sn=metadata.sn,
                    model_sw=metadata.model_sw,
                    status="Success",
                    data=device_data,
                    processing_time=processing_time,
                    retry_count=retry_count,
                    last_attempt=datetime.now().isoformat(),
                    connection_status=DEVICE_STATUS["SUCCESS"],
                    detected_device_type=detected_type
                )
            else:
                # Failed, check if we should retry
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying {device_info.host} (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    # Max retries reached
                    return ProcessingResult(
                        ip_mgmt=metadata.ip_mgmt,
                        nama_sw=metadata.nama_sw,
                        sn=metadata.sn,
                        model_sw=metadata.model_sw,
                        status="Failed",
                        error=error,
                        processing_time=processing_time,
                        retry_count=retry_count,
                        last_attempt=datetime.now().isoformat(),
                        connection_status=DEVICE_STATUS["FAILED"],
                        detected_device_type=detected_type
                    )
                    
        except Exception as e:
            retry_count += 1
            if retry_count >= max_retries:
                processing_time = (datetime.now() - start_time).total_seconds()
                return ProcessingResult(
                    ip_mgmt=metadata.ip_mgmt,
                    nama_sw=metadata.nama_sw,
                    sn=metadata.sn,
                    model_sw=metadata.model_sw,
                    status="Failed",
                    error=f"Unexpected error: {str(e)}",
                    processing_time=processing_time,
                    retry_count=retry_count,
                    last_attempt=datetime.now().isoformat(),
                    connection_status=DEVICE_STATUS["FAILED"]
                )
            else:
                logger.info(f"Retrying {device_info.host} due to exception (attempt {retry_count + 1}/{max_retries})")
                time.sleep(2)
                continue


# Flask API Routes

@app.route('/')
def serve_react_app():
    """Serve the React application."""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_react_routes(path):
    """Serve React routes - catch all for React Router."""
    if path.startswith('api/'):
        return jsonify({"error": "API endpoint not found"}), 404
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/system_info', methods=['GET'])
def get_system_info():
    """Get system information."""
    try:
        supported_devices = config_manager.get_supported_device_types()
        
        return jsonify({
            "status": "success",
            "data": {
                "supported_devices": supported_devices,
                "max_workers": MAX_WORKERS,
                "default_timeout": DEFAULT_TIMEOUT,
                "default_retry_attempts": DEFAULT_RETRY_ATTEMPTS,
                "output_directory": str(data_processor.output_dir),
                "version": "2.1.0",
                "features": [
                    "SSH Device Connection",
                    "Auto Device Type Detection",
                    "Retry Mechanism",
                    "Progress Tracking", 
                    "Batch Processing",
                    "Data Filtering",
                    "File Comparison",
                    "Stop Functionality",
                    "Web File Upload",
                    "Excel Export",
                    "Output File Management",
                    "Real-time Logs"
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get system info: {str(e)}"
        }), 500

@app.route('/api/upload_csv', methods=['POST'])
def upload_csv():
    """Handle CSV file upload for web interface."""
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Validate CSV
            try:
                separator = get_csv_separator(filepath)
                df = pd.read_csv(filepath, sep=separator)
                df.columns = [col.strip() for col in df.columns]
                
                # Validate columns and data
                errors, warnings, column_mapping = validate_csv_columns(df)
                
                if errors:
                    os.remove(filepath)  # Clean up invalid file
                    error_message = "CSV validation failed:\n" + "\n".join(errors)
                    if warnings:
                        error_message += f"\n\nWarnings: {len(warnings)} issues found"
                    return jsonify({
                        "status": "error",
                        "message": error_message,
                        "errors": errors,
                        "warnings": warnings[:10]  # Return first 10 warnings
                    }), 400
                
                # Count total devices
                device_count = len(df)
                
                return jsonify({
                    "status": "success",
                    "message": f"File uploaded successfully. {device_count} devices found.",
                    "filepath": filepath,
                    "device_count": device_count,
                    "warnings": warnings[:10] if warnings else []
                })
                
            except Exception as e:
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    "status": "error",
                    "message": f"Error reading CSV: {str(e)}"
                }), 400
        
        return jsonify({"status": "error", "message": "Invalid file type. Only CSV files are allowed."}), 400
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/process_devices', methods=['POST'])
def process_devices_from_file():
    """Start device processing with uploaded file."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        filepath = data.get('filepath', '')
        retry_failed_only = data.get('retry_failed_only', False)
        
        # Validate credentials are provided
        if not username or not password:
            return jsonify({
                "status": "error",
                "message": "Username and password are required for SSH/TACACS authentication"
            }), 400
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({
                "status": "error",
                "message": "No valid file found. Please upload a CSV file first."
            }), 400
        
        # Validate CSV file again
        try:
            separator = get_csv_separator(filepath)
            full_df = pd.read_csv(filepath, sep=separator)
            device_count = len(full_df)
            
            if device_count == 0:
                return jsonify({"status": "error", "message": "The CSV file is empty."}), 400
                
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return jsonify({
                "status": "error",
                "message": f"Error reading CSV file: {str(e)}"
            }), 400
        
        # Create new processing session
        session_id = str(uuid.uuid4())
        session = ProcessingSession(
            session_id=session_id,
            total_devices=device_count
        )
        
        processing_sessions[session_id] = session
        global current_session_id
        current_session_id = session_id
        
        # Start processing in thread
        thread = threading.Thread(
            target=threaded_process_devices,
            args=(username, password, filepath, session_id, retry_failed_only)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "loading",
            "message": f"Processing started for {device_count} devices",
            "session_id": session_id,
            "total_devices": device_count
        })
        
    except Exception as e:
        logger.error(f"Error starting process: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error starting process: {str(e)}"
        }), 500

@app.route('/api/processing_status/<session_id>', methods=['GET'])
def get_processing_status(session_id):
    """Get real-time processing status with progress."""
    try:
        # Check if results are ready
        if f"{session_id}_results" in processing_sessions:
            results = processing_sessions[f"{session_id}_results"]
            # Clean up completed session
            if session_id in processing_sessions:
                del processing_sessions[session_id]
            del processing_sessions[f"{session_id}_results"]
            return jsonify(results)
        
        # Check if session exists and is active
        if session_id in processing_sessions:
            session = processing_sessions[session_id]
            progress_percentage = (session.completed / session.total_devices * 100) if session.total_devices > 0 else 0
            
            return jsonify({
                "status": "processing",
                "message": f"Processing... {session.completed}/{session.total_devices} devices completed",
                "progress": {
                    "total": session.total_devices,
                    "completed": session.completed,
                    "successful": session.successful,
                    "failed": session.failed,
                    "percentage": round(progress_percentage, 1)
                },
                "session_id": session_id
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Session not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error getting status: {str(e)}"
        }), 500

@app.route('/api/stop_processing/<session_id>', methods=['POST'])
def stop_processing(session_id):
    """Stop processing for a specific session."""
    try:
        if session_id in processing_sessions:
            session = processing_sessions[session_id]
            session.is_stopped = True
            logger.info(f"Processing stopped for session {session_id}")
            
            return jsonify({
                "status": "success",
                "message": "Processing stop requested"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Session not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Error stopping processing: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error stopping processing: {str(e)}"
        }), 500

@app.route('/api/output_files', methods=['GET'])
def list_output_files():
    """List all output files."""
    try:
        files = data_processor.list_output_files()
        return jsonify({
            "status": "success",
            "data": files,
            "total": len(files)
        })
    except Exception as e:
        logger.error(f"Error listing output files: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error listing files: {str(e)}"
        }), 500

@app.route('/api/output_files/<filename>', methods=['GET'])
def download_output_file(filename):
    """Download a specific output file."""
    try:
        filepath = os.path.join(data_processor.output_dir, filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype='application/json'
            )
        else:
            return jsonify({
                "status": "error",
                "message": "File not found"
            }), 404
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error downloading file: {str(e)}"
        }), 500

@app.route('/api/output_files/<filename>', methods=['DELETE'])
def delete_output_file(filename):
    """Delete a specific output file."""
    try:
        if data_processor.delete_output_file(filename):
            return jsonify({
                "status": "success",
                "message": f"File {filename} deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "File not found or could not be deleted"
            }), 404
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error deleting file: {str(e)}"
        }), 500

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent logs from memory."""
    try:
        logs = in_memory_handler.get_logs()
        return jsonify({
            "status": "success",
            "data": logs,
            "total": len(logs)
        })
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error getting logs: {str(e)}"
        }), 500

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """Clear logs from memory."""
    try:
        in_memory_handler.clear_logs()
        return jsonify({
            "status": "success",
            "message": "Logs cleared successfully"
        })
    except Exception as e:
        logger.error(f"Error clearing logs: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error clearing logs: {str(e)}"
        }), 500

@app.route('/api/compare_files', methods=['POST'])
def compare_files():
    """Compare two configuration files."""
    try:
        data = request.get_json()
        file1 = data.get('file1')
        file2 = data.get('file2')
        
        if not file1 or not file2:
            return jsonify({
                "status": "error",
                "message": "Two files must be selected for comparison"
            }), 400
        
        comparison_results = data_processor.compare_configurations(file1, file2)
        
        return jsonify({
            "status": "success",
            "data": comparison_results,
            "file1": os.path.basename(file1),
            "file2": os.path.basename(file2),
            "total_devices_compared": len(comparison_results)
        })
        
    except Exception as e:
        logger.error(f"Error comparing files: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error comparing files: {str(e)}"
        }), 500

@app.route('/api/retry_failed', methods=['POST'])
def retry_failed_devices():
    """Retry only failed devices from previous session."""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        previous_results = data.get('results', [])
        
        # Validate credentials
        if not username or not password:
            return jsonify({
                "status": "error",
                "message": "Username and password are required for SSH/TACACS authentication"
            }), 400
        
        if not previous_results:
            return jsonify({
                "status": "error",
                "message": "No previous results provided for retry"
            }), 400
        
        # Filter failed devices
        failed_devices = [device for device in previous_results if device.get('status') != 'Success']
        
        if not failed_devices:
            return jsonify({
                "status": "info",
                "message": "No failed devices to retry"
            })
        
        # Create temporary CSV for failed devices
        failed_df = pd.DataFrame(failed_devices)
        temp_csv_path = data_processor.output_dir / f"retry_devices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        failed_df.to_csv(temp_csv_path, index=False)
        
        # Create new session for retry
        session_id = str(uuid.uuid4())
        session = ProcessingSession(
            session_id=session_id,
            total_devices=len(failed_devices)
        )
        
        processing_sessions[session_id] = session
        global current_session_id
        current_session_id = session_id
        
        # Start retry processing
        thread = threading.Thread(
            target=threaded_process_devices,
            args=(username, password, str(temp_csv_path), session_id, True)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "loading",
            "message": f"Retrying {len(failed_devices)} failed devices",
            "session_id": session_id,
            "total_devices": len(failed_devices)
        })
        
    except Exception as e:
        logger.error(f"Error retrying failed devices: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error retrying devices: {str(e)}"
        }), 500

@app.route('/api/filter_results', methods=['POST'])
def filter_results():
    """Filter results based on criteria."""
    try:
        data = request.get_json()
        results = data.get('results', [])
        filter_type = data.get('filter_type', 'all')
        filter_value = data.get('filter_value', '')
        
        filtered_results = data_processor.filter_results(results, filter_type, filter_value)
        
        return jsonify({
            "status": "success",
            "data": filtered_results,
            "total_filtered": len(filtered_results),
            "total_original": len(results)
        })
        
    except Exception as e:
        logger.error(f"Error filtering results: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error filtering results: {str(e)}"
        }), 500

@app.route('/api/export_excel', methods=['POST'])
def export_to_excel():
    """Excel export - Web version downloads directly."""
    try:
        data = request.get_json().get('data', [])
        
        if not data:
            return jsonify({"status": "error", "message": "No data to export"}), 400
        
        # Create temporary Excel file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"network_data_export_{timestamp}.xlsx"
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Export to Excel
        data_processor.export_to_excel(data, temp_path)
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to export to Excel: {str(e)}"
        }), 500

@app.route('/api/generate_chart', methods=['POST'])
def generate_chart_data():
    """Generate chart data."""
    try:
        request_data = request.get_json()
        data = request_data.get('data', [])
        filter_by = request_data.get('filter_by', 'model_sw')
        
        if not data:
            return jsonify({"status": "error", "message": "No data to generate chart"}), 400
        
        # Generate appropriate chart based on filter
        if filter_by == "status":
            chart_data = chart_generator.generate_status_pie_chart(data)
        else:
            chart_data = chart_generator.generate_bar_chart(data, filter_by)
        
        return jsonify({
            "status": "success",
            "data": chart_data
        })
        
    except Exception as e:
        logger.error(f"Error generating chart: {e}")
        return jsonify({
            "status": "error",
            "message": f"Chart generation failed: {str(e)}"
        }), 500

@app.route('/api/progress_chart/<session_id>', methods=['GET'])
def get_progress_chart(session_id):
    """Get real-time progress chart."""
    try:
        if session_id in processing_sessions:
            session = processing_sessions[session_id]
            chart_data = chart_generator.generate_progress_chart(session)
            
            return jsonify({
                "status": "success",
                "data": chart_data
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Session not found"
            }), 404
            
    except Exception as e:
        logger.error(f"Error generating progress chart: {e}")
        return jsonify({
            "status": "error",
            "message": f"Progress chart generation failed: {str(e)}"
        }), 500

# Health check endpoint for Railway
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Railway."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.1.0"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask production server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)