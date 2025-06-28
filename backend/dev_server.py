# backend/dev_server.py
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
from netmiko.exceptions import NetmikoBaseException, NetmikoAuthenticationException, NetmikoTimeoutException
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict

# Constants
DEFAULT_TIMEOUT = 20
MAX_WORKERS = 10
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
SUPPORTED_FILE_TYPES = ("CSV Files (*.csv)",)
SUPPORTED_EXPORT_TYPES = ("JSON Files (*.json)",)
EXCEL_ENGINE = "openpyxl"

# Configure logging
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEFAULT_OUTPUT_DIR, 'network_fetcher.log')),
        logging.StreamHandler()
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
                "system": [
                    "show version",
                    "show running-config | section hostname",
                    "show inventory"
                ],
                "interfaces": [
                    "show ip interface brief",
                    "show interface status"
                ],
                "configuration": [
                    "show running-config"
                ]
            },
            "cisco_xe": {
                "system": [
                    "show version",
                    "show running-config | section hostname",
                    "show inventory"
                ],
                "interfaces": [
                    "show ip interface brief", 
                    "show interface status"
                ],
                "configuration": [
                    "show running-config"
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
    
    def connect_and_collect_data(self, device_info: DeviceInfo, retry_count: int = 0, session: ProcessingSession = None):
        """Connect to device and collect data with error handling."""
        collected_data = {}
        start_time = datetime.now()
        
        try:
            logger.info(f"Connecting to device: {device_info.host} (attempt {retry_count + 1})")
            
            # Create connection parameters
            connection_params = asdict(device_info)
            
            with ConnectHandler(**connection_params) as connection:
                detected_device_type = connection.device_type
                logger.info(f"Device {device_info.host} detected as {detected_device_type}")
                
                commands_by_category = self.config_manager.get_commands_for_device(detected_device_type)
                
                if not commands_by_category:
                    error_msg = f"Device type '{detected_device_type}' not supported in configuration"
                    logger.warning(error_msg)
                    return None, error_msg, DEVICE_STATUS["FAILED"]
                
                # Execute commands by category
                for category, commands in commands_by_category.items():
                    logger.debug(f"Processing category '{category}' with {len(commands)} commands")
                    collected_data[category] = {}
                    
                    for command in commands:
                        # Check if session is stopped
                        if session and session.is_stopped:
                            return None, "Processing stopped by user", DEVICE_STATUS["STOPPED"]
                            
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
                
                return collected_data, None, DEVICE_STATUS["SUCCESS"]
                
        except NetmikoAuthenticationException as e:
            error_msg = f"Authentication failed for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"]
            
        except NetmikoTimeoutException as e:
            error_msg = f"Connection timeout for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"]
            
        except NetmikoBaseException as e:
            error_msg = f"Netmiko error for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"]
            
        except Exception as e:
            error_msg = f"Unexpected error for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"]


class DataProcessor:
    """Data processor with filtering and comparison capabilities."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_results(self, results, filename: str = None):
        """Save processing results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"collected_data_{timestamp}.json"
        
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
                elif filter_type == "all":
                    filtered_data.append(item)
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error filtering results: {e}")
            return data
    
    def compare_configurations(self, file1_path: str, file2_path: str):
        """Configuration comparison with detailed diff analysis."""
        try:
            data1 = self.load_results(file1_path)
            data2 = self.load_results(file2_path)
            
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
            config1 = device1.get("data", {}).get("configuration", {})
            config2 = device2.get("data", {}).get("configuration", {})
            
            if not config1 or not config2:
                return {
                    "ip_mgmt": device1.get("ip_mgmt", "N/A"),
                    "nama_sw": device1.get("nama_sw", "N/A"),
                    "status": "No Configuration",
                    "changes": ["No configuration data available for comparison"],
                    "details": "Configuration data missing"
                }
            
            # Compare running configurations
            if "show running-config" in config1 and "show running-config" in config2:
                config_text1 = str(config1["show running-config"])
                config_text2 = str(config2["show running-config"])
                
                if config_text1 != config_text2:
                    # Generate detailed diff
                    diff = self._generate_config_diff(config_text1, config_text2)
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
    
    def _generate_config_diff(self, config1: str, config2: str):
        """Generate detailed configuration diff."""
        try:
            lines1 = config1.splitlines()
            lines2 = config2.splitlines()
            
            changes = []
            
            # Generate unified diff
            diff = difflib.unified_diff(
                lines1, lines2,
                fromfile="Previous Config",
                tofile="Current Config",
                lineterm=""
            )
            
            current_section = None
            for line in diff:
                if line.startswith('@@'):
                    continue
                elif line.startswith('---') or line.startswith('+++'):
                    continue
                elif line.startswith('-'):
                    changes.append(f"REMOVED: {line[1:].strip()}")
                elif line.startswith('+'):
                    changes.append(f"ADDED: {line[1:].strip()}")
                elif line.startswith(' '):
                    # Context line, can be used to track sections
                    if line.strip().startswith('interface') or line.strip().startswith('router'):
                        current_section = line.strip()
            
            return changes[:50]  # Limit to first 50 changes for readability
            
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


# Flask Development Server
app = Flask(__name__)
CORS(app)  # Enable CORS for development

# Global variables for the Flask app
config_manager = ConfigManager()
device_manager = NetworkDeviceManager(config_manager)
data_processor = DataProcessor()
chart_generator = ChartGenerator()

# Global processing state
processing_sessions = {}
current_session_id = None

def show_file_dialog(dialog_type="open", file_types=None, allow_multiple=False):
    """Show file dialog using tkinter."""
    try:
        # Create a root window and hide it
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        if dialog_type == "open":
            if allow_multiple:
                result = filedialog.askopenfilenames(
                    title="Select Files",
                    filetypes=[("CSV Files", "*.csv"), ("JSON Files", "*.json"), ("All Files", "*.*")]
                )
                return list(result) if result else None
            else:
                result = filedialog.askopenfilename(
                    title="Select File",
                    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
                )
                return [result] if result else None
        elif dialog_type == "save":
            result = filedialog.asksaveasfilename(
                title="Save File",
                defaultextension=".xlsx",
                filetypes=[("Excel Files", "*.xlsx"), ("All Files", "*.*")]
            )
            return result
        
        root.destroy()
        return None
        
    except Exception as e:
        logger.error(f"Error showing file dialog: {e}")
        return None

def get_csv_separator(file_path: str):
    """Detects the separator of a CSV file by checking the header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header = f.readline()
            if ';' in header:
                logger.info("Semicolon separator detected in CSV.")
                return ';'
            logger.info("Comma separator assumed for CSV.")
            return ','
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

        # Check if dataframe is empty
        if df.empty:
            raise ValueError("No valid device entries found in CSV file.")

        # Map CSV columns to expected names (case-insensitive)
        column_mapping = {}
        required_aliases = {
            'IP MGMT': ['ip mgmt', 'ip_mgmt', 'ip', 'management ip'],
            'Nama SW': ['nama sw', 'nama_sw', 'name', 'hostname', 'device_name'],
            'SN': ['sn', 'serial', 'serial_number', 'serial number'],
            'Model SW': ['model sw', 'model_sw', 'model', 'device_model']
        }
        
        df_cols_lower = {col.lower(): col for col in df.columns}

        for req_col, aliases in required_aliases.items():
            for alias in [req_col.lower()] + aliases:
                if alias in df_cols_lower:
                    column_mapping[req_col] = df_cols_lower[alias]
                    break
        
        missing_cols = [col for col in required_aliases if col not in column_mapping]
        if missing_cols:
            available_cols = list(df.columns)
            raise ValueError(f"CSV must contain columns similar to: {list(required_aliases.keys())}. Missing mappings for: {missing_cols}. Available columns in file: {available_cols}")

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
        output_filename = data_processor.save_results(results)
        
        # Prepare final response
        response = {
            "status": "success" if not session.is_stopped else "stopped",
            "message": f"Processing complete: {session.successful} successful, {session.failed} failed. Results saved to {Path(output_filename).name}",
            "data": [asdict(result) for result in results],
            "session_id": session_id,
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
            
            device_data, error, final_status = device_manager.connect_and_collect_data(
                device_info, retry_count, session
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
                    connection_status=DEVICE_STATUS["SUCCESS"]
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
                        connection_status=DEVICE_STATUS["FAILED"]
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
                "version": "2.0.0",
                "features": [
                    "Retry Mechanism",
                    "Progress Tracking", 
                    "Batch Processing",
                    "Data Filtering",
                    "File Comparison",
                    "Stop Functionality"
                ]
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get system info: {str(e)}"
        }), 500

@app.route('/api/process_devices', methods=['POST'])
def process_devices_from_file():
    """Start device processing."""
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        retry_failed_only = data.get('retry_failed_only', False)
        
        # Show file dialog
        result = show_file_dialog("open", allow_multiple=False)
        
        if not result:
            return jsonify({
                "status": "info",
                "message": "File selection cancelled"
            })
        
        file_path = result[0]
        logger.info(f"File selected: {file_path}")
        
        # Validate CSV file
        try:
            separator = get_csv_separator(file_path)
            df = pd.read_csv(file_path, sep=separator, nrows=1)
            
            df.columns = [col.strip().lower() for col in df.columns]
            required_patterns = ['ip', 'nama', 'sn', 'model']
            
            missing_patterns = []
            for pattern in required_patterns:
                if not any(pattern in col for col in df.columns):
                    missing_patterns.append(pattern)

            if missing_patterns:
                available_cols = list(pd.read_csv(file_path, sep=separator, nrows=0).columns)
                return jsonify({
                    "status": "error",
                    "message": f"CSV file must contain columns for IP, Nama, SN, and Model. Could not find columns for: {missing_patterns}. Available columns: {available_cols}"
                })
            
            full_df = pd.read_csv(file_path, sep=separator)
            device_count = len(full_df)
            if device_count == 0:
                return jsonify({"status": "error", "message": "The selected CSV file is empty."})
                
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return jsonify({
                "status": "error",
                "message": f"Error reading CSV file: {str(e)}"
            })
        
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
            args=(username, password, file_path, session_id, retry_failed_only)
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

@app.route('/api/retry_failed', methods=['POST'])
def retry_failed_devices():
    """Retry only failed devices from previous session."""
    try:
        data = request.get_json()
        username = data.get('username', '')
        password = data.get('password', '')
        previous_results = data.get('results', [])
        
        if not previous_results:
            return jsonify({
                "status": "error",
                "message": "No previous results provided for retry"
            })
        
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

@app.route('/api/compare_files', methods=['POST'])
def compare_files():
    """Compare two configuration files."""
    try:
        # Show file dialog for selecting two files
        result = show_file_dialog("open", allow_multiple=True)
        
        if not result or len(result) != 2:
            return jsonify({
                "status": "error",
                "message": "Please select exactly two JSON files to compare"
            })
        
        file1_path, file2_path = result
        
        # Perform comparison
        comparison_results = data_processor.compare_configurations(file1_path, file2_path)
        
        return jsonify({
            "status": "success",
            "data": comparison_results,
            "file1": Path(file1_path).name,
            "file2": Path(file2_path).name,
            "total_devices_compared": len(comparison_results)
        })
        
    except Exception as e:
        logger.error(f"Error comparing files: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error comparing files: {str(e)}"
        }), 500

@app.route('/api/export_excel', methods=['POST'])
def export_to_excel():
    """Excel export."""
    try:
        data = request.get_json().get('data', [])
        
        if not data:
            return jsonify({"status": "error", "message": "No data to export"}), 400
        
        # Show save dialog
        save_path = show_file_dialog("save")
        
        if not save_path:
            return jsonify({"status": "info", "message": "Export cancelled"})
        
        # Export to Excel
        data_processor.export_to_excel(data, save_path)
        
        return jsonify({
            "status": "success",
            "message": f"Data successfully exported to {Path(save_path).name}"
        })
        
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

if __name__ == '__main__':
    logger.info("Starting Flask development server on http://localhost:5000")
    app.run(host='localhost', port=5000, debug=True, threaded=True)