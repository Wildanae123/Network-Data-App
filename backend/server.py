# backend/server.py

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
import requests
import ssl
import webbrowser
import threading
import urllib3
import tempfile
import io
import base64
import sys
import queue
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from collections import defaultdict
from jsonrpclib import Server

# Disable SSL warnings and configure SSL bypass
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context for bypassing certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Constants
DEFAULT_TIMEOUT = 30
MAX_WORKERS = 10
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LOGS_DIR = DEFAULT_OUTPUT_DIR
SUPPORTED_FILE_TYPES = ("CSV Files (*.csv)",)
SUPPORTED_EXPORT_TYPES = ("JSON Files (*.json)",)
EXCEL_ENGINE = "openpyxl"

# Arista eAPI comparison commands
ARISTA_COMPARISON_COMMANDS = {
    'mac_address_table': {
        'name': 'MAC Address Table',
        'commands': ['show mac address-table'],
        'description': 'Compare MAC address tables between snapshots'
    },
    'ip_arp': {
        'name': 'IP ARP Table',
        'commands': ['show ip arp'],
        'description': 'Compare ARP tables between snapshots'
    },
    'interfaces_status': {
        'name': 'Interface Status',
        'commands': ['show interfaces status'],
        'description': 'Compare interface status between snapshots'
    },
    'mlag_interfaces': {
        'name': 'MLAG Interfaces',
        'commands': ['show mlag interfaces detail'],
        'description': 'Compare MLAG interface details between snapshots'
    }
}

# API endpoint configurations for different vendors
API_ENDPOINTS = {
    'arista_eos': {
        'endpoint': '/command-api',
        'port_http': 80,
        'port_https': 443,
        'default_protocol': 'https',
        'content_type': 'application/json-rpc',
        'api_type': 'json-rpc'
    },
    'cisco_nxos': {
        'endpoint': '/ins',
        'port_http': 80,
        'port_https': 443,
        'default_protocol': 'https',
        'content_type': 'application/json',
        'api_type': 'rest'
    },
    'cisco_xe': {
        'endpoint': '/restconf/data',
        'port_http': 80,
        'port_https': 443,
        'default_protocol': 'https',
        'content_type': 'application/yang-data+json',
        'api_type': 'restconf'
    }
}

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

# Configure logging with real-time streaming
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Global log queue for real-time streaming
log_queue = queue.Queue(maxsize=1000)

class StreamingLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
        self.logs = []
        self.max_logs = 1000
    
    def emit(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': self.format(record),
            'module': record.module
        }
        
        # Add to memory storage
        self.logs.append(log_entry)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Add to streaming queue
        try:
            if not self.log_queue.full():
                self.log_queue.put_nowait(log_entry)
        except queue.Full:
            pass
    
    def get_logs(self):
        return self.logs
    
    def clear_logs(self):
        self.logs = []

# Create streaming log handler
streaming_handler = StreamingLogHandler(log_queue)
streaming_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DEFAULT_OUTPUT_DIR, 'network_fetcher_dev.log')),
        logging.StreamHandler(),
        streaming_handler
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
    protocol: str = "https"
    port: int = None

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
    api_endpoint: Optional[str] = None
    api_status: Optional[str] = None
    api_response_time: Optional[float] = None

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

@dataclass
class ComparisonResult:
    """Data class for comparison results."""
    ip_mgmt: str
    hostname: str
    first_snapshot: Dict
    second_snapshot: Dict
    compare_result: Dict

def create_ssl_bypass_transport():
    """Create a transport that bypasses SSL verification."""
    import jsonrpclib
    from jsonrpclib.jsonrpc import SafeTransport
    
    class SSLBypassTransport(SafeTransport):
        def make_connection(self, host):
            # Create HTTPS connection that ignores SSL certificates
            try:
                # Try Python 3 approach first
                import http.client
                return http.client.HTTPSConnection(
                    host,
                    context=ssl_context,
                    timeout=30
                )
            except ImportError:
                # Fallback for Python 2
                import httplib
                return httplib.HTTPSConnection(
                    host,
                    timeout=30
                )
    
    return SSLBypassTransport()

class APIClientBase:
    """Base class for vendor API clients."""
    
    def __init__(self, host: str, username: str, password: str, timeout: int = DEFAULT_TIMEOUT, protocol: str = "https", port: int = None):
        self.host = host
        self.username = username
        self.password = password
        self.timeout = timeout
        self.protocol = protocol
        self.port = port
        
        # Create session with SSL verification disabled
        self.session = requests.Session()
        self.session.auth = (username, password)
        self.session.verify = False

        # Additional SSL configuration
        self.session.headers.update({
            'User-Agent': 'NetworkDataApp/2.2.0'
        })
        
        # Configure adapter for SSL bypass
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        adapter = HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def _get_base_url(self, endpoint: str) -> str:
        """Get the base URL for API calls."""
        port = self.port or (443 if self.protocol == 'https' else 80)
        return f"{self.protocol}://{self.host}:{port}{endpoint}"
    
    def test_connection(self) -> tuple[bool, str]:
        """Test API connectivity."""
        raise NotImplementedError("Subclasses must implement test_connection")
    
    def execute_commands(self, commands: List[str]) -> tuple[Dict, Optional[str]]:
        """Execute a list of commands via API."""
        raise NotImplementedError("Subclasses must implement execute_commands")

class AristaEAPIClient(APIClientBase):
    """Arista eAPI client using JSON-RPC."""
    
    def __init__(self, host: str, username: str, password: str, timeout: int = DEFAULT_TIMEOUT, protocol: str = "https", port: int = None):
        super().__init__(host, username, password, timeout, protocol, port)
        self.endpoint = API_ENDPOINTS['arista_eos']['endpoint']
        if not self.port:
            self.port = API_ENDPOINTS['arista_eos'][f'port_{protocol}']

            self.server_url = f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}{self.endpoint}"
        self.switch = None
    
    def _get_server_connection(self):
        """Get or create JSON-RPC server connection."""
        if self.switch is None:
            try:
                from jsonrpclib.jsonrpc import ServerProxy

                # Use custom transport that bypasses SSL
                transport = create_ssl_bypass_transport()

                self.switch = ServerProxy(
                    self.server_url,
                    transport=transport,
                    verbose=False
                )

                # Set socket timeout
                socket.setdefaulttimeout(self.timeout)

            except Exception as e:
                logger.error(f"Failed to create JSON-RPC server connection: {e}")
                raise
        return self.switch

    def test_connection(self) -> tuple[bool, str]:
        """Test eAPI connectivity with show version."""
        try:
            switch = self._get_server_connection()
            result = switch.runCmds(version=1, cmds=['show version'], format='json')
            if result and len(result) > 0:
                return True, "eAPI connection successful"
            else:
                return False, "eAPI test failed: No response"
        except Exception as e:
            error_msg = f"eAPI connection failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def execute_commands(self, commands: List[str]) -> tuple[Dict, Optional[str]]:
        """Execute commands via Arista eAPI using jsonrpclib."""
        try:
            switch = self._get_server_connection()
            logger.debug(f"Executing commands via eAPI: {commands}")
            
            # Execute commands using jsonrpclib
            result = switch.runCmds(version=1, cmds=commands, format='json')
            
            if result and isinstance(result, list):
                # Transform result into command-output mapping
                output = {}
                for i, cmd in enumerate(commands):
                    if i < len(result):
                        output[cmd] = result[i]
                    else:
                        output[cmd] = {"error": "No result returned for this command"}
                
                logger.debug(f"Successfully executed {len(commands)} commands via eAPI")
                return output, None
            else:
                error_msg = "Invalid eAPI response format"
                logger.error(error_msg)
                return {}, error_msg
                
        except Exception as e:
            error_msg = f"eAPI execution error: {str(e)}"
            logger.error(error_msg)
            
            # Try to provide more specific error information
            if "Authentication failed" in str(e):
                error_msg = "eAPI Authentication failed: Check username/password"
            elif "Connection refused" in str(e):
                error_msg = "eAPI Connection refused: Check if eAPI is enabled on device"
            elif "timeout" in str(e).lower():
                error_msg = f"eAPI Connection timeout after {self.timeout}s"
            elif "SSL" in str(e) or "certificate" in str(e).lower():
                error_msg = "eAPI SSL/Certificate error: Check device HTTPS configuration"
            
            return {}, error_msg

class ConfigManager:
    """Manages configuration loading and validation for API-based connections."""
    
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
            
            logger.info(f"API configuration loaded successfully from {file_path}")
            logger.debug(f"Loaded device types: {list(self.config_data.keys())}")
            return self.config_data
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            raise
    
    def _create_default_config(self, file_path):
        """Create a default configuration file for API-based commands."""
        default_config = {
            "arista_eos": {
                "mac_address_table": [
                    "show mac address-table"
                ],
                "ip_arp": [
                    "show ip arp"
                ],
                "interfaces_status": [
                    "show interfaces status"
                ],
                "mlag_interfaces": [
                    "show mlag interfaces detail"
                ],
                "system_info": [
                    "show version",
                    "show hostname",
                    "show inventory"
                ]
            }
        }
        
        try:
            with open(file_path, "w", encoding='utf-8') as f:
                yaml.dump(default_config, f, default_flow_style=False, indent=2)
            logger.info(f"Created default API configuration file: {file_path}")
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

class NetworkDeviceAPIManager:
    """Network device manager using vendor APIs instead of SSH."""
    
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
        
        logger.warning(f"Could not detect device type for model '{model_sw}', defaulting to arista_eos")
        return "arista_eos"
    
    def create_api_client(self, device_info: DeviceInfo) -> APIClientBase:
        """Create appropriate API client based on device type."""
        if device_info.device_type == "arista_eos":
            return AristaEAPIClient(
                device_info.host, 
                device_info.username, 
                device_info.password,
                device_info.conn_timeout,
                device_info.protocol,
                device_info.port
            )
        else:
            logger.warning(f"Unsupported device type '{device_info.device_type}', defaulting to Arista eAPI")
            return AristaEAPIClient(
                device_info.host, 
                device_info.username, 
                device_info.password,
                device_info.conn_timeout,
                device_info.protocol,
                device_info.port
            )
    
    def connect_and_collect_data(self, device_info: DeviceInfo, model_sw: str = None, retry_count: int = 0, session: ProcessingSession = None, selected_commands: List[str] = None):
        """Connect to device via API and collect data with error handling."""
        collected_data = {}
        start_time = datetime.now()
        detected_device_type = None
        api_endpoint = None
        api_response_time = None
        
        try:
            if model_sw and device_info.device_type == "autodetect":
                detected_type = self.detect_device_type_by_model(model_sw)
                device_info.device_type = detected_type
            elif device_info.device_type == "autodetect":
                device_info.device_type = "arista_eos"
            
            detected_device_type = device_info.device_type
            
            logger.info(f"Connecting to device: {device_info.host} via {detected_device_type} API (attempt {retry_count + 1})")
            
            api_client = self.create_api_client(device_info)
            api_endpoint = f"{device_info.protocol}://{device_info.host}:{api_client.port}{api_client.endpoint}"
            
            api_test_start = time.time()
            connection_ok, connection_msg = api_client.test_connection()
            api_response_time = time.time() - api_test_start
            
            if not connection_ok:
                error_msg = f"API connection test failed: {connection_msg}"
                logger.error(error_msg)
                return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type, api_endpoint, api_response_time
            
            logger.info(f"Successfully connected to {device_info.host} via {detected_device_type} API")
            
            commands_by_category = self.config_manager.get_commands_for_device(detected_device_type)
            
            if not commands_by_category:
                error_msg = f"Device type '{detected_device_type}' not supported in API configuration"
                logger.warning(error_msg)
                return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type, api_endpoint, api_response_time
            
            # Filter commands if specific commands are selected
            if selected_commands:
                filtered_commands = {}
                for category in selected_commands:
                    if category in commands_by_category:
                        filtered_commands[category] = commands_by_category[category]
                commands_by_category = filtered_commands
            
            for category, commands in commands_by_category.items():
                logger.debug(f"Processing category '{category}' with {len(commands)} commands via API")
                
                if session and session.is_stopped:
                    return None, "Processing stopped by user", DEVICE_STATUS["STOPPED"], detected_device_type, api_endpoint, api_response_time
                
                try:
                    category_results, category_error = api_client.execute_commands(commands)
                    
                    if category_error:
                        logger.warning(f"API error in category '{category}': {category_error}")
                        collected_data[category] = {"error": category_error}
                    else:
                        collected_data[category] = category_results
                        logger.debug(f"Category '{category}' executed successfully via API")
                        
                except Exception as cmd_error:
                    error_msg = f"Category '{category}' failed via API: {str(cmd_error)}"
                    logger.warning(f"{error_msg} on {device_info.host}")
                    collected_data[category] = {"error": error_msg}
            
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"API data collection completed for {device_info.host} in {processing_time:.2f}s")
            
            return collected_data, None, DEVICE_STATUS["SUCCESS"], detected_device_type, api_endpoint, api_response_time
            
        except Exception as e:
            error_msg = f"Unexpected API error for {device_info.host}: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, DEVICE_STATUS["FAILED"], detected_device_type, api_endpoint, api_response_time

class DataProcessor:
    """Data processor with filtering and comparison capabilities."""
    
    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def save_results(self, results, session_id: str = None, selected_commands: List[str] = None):
        """Save processing results to JSON file with command info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
        # Include command info in filename if available
        cmd_suffix = ""
        if selected_commands and len(selected_commands) > 0:
            cmd_suffix = f"_{'_'.join(selected_commands[:3])}"
            if len(selected_commands) > 3:
                cmd_suffix += "_etc"
    
        filename = f"collected_data_{session_id}{cmd_suffix}_{timestamp}.json" if session_id else f"collected_data{cmd_suffix}_{timestamp}.json"
    
        filepath = self.output_dir / filename
    
        try:
            # Add metadata to the results
            output_data = {
                "metadata": {
                    "timestamp": timestamp,
                    "session_id": session_id,
                    "selected_commands": selected_commands,
                    "total_devices": len(results),
                    "successful_devices": len([r for r in results if r.status == "Success"]),
                    "failed_devices": len([r for r in results if r.status == "Failed"])
                },
                "results": [asdict(result) for result in results]
            }
            
            with open(filepath, "w", encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            
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
    
    def export_to_excel_comparison(self, comparison_results: List[ComparisonResult], filepath: str):
        """Export comparison results to Excel with specific columns."""
        try:
            flat_data = []
            for result in comparison_results:
                flat_item = {
                    "IP": result.ip_mgmt,
                    "Hostname": result.hostname,
                    "First_Snapshot": json.dumps(result.first_snapshot, indent=2) if result.first_snapshot else "N/A",
                    "Second_Snapshot": json.dumps(result.second_snapshot, indent=2) if result.second_snapshot else "N/A",
                    "Compare_Result": json.dumps(result.compare_result, indent=2) if result.compare_result else "No differences"
                }
                flat_data.append(flat_item)
            
            df = pd.DataFrame(flat_data)
            df.to_excel(filepath, index=False, engine=EXCEL_ENGINE)
            
            logger.info(f"Comparison data exported to Excel: {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting comparison to Excel: {e}")
            raise
    
    def compare_snapshots(self, first_snapshot_file: str, second_snapshot_file: str, command_category: str):
        """Compare two snapshots for specific command category."""
        try:
            first_data = self.load_results(first_snapshot_file)
            second_data = self.load_results(second_snapshot_file)
            
            # Create device mapping
            first_devices = {item["ip_mgmt"]: item for item in first_data}
            second_devices = {item["ip_mgmt"]: item for item in second_data}
            
            comparison_results = []
            
            # Compare devices that exist in both snapshots
            for ip, first_device in first_devices.items():
                if ip in second_devices:
                    second_device = second_devices[ip]
                    
                    # Extract the specific command data
                    first_cmd_data = first_device.get("data", {}).get(command_category, {})
                    second_cmd_data = second_device.get("data", {}).get(command_category, {})
                    
                    # Compare the data
                    compare_result = self._compare_command_data(first_cmd_data, second_cmd_data, command_category)
                    
                    comparison_result = ComparisonResult(
                        ip_mgmt=ip,
                        hostname=first_device.get("nama_sw", "Unknown"),
                        first_snapshot=first_cmd_data,
                        second_snapshot=second_cmd_data,
                        compare_result=compare_result
                    )
                    comparison_results.append(comparison_result)
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing snapshots: {e}")
            raise
    
    def _compare_command_data(self, first_data: Dict, second_data: Dict, command_category: str) -> Dict:
        """Compare command data between two snapshots."""
        try:
            differences = {
                "status": "no_changes",
                "summary": "",
                "details": []
            }
            
            if command_category == "mac_address_table":
                differences = self._compare_mac_table(first_data, second_data)
            elif command_category == "ip_arp":
                differences = self._compare_arp_table(first_data, second_data)
            elif command_category == "interfaces_status":
                differences = self._compare_interfaces(first_data, second_data)
            elif command_category == "mlag_interfaces":
                differences = self._compare_mlag(first_data, second_data)
            else:
                # Generic comparison
                if first_data != second_data:
                    differences["status"] = "changed"
                    differences["summary"] = "Data has changed between snapshots"
                    differences["details"] = ["Generic data comparison shows differences"]
            
            return differences
            
        except Exception as e:
            logger.error(f"Error comparing command data: {e}")
            return {
                "status": "error",
                "summary": f"Error during comparison: {str(e)}",
                "details": []
            }
    
    def _compare_mac_table(self, first_data: Dict, second_data: Dict) -> Dict:
        """Compare MAC address tables."""
        differences = {
            "status": "no_changes",
            "summary": "",
            "details": []
        }
        
        try:
            # Extract MAC table entries
            first_cmd = first_data.get('show mac address-table', {})
            second_cmd = second_data.get('show mac address-table', {})
            
            if not first_cmd or not second_cmd:
                differences["status"] = "error"
                differences["summary"] = "MAC table data missing"
                return differences
            
            first_entries = first_cmd.get('unicastTable', {}).get('tableEntries', [])
            second_entries = second_cmd.get('unicastTable', {}).get('tableEntries', [])
            
            # Create MAC address sets for comparison
            first_macs = {entry['macAddress']: entry for entry in first_entries}
            second_macs = {entry['macAddress']: entry for entry in second_entries}
            
            added_macs = set(second_macs.keys()) - set(first_macs.keys())
            removed_macs = set(first_macs.keys()) - set(second_macs.keys())
            
            if added_macs or removed_macs:
                differences["status"] = "changed"
                differences["summary"] = f"MAC table changes: {len(added_macs)} added, {len(removed_macs)} removed"
                
                for mac in added_macs:
                    entry = second_macs[mac]
                    differences["details"].append(f"ADDED: MAC {mac} on VLAN {entry.get('vlanId')} interface {entry.get('interface')}")
                
                for mac in removed_macs:
                    entry = first_macs[mac]
                    differences["details"].append(f"REMOVED: MAC {mac} from VLAN {entry.get('vlanId')} interface {entry.get('interface')}")
            
        except Exception as e:
            differences["status"] = "error"
            differences["summary"] = f"Error comparing MAC tables: {str(e)}"
        
        return differences
    
    def _compare_arp_table(self, first_data: Dict, second_data: Dict) -> Dict:
        """Compare ARP tables."""
        differences = {
            "status": "no_changes",
            "summary": "",
            "details": []
        }
        
        try:
            first_cmd = first_data.get('show ip arp', {})
            second_cmd = second_data.get('show ip arp', {})
            
            if not first_cmd or not second_cmd:
                differences["status"] = "error"
                differences["summary"] = "ARP table data missing"
                return differences
            
            first_entries = first_cmd.get('ipV4Neighbors', [])
            second_entries = second_cmd.get('ipV4Neighbors', [])
            
            first_ips = {entry['address']: entry for entry in first_entries}
            second_ips = {entry['address']: entry for entry in second_entries}
            
            added_ips = set(second_ips.keys()) - set(first_ips.keys())
            removed_ips = set(first_ips.keys()) - set(second_ips.keys())
            
            if added_ips or removed_ips:
                differences["status"] = "changed"
                differences["summary"] = f"ARP table changes: {len(added_ips)} added, {len(removed_ips)} removed"
                
                for ip in added_ips:
                    entry = second_ips[ip]
                    differences["details"].append(f"ADDED: IP {ip} with MAC {entry.get('hwAddress')} on {entry.get('interface')}")
                
                for ip in removed_ips:
                    entry = first_ips[ip]
                    differences["details"].append(f"REMOVED: IP {ip} with MAC {entry.get('hwAddress')} from {entry.get('interface')}")
            
        except Exception as e:
            differences["status"] = "error"
            differences["summary"] = f"Error comparing ARP tables: {str(e)}"
        
        return differences
    
    def _compare_interfaces(self, first_data: Dict, second_data: Dict) -> Dict:
        """Compare interface status."""
        differences = {
            "status": "no_changes",
            "summary": "",
            "details": []
        }
        
        try:
            first_cmd = first_data.get('show interfaces status', {})
            second_cmd = second_data.get('show interfaces status', {})
            
            if not first_cmd or not second_cmd:
                differences["status"] = "error"
                differences["summary"] = "Interface status data missing"
                return differences
            
            first_intfs = first_cmd.get('interfaceStatuses', {})
            second_intfs = second_cmd.get('interfaceStatuses', {})
            
            status_changes = []
            
            for intf_name in set(list(first_intfs.keys()) + list(second_intfs.keys())):
                first_status = first_intfs.get(intf_name, {}).get('linkStatus')
                second_status = second_intfs.get(intf_name, {}).get('linkStatus')
                
                if first_status != second_status:
                    status_changes.append(f"Interface {intf_name}: {first_status} -> {second_status}")
            
            if status_changes:
                differences["status"] = "changed"
                differences["summary"] = f"Interface status changes: {len(status_changes)} interfaces"
                differences["details"] = status_changes
            
        except Exception as e:
            differences["status"] = "error"
            differences["summary"] = f"Error comparing interfaces: {str(e)}"
        
        return differences
    
    def _compare_mlag(self, first_data: Dict, second_data: Dict) -> Dict:
        """Compare MLAG interfaces."""
        differences = {
            "status": "no_changes",
            "summary": "",
            "details": []
        }
        
        try:
            first_cmd = first_data.get('show mlag interfaces detail', {})
            second_cmd = second_data.get('show mlag interfaces detail', {})
            
            if not first_cmd or not second_cmd:
                differences["status"] = "error"
                differences["summary"] = "MLAG interface data missing"
                return differences
            
            first_intfs = first_cmd.get('interfaces', {})
            second_intfs = second_cmd.get('interfaces', {})
            
            mlag_changes = []
            
            for mlag_id in set(list(first_intfs.keys()) + list(second_intfs.keys())):
                first_status = first_intfs.get(mlag_id, {}).get('status')
                second_status = second_intfs.get(mlag_id, {}).get('status')
                
                if first_status != second_status:
                    mlag_changes.append(f"MLAG {mlag_id}: {first_status} -> {second_status}")
            
            if mlag_changes:
                differences["status"] = "changed"
                differences["summary"] = f"MLAG changes: {len(mlag_changes)} interfaces"
                differences["details"] = mlag_changes
            
        except Exception as e:
            differences["status"] = "error"
            differences["summary"] = f"Error comparing MLAG: {str(e)}"
        
        return differences

def process_single_device_with_retry(device_info: DeviceInfo, metadata: DeviceMetadata, session: ProcessingSession, device_manager: NetworkDeviceAPIManager, selected_commands: List[str] = None):
    """Process a single device with retry mechanism using APIs."""
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
            connection_status = DEVICE_STATUS["CONNECTING"] if retry_count == 0 else DEVICE_STATUS["RETRYING"]
            
            device_data, error, final_status, detected_type, api_endpoint, api_response_time = device_manager.connect_and_collect_data(
                device_info, metadata.model_sw, retry_count, session, selected_commands
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if error is None:
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
                    detected_device_type=detected_type,
                    api_endpoint=api_endpoint,
                    api_status="Connected",
                    api_response_time=api_response_time
                )
            else:
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying {device_info.host} via API (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(2)
                    continue
                else:
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
                        detected_device_type=detected_type,
                        api_endpoint=api_endpoint,
                        api_status="Failed",
                        api_response_time=api_response_time
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
                    connection_status=DEVICE_STATUS["FAILED"],
                    api_status="Error"
                )
            else:
                logger.info(f"Retrying {device_info.host} due to exception (attempt {retry_count + 1}/{max_retries})")
                time.sleep(2)
                continue

def threaded_process_devices(username: str, password: str, file_path: str, session_id: str, selected_commands: List[str] = None, retry_failed_only: bool = False):
    """Threaded processing with retry mechanism and progress tracking using APIs."""
    try:
        session = processing_sessions[session_id]
        session.start_time = datetime.now()
        
        logger.info(f"Starting API-based threaded processing for session {session_id}")
        
        if retry_failed_only:
            logger.info("Retrying failed devices only")
        
        separator = get_csv_separator(file_path)
        df = pd.read_csv(file_path, sep=separator)
        df.columns = [col.strip() for col in df.columns]

        errors, warnings, column_mapping = validate_csv_columns(df)
        
        if errors:
            error_message = "CSV validation failed:\n" + "\n".join(errors)
            if warnings:
                error_message += "\n\nWarnings:\n" + "\n".join(warnings[:5])
                if len(warnings) > 5:
                    error_message += f"\n... and {len(warnings) - 5} more warnings"
            
            raise ValueError(error_message)

        session.total_devices = len(df)
        logger.info(f"Processing {len(df)} devices from CSV file using vendor APIs")

        results = []
        
        for idx, row in df.iterrows():
            if session.is_stopped:
                logger.info("API processing stopped by user")
                break
                
            device_info = DeviceInfo(
                host=str(row[column_mapping['IP MGMT']]).strip(),
                username=username,
                password=password,
                conn_timeout=DEFAULT_TIMEOUT,
                protocol="https",
                port=None
            )
            
            metadata = DeviceMetadata(
                ip_mgmt=str(row[column_mapping['IP MGMT']]).strip(),
                nama_sw=str(row[column_mapping['Nama SW']]).strip(),
                sn=str(row[column_mapping['SN']]).strip(),
                model_sw=str(row[column_mapping['Model SW']]).strip()
            )
            
            result = process_single_device_with_retry(device_info, metadata, session, device_manager, selected_commands)
            results.append(result)
            
            session.completed += 1
            if result.status == "Success":
                session.successful += 1
            else:
                session.failed += 1
            
            logger.info(f"API Progress: {session.completed}/{session.total_devices} - {metadata.ip_mgmt}: {result.status}")
        
        session.end_time = datetime.now()
        output_filename = data_processor.save_results(results, session_id, selected_commands)
        session.output_file = output_filename
        
        response = {
            "status": "success" if not session.is_stopped else "stopped",
            "message": f"API processing complete: {session.successful} successful, {session.failed} failed. Results saved to {Path(output_filename).name}",
            "data": [asdict(result) for result in results],
            "session_id": session_id,
            "output_file": Path(output_filename).name,
            "summary": {
                "total": session.total_devices,
                "successful": session.successful,
                "failed": session.failed,
                "completed": session.completed,
                "duration": (session.end_time - session.start_time).total_seconds() if session.end_time else 0,
                "connection_method": "Vendor APIs"
            }
        }
        
        processing_sessions[session_id] = session
        processing_sessions[f"{session_id}_results"] = response
        
        logger.info(f"API processing completed for session {session_id}: {len(results)} devices processed")
        
    except Exception as e:
        logger.error(f"Error in API threaded processing: {e}", exc_info=True)
        session.end_time = datetime.now()
        response = {
            "status": "error",
            "message": f"API processing failed: {str(e)}",
            "session_id": session_id
        }
        processing_sessions[f"{session_id}_results"] = response

# Helper Functions
def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv']

def validate_csv_columns(df):
    """Validate CSV columns and provide detailed error messages."""
    errors = []
    warnings = []
    
    required_columns = {
        'IP MGMT': ['ip mgmt', 'ip_mgmt', 'ip', 'management ip', 'mgmt_ip', 'device_ip'],
        'Nama SW': ['nama sw', 'nama_sw', 'name', 'hostname', 'device_name', 'switch_name'],
        'SN': ['sn', 'serial', 'serial_number', 'serial number', 'serial_no'],
        'Model SW': ['model sw', 'model_sw', 'model', 'device_model', 'switch_model']
    }
    
    if df.empty:
        errors.append("CSV file is empty. Please provide a file with device information.")
        return errors, warnings, {}
    
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
    
    if not errors:
        for idx, row in df.iterrows():
            row_num = idx + 2
            
            ip_val = str(row[column_mapping['IP MGMT']]).strip()
            if not ip_val or ip_val.lower() in ['nan', 'none', 'null', '']:
                errors.append(f"Row {row_num}: Missing IP address")
            elif not validate_ip_address(ip_val):
                errors.append(f"Row {row_num}: Invalid IP address format '{ip_val}'")
            
            name_val = str(row[column_mapping['Nama SW']]).strip()
            if not name_val or name_val.lower() in ['nan', 'none', 'null', '']:
                warnings.append(f"Row {row_num}: Missing device name")
            
            model_val = str(row[column_mapping['Model SW']]).strip()
            if not model_val or model_val.lower() in ['nan', 'none', 'null', '']:
                warnings.append(f"Row {row_num}: Missing device model - will use API autodetect")
            
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
            separators = {',': header.count(','), ';': header.count(';'), '\t': header.count('\t'), '|': header.count('|')}
            separator = max(separators.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected CSV separator: '{separator}'")
            return separator
    except Exception as e:
        logger.warning(f"Could not detect separator for {file_path}, defaulting to comma. Error: {e}")
        return ','

# Create Flask app with proper configuration for packaging
if getattr(sys, 'frozen', False):
    # If running as PyInstaller bundle
    template_folder = os.path.join(sys._MEIPASS, '../frontend/dist')
    static_folder = os.path.join(sys._MEIPASS, '../frontend/dist')
    app = Flask(__name__, static_folder=static_folder, template_folder=template_folder)
else:
    # If running in development
    app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')

CORS(app, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# Configure file upload
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
config_manager = ConfigManager()
device_manager = NetworkDeviceAPIManager(config_manager)
data_processor = DataProcessor()
processing_sessions = {}
current_session_id = None

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
    """Get system information with API endpoint details."""
    try:
        supported_devices = config_manager.get_supported_device_types()
        
        api_info = {}
        for device_type in supported_devices:
            if device_type in API_ENDPOINTS:
                endpoint_config = API_ENDPOINTS[device_type]
                api_info[device_type] = {
                    "endpoint": endpoint_config["endpoint"],
                    "protocol": endpoint_config["default_protocol"],
                    "port_http": endpoint_config["port_http"],
                    "port_https": endpoint_config["port_https"],
                    "api_type": endpoint_config["api_type"],
                    "content_type": endpoint_config["content_type"]
                }
        
        return jsonify({
            "status": "success",
            "data": {
                "supported_devices": supported_devices,
                "api_endpoints": api_info,
                "comparison_commands": ARISTA_COMPARISON_COMMANDS,
                "max_workers": MAX_WORKERS,
                "default_timeout": DEFAULT_TIMEOUT,
                "default_retry_attempts": DEFAULT_RETRY_ATTEMPTS,
                "output_directory": str(data_processor.output_dir),
                "version": "2.2.0-DEV",
                "connection_method": "Vendor APIs (HTTP/HTTPS)",                
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get system info: {str(e)}"
        }), 500

@app.route('/api/comparison_commands', methods=['GET'])
def get_comparison_commands():
    """Get available comparison commands."""
    try:
        return jsonify({
            "status": "success",
            "data": ARISTA_COMPARISON_COMMANDS
        })
    except Exception as e:
        logger.error(f"Error getting comparison commands: {e}")
        return jsonify({
            "status": "error",
            "message": f"Failed to get comparison commands: {str(e)}"
        }), 500

@app.route('/api/logs/stream')
def stream_logs():
    """Stream logs using Server-Sent Events."""
    def event_stream():
        while True:
            try:
                # Get log from queue with timeout
                log_entry = log_queue.get(timeout=30)
                yield f"data: {json.dumps(log_entry)}\n\n"
            except queue.Empty:
                # Send heartbeat
                yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': datetime.now().isoformat()})}\n\n"
            except Exception as e:
                logger.error(f"Error in log stream: {e}")
                break
    
    return Response(event_stream(), mimetype="text/plain")

@app.route('/api/logs/clear', methods=['POST'])
def clear_logs():
    """Clear all logs from memory."""
    try:
        streaming_handler.clear_logs()
        logger.info("System logs cleared by user")
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

@app.route('/api/output_files/<filename>', methods=['DELETE'])
def delete_output_file(filename):
    """Delete a specific output file."""
    try:
        filepath = os.path.join(data_processor.output_dir, filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            os.remove(filepath)
            logger.info(f"Output file deleted: {filename}")
            return jsonify({
                "status": "success",
                "message": f"File {filename} deleted successfully"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "File not found"
            }), 404
    except Exception as e:
        logger.error(f"Error deleting file {filename}: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error deleting file: {str(e)}"
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
            
            try:
                separator = get_csv_separator(filepath)
                df = pd.read_csv(filepath, sep=separator)
                df.columns = [col.strip() for col in df.columns]
                
                errors, warnings, column_mapping = validate_csv_columns(df)
                
                if errors:
                    os.remove(filepath)
                    error_message = "CSV validation failed:\n" + "\n".join(errors)
                    if warnings:
                        error_message += f"\n\nWarnings: {len(warnings)} issues found"
                    return jsonify({
                        "status": "error",
                        "message": error_message,
                        "errors": errors,
                        "warnings": warnings[:10]
                    }), 400
                
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
        selected_commands = data.get('selected_commands', [])
        retry_failed_only = data.get('retry_failed_only', False)
        
        if not username or not password:
            return jsonify({
                "status": "error",
                "message": "Username and password are required for API authentication"
            }), 400
        
        if not filepath or not os.path.exists(filepath):
            return jsonify({
                "status": "error",
                "message": "No valid file found. Please upload a CSV file first."
            }), 400
        
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
        
        session_id = str(uuid.uuid4())
        session = ProcessingSession(
            session_id=session_id,
            total_devices=device_count
        )
        
        processing_sessions[session_id] = session
        global current_session_id
        current_session_id = session_id
        
        thread = threading.Thread(
            target=threaded_process_devices,
            args=(username, password, filepath, session_id, selected_commands, retry_failed_only)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "loading",
            "message": f"API processing started for {device_count} devices with selected commands: {selected_commands}",
            "session_id": session_id,
            "total_devices": device_count,
            "selected_commands": selected_commands
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
        if f"{session_id}_results" in processing_sessions:
            results = processing_sessions[f"{session_id}_results"]
            if session_id in processing_sessions:
                del processing_sessions[session_id]
            del processing_sessions[f"{session_id}_results"]
            return jsonify(results)
        
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

@app.route('/api/compare_snapshots', methods=['POST'])
def compare_snapshots():
    """Compare two snapshots for a specific command category."""
    try:
        data = request.get_json()
        first_file = data.get('first_file')
        second_file = data.get('second_file')
        command_category = data.get('command_category')
        
        if not all([first_file, second_file, command_category]):
            return jsonify({
                "status": "error",
                "message": "Missing required parameters: first_file, second_file, command_category"
            }), 400
        
        if command_category not in ARISTA_COMPARISON_COMMANDS:
            return jsonify({
                "status": "error",
                "message": f"Invalid command category. Available: {list(ARISTA_COMPARISON_COMMANDS.keys())}"
            }), 400
        
        # Get full file paths
        first_file_path = os.path.join(data_processor.output_dir, first_file)
        second_file_path = os.path.join(data_processor.output_dir, second_file)
        
        if not os.path.exists(first_file_path) or not os.path.exists(second_file_path):
            return jsonify({
                "status": "error",
                "message": "One or both snapshot files not found"
            }), 404
        
        comparison_results = data_processor.compare_snapshots(
            first_file_path, second_file_path, command_category
        )
        
        # Export to Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_filename = f"comparison_{command_category}_{timestamp}.xlsx"
        excel_path = os.path.join(data_processor.output_dir, excel_filename)
        
        data_processor.export_to_excel_comparison(comparison_results, excel_path)
        
        return jsonify({
            "status": "success",
            "data": [asdict(result) for result in comparison_results],
            "command_category": command_category,
            "first_file": first_file,
            "second_file": second_file,
            "total_compared": len(comparison_results),
            "excel_file": excel_filename
        })
        
    except Exception as e:
        logger.error(f"Error comparing snapshots: {e}")
        return jsonify({
            "status": "error",
            "message": f"Error comparing snapshots: {str(e)}"
        }), 500

@app.route('/api/output_files', methods=['GET'])
def list_output_files():
    """List all output files."""
    try:
        files = []
        for file_path in data_processor.output_dir.glob("*.json"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "filepath": str(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        # Also include Excel files
        for file_path in data_processor.output_dir.glob("*.xlsx"):
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "filepath": str(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        files.sort(key=lambda x: x["modified"], reverse=True)
        
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
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' if filename.endswith('.xlsx') else 'application/json'
            return send_file(
                filepath,
                as_attachment=True,
                download_name=filename,
                mimetype=mimetype
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

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent logs from memory."""
    try:
        logs = streaming_handler.get_logs()
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

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.2.0-DEV",
        "connection_method": "APIs",
        "supported_apis": list(API_ENDPOINTS.keys()),
        "output_directory": str(data_processor.output_dir)
    })

def open_browser():
    """Open browser after a short delay to ensure server is ready."""
    import time
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Start log streaming automatically
    logger.info("Starting Network Data App Production Server")
    logger.info(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    logger.info(f"Upload directory: {UPLOAD_FOLDER}")
    logger.info(f"Supported API endpoints: {list(API_ENDPOINTS.keys())}")
    logger.info(f"Available comparison commands: {list(ARISTA_COMPARISON_COMMANDS.keys())}")
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Production server configuration
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        threaded=True
    )