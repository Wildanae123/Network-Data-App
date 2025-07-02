# backend/test_connection.py
"""
Simple test script to verify Arista eAPI connection using jsonrpclib with SSL bypass
"""

import ssl
import socket
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Create unverified SSL context for bypassing certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

try:
    from jsonrpclib import Server
    from jsonrpclib.jsonrpc import ServerProxy, SafeTransport
except ImportError:
    print("Installing jsonrpclib-pelix...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "jsonrpclib-pelix"])
    from jsonrpclib import Server
    from jsonrpclib.jsonrpc import ServerProxy, SafeTransport

class SSLBypassTransport(SafeTransport):
    """Custom transport that bypasses SSL certificate verification"""
    
    def __init__(self, timeout=30):
        super().__init__()
        self.timeout = timeout
    
    def make_connection(self, host):
        """Create HTTPS connection that ignores SSL certificates"""
        try:
            # Try Python 3 approach first
            import http.client
            connection = http.client.HTTPSConnection(
                host,
                context=ssl_context,
                timeout=self.timeout
            )
            return connection
        except ImportError:
            # Fallback for Python 2
            import httplib
            connection = httplib.HTTPSConnection(
                host,
                timeout=self.timeout
            )
            # Monkey patch to use our SSL context
            if hasattr(connection, 'sock') and connection.sock:
                connection.sock = ssl_context.wrap_socket(
                    connection.sock,
                    server_hostname=host
                )
            return connection

def create_ssl_bypass_server(url, timeout=30):
    """Create a jsonrpc server with SSL bypass"""
    try:
        # Create custom transport
        transport = SSLBypassTransport(timeout=timeout)
        
        # Create server with custom transport
        server = ServerProxy(
            url,
            transport=transport,
            verbose=False
        )
        
        return server
    except Exception as e:
        print(f"Error creating SSL bypass server: {e}")
        # Fallback to regular server
        return Server(url)

def test_arista_connection():
    """Test connection to Arista device with SSL bypass"""
    
    # Get connection details from user
    host = input("Enter Arista device IP: ")
    username = input("Enter username: ")
    password = input("Enter password: ")
    
    # Use HTTPS with default port (443)
    protocol = "https"
    
    # Create the connection URL
    url = f"{protocol}://{username}:{password}@{host}/command-api"
    
    print(f"\nTesting connection to: {host}")
    print(f"Protocol: {protocol.upper()} (default port 443)")
    print(f"URL format: {protocol}://username:password@host/command-api")
    
    print("🔓 SSL certificate verification: DISABLED")
    
    try:
        # Set socket timeout
        socket.setdefaulttimeout(30)
        
        # Create server connection with SSL bypass
        print("Creating SSL bypass connection...")
        switch = create_ssl_bypass_server(url, timeout=30)
        
        # Test with show version
        print("Executing 'show version'...")
        result = switch.runCmds(version=1, cmds=['show version'], format='json')
        
        if result and len(result) > 0:
            print("✅ Connection successful!")
            version_info = result[0]
            print(f"Device Model: {version_info.get('modelName', 'Unknown')}")
            print(f"System MAC: {version_info.get('systemMacAddress', 'Unknown')}")
            print(f"Software Version: {version_info.get('version', 'Unknown')}")
            print(f"Serial Number: {version_info.get('serialNumber', 'Unknown')}")
            print(f"Hostname: {version_info.get('hostname', 'Unknown')}")
            
            # Test additional command
            try:
                print("\nTesting additional command: 'show hostname'...")
                hostname_result = switch.runCmds(version=1, cmds=['show hostname'], format='json')
                if hostname_result and len(hostname_result) > 0:
                    print(f"Hostname from command: {hostname_result[0].get('hostname', 'Unknown')}")
            except Exception as e:
                print(f"Additional command failed: {e}")
            
            return True
        else:
            print("❌ Connection failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        
        # Provide helpful error messages
        error_msg = str(e).lower()
        if "authentication failed" in error_msg or "unauthorized" in error_msg:
            print("💡 Check your username and password")
        elif "connection refused" in error_msg:
            print("💡 Check if eAPI is enabled on the device:")
            print("   Configure: 'management api http-commands'")
            print("   Configure: 'no shutdown' under the management api")
        elif "timeout" in error_msg or "timed out" in error_msg:
            print("💡 Check network connectivity and device IP")
            print("💡 Verify the device is reachable (try ping)")
        elif "ssl" in error_msg or "certificate" in error_msg:
            print("💡 SSL certificate issue detected")
            print("💡 Check device SSL configuration")
        elif "name or service not known" in error_msg:
            print("💡 DNS resolution failed - check the IP address")
        elif "no route to host" in error_msg:
            print("💡 Network routing issue - check network connectivity")
        else:
            print("💡 Unknown error - check device configuration and network")
            
        return False

def test_multiple_commands():
    """Test multiple commands on the device"""
    
    host = input("Enter Arista device IP: ")
    username = input("Enter username: ")
    password = input("Enter password: ")
    
    # Use HTTPS with default port (443)
    protocol = "https"
    
    # Create the connection URL
    url = f"{protocol}://{username}:{password}@{host}/command-api"
    
    commands_to_test = [
        'show version',
        'show hostname',
        'show ip interface brief',
        'show mac address-table count',
        'show interfaces status'
    ]
    
    try:
        socket.setdefaulttimeout(30)
        switch = create_ssl_bypass_server(url, timeout=30)
        
        print(f"\n🧪 Testing multiple commands on {host}:")
        print(f"Protocol: HTTPS (default port 443)")
        print("🔓 SSL certificate verification: DISABLED")
        print("=" * 50)
        
        for cmd in commands_to_test:
            try:
                print(f"\nExecuting: {cmd}")
                result = switch.runCmds(version=1, cmds=[cmd], format='json')
                if result and len(result) > 0:
                    print(f"✅ Success - Response size: {len(str(result[0]))} chars")
                else:
                    print("❌ No response")
            except Exception as e:
                print(f"❌ Failed: {str(e)}")
        
        print("\n" + "=" * 50)
        print("Multi-command test completed")
        
    except Exception as e:
        print(f"❌ Connection setup failed: {str(e)}")

if __name__ == "__main__":
    print("=== Arista eAPI Connection Test with SSL Bypass ===")
    print("This tool tests eAPI connectivity with SSL certificate bypass")
    print()
    
    test_type = input("Select test type:\n1. Basic connection test\n2. Multi-command test\nChoose (1-2, default: 1): ").strip()
    
    if test_type == "2":
        test_multiple_commands()
    else:
        test_arista_connection()
    
    input("\nPress Enter to exit...")