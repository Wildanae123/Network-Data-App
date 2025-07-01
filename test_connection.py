# backend/test_connection.py
"""
Simple test script to verify Arista eAPI connection using jsonrpclib
"""

try:
    from jsonrpclib import Server
except ImportError:
    print("Installing jsonrpclib-pelix...")
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "jsonrpclib-pelix"])
    from jsonrpclib import Server

def test_arista_connection():
    """Test connection to Arista device"""
    
    # Get connection details from user
    host = input("Enter Arista device IP: ")
    username = input("Enter username: ")
    password = input("Enter password: ")
    
    # Create the connection URL
    url = f"https://{username}:{password}@{host}/command-api"
    
    print(f"\nTesting connection to: {host}")
    print("URL format: https://username:password@host/command-api")
    
    try:
        # Create server connection
        switch = Server(url)
        
        # Test with show version
        print("Executing 'show version'...")
        result = switch.runCmds(version=1, cmds=['show version'], format='json')
        
        if result and len(result) > 0:
            print("✅ Connection successful!")
            print(f"Device Model: {result[0].get('modelName', 'Unknown')}")
            print(f"System MAC: {result[0].get('systemMacAddress', 'Unknown')}")
            print(f"Software Version: {result[0].get('version', 'Unknown')}")
            return True
        else:
            print("❌ Connection failed: No response")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        
        # Provide helpful error messages
        if "Authentication failed" in str(e):
            print("💡 Check your username and password")
        elif "Connection refused" in str(e):
            print("💡 Check if eAPI is enabled: 'management api http-commands'")
        elif "timeout" in str(e).lower():
            print("💡 Check network connectivity and device IP")
        elif "SSL" in str(e):
            print("💡 Try using HTTP instead of HTTPS if SSL is not configured")
            
        return False

if __name__ == "__main__":
    print("=== Arista eAPI Connection Test ===")
    test_arista_connection()
    input("\nPress Enter to exit...")