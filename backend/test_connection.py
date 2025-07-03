# backend/test_connection.py
"""
Simple test script to verify Arista eAPI connection using jsonrpclib with SSL bypass
"""

import ssl
import socket
import urllib3
import getpass
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    from jsonrpclib import Server
except ImportError:
    print("Menginstal jsonrpclib-pelix...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "jsonrpclib-pelix"])
        from jsonrpclib import Server
    except subprocess.CalledProcessError as e:
        print(f"Gagal menginstal jsonrpclib-pelix. Error: {e}")
        print("Silakan instal manual: pip install jsonrpclib-pelix")
        sys.exit(1)

def test_arista_connection():
    """Menguji koneksi ke perangkat Arista dengan bypass SSL."""

    # Ambil detail koneksi dari pengguna
    host = input("Masukkan IP perangkat Arista: ")
    username = input("Masukkan username: ")
    # Gunakan getpass agar password tidak terlihat saat diketik
    password = getpass.getpass("Masukkan password: ")

    protocol = "https"
    url = f"{protocol}://{username}:{password}@{host}/command-api"

    print(f"\nMenguji koneksi ke: {host}")
    print(f"Protokol: {protocol.upper()} (port default 443)")
    print("🔓 Verifikasi sertifikat SSL: DINONAKTIFKAN (Metode Global)")

    try:
        # Atur timeout default untuk semua koneksi socket
        socket.setdefaulttimeout(30)

        # Buat koneksi server. Tidak perlu transport kustom lagi.
        print("Membuat koneksi...")
        switch = Server(url)

        # Uji dengan perintah 'show version'
        print("Menjalankan 'show version'...")
        result = switch.runCmds(version=1, cmds=['show version'], format='json')

        if result and len(result) > 0:
            print("\n✅ Koneksi berhasil!")
            version_info = result[0]
            print(f"  Model Perangkat: {version_info.get('modelName', 'Tidak diketahui')}")
            print(f"  System MAC: {version_info.get('systemMacAddress', 'Tidak diketahui')}")
            print(f"  Versi Software: {version_info.get('version', 'Tidak diketahui')}")
            print(f"  Nomor Seri: {version_info.get('serialNumber', 'Tidak diketahui')}")
            print(f"  Hostname: {version_info.get('hostname', 'Tidak diketahui')}")
            return True
        else:
            print("❌ Koneksi gagal: Tidak ada respons dari perangkat.")
            return False

    except Exception as e:
        print(f"\n❌ Koneksi gagal: {str(e)}")

        # Pesan bantuan yang informatif
        error_msg = str(e).lower()
        if "authentication failed" in error_msg or "unauthorized" in error_msg:
            print("💡 Cek kembali username dan password Anda.")
        elif "connection refused" in error_msg:
            print("💡 Pastikan eAPI sudah diaktifkan di perangkat:")
            print("   (config)# management api http-commands")
            print("   (config-mgmt-api-http-cmds)# no shutdown")
        elif "timeout" in error_msg or "timed out" in error_msg:
            print("💡 Cek konektivitas jaringan dan IP perangkat.")
            print("💡 Coba lakukan ping ke perangkat.")
        elif "ssl" in error_msg or "certificate" in error_msg:
            # Pesan ini seharusnya tidak muncul lagi, tapi tetap berguna
            print("💡 Terdeteksi masalah SSL.")
        elif "name or service not known" in error_msg or "no address associated" in error_msg:
            print("💡 Resolusi DNS gagal. Pastikan IP atau hostname sudah benar.")
        elif "no route to host" in error_msg:
            print("💡 Masalah routing jaringan. Cek gateway atau firewall.")
        else:
            print("💡 Terjadi kesalahan yang tidak diketahui. Periksa konfigurasi perangkat dan jaringan.")

        return False

def test_multiple_commands():
    """Menguji beberapa perintah di perangkat."""

    host = input("Masukkan IP perangkat Arista: ")
    username = input("Masukkan username: ")
    password = getpass.getpass("Masukkan password: ")

    protocol = "https"
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
        switch = Server(url)

        print(f"\n🧪 Menguji beberapa perintah di {host}:")
        print("=" * 50)

        for cmd in commands_to_test:
            try:
                print(f"\nMenjalankan: {cmd}")
                result = switch.runCmds(version=1, cmds=[cmd], format='json')
                if result and len(result) > 0:
                    print(f"✅ Sukses - Ukuran respons: {len(str(result[0]))} karakter")
                else:
                    print("❌ Tidak ada respons")
            except Exception as e:
                print(f"❌ Gagal: {str(e)}")

        print("\n" + "=" * 50)
        print("Pengujian multi-perintah selesai.")

    except Exception as e:
        print(f"❌ Pengaturan koneksi gagal: {str(e)}")

if __name__ == "__main__":
    print("=== Uji Koneksi Arista eAPI dengan Bypass SSL ===")
    print("Alat ini menguji konektivitas eAPI dengan menonaktifkan verifikasi sertifikat SSL.\n")

    while True:
        test_type = input("Pilih tipe pengujian:\n1. Uji koneksi dasar\n2. Uji multi-perintah\nPilih (1-2, default: 1): ").strip()
        if test_type in ["1", "2", ""]:
            break
        print("Pilihan tidak valid, silakan pilih 1 atau 2.")

    if test_type == "2":
        test_multiple_commands()
    else:
        test_arista_connection()

    input("\nTekan Enter untuk keluar...")