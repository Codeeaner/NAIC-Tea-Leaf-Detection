#!/usr/bin/env python3
"""
Installation script for Tea Leaf Analytics functionality.

This script helps set up the analytics dependencies and provides
guidance for Ollama installation.
"""

import subprocess
import sys
import os
import requests
import platform
from pathlib import Path

def install_python_dependencies():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "ollama==0.3.3", 
            "requests==2.31.0"
        ])
        print("✅ Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Python dependencies: {e}")
        return False

def check_ollama_installation():
    """Check if Ollama is installed and accessible."""
    print("🔍 Checking Ollama installation...")
    
    try:
        # Try to run ollama command
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama command failed")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found in PATH")
        return False

def check_ollama_service():
    """Check if Ollama service is running."""
    print("🔍 Checking Ollama service...")
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            
            # Check for qwen3-vl:235b-cloud Vision model
            models = response.json().get("models", [])
            llama_models = [m for m in models if "qwen3-vl:235b-cloud" in m.get("name", "")]
            
            if llama_models:
                print(f"✅ Found qwen3-vl:235b-cloud Vision model: {llama_models[0]['name']}")
                return True
            else:
                print("⚠️  qwen3-vl:235b-cloud Vision model not found")
                return False
        else:
            print(f"❌ Ollama service responded with status {response.status_code}")
            return False
    except requests.RequestException:
        print("❌ Cannot connect to Ollama service")
        return False

def install_ollama_model():
    """Install qwen3-vl:235b-cloud Vision model."""
    print("📥 Installing qwen3-vl:235b-cloud Vision model...")
    
    try:
        # Try to pull the model
        result = subprocess.run([
            "ollama", "pull", "qwen3-vl:235b-cloud"
        ], capture_output=True, text=True, timeout=1200)  # 20 minute timeout for larger model
        
        if result.returncode == 0:
            print("✅ qwen3-vl:235b-cloud Vision model installed successfully")
            return True
        else:
            print(f"❌ Failed to install model: {result.stderr}")
            
            # Try smaller model as fallback
            print("🔄 Trying smaller model as fallback...")
            result = subprocess.run([
                "ollama", "pull", "llama3.2-vision:11b"
            ], capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Llama 3.2 Vision 11B model installed as fallback")
                print("⚠️  Note: Update analytics_service.py to use 'llama3.2-vision:11b'")
                return True
            else:
                print(f"❌ Failed to install fallback model: {result.stderr}")
                return False
    except subprocess.TimeoutExpired:
        print("❌ Model installation timed out")
        return False
    except FileNotFoundError:
        print("❌ Ollama command not found")
        return False

def create_directories():
    """Create required directories."""
    print("📁 Creating required directories...")
    
    directories = ["analytics", "test_results"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}/")
    
    return True

def get_ollama_install_instructions():
    """Get platform-specific Ollama installation instructions."""
    system = platform.system().lower()
    
    if system == "windows":
        return """
🪟 Windows Installation:
1. Download Ollama from: https://ollama.com/download/windows
2. Run the installer and follow the setup wizard
3. Ollama will start automatically as a Windows service
4. Open Command Prompt and run: ollama pull qwen3-vl:235b-cloud
"""
    elif system == "darwin":
        return """
🍎 macOS Installation:
1. Download Ollama from: https://ollama.com/download/mac
2. Install the .app file
3. Open Terminal and run:
   curl -fsSL https://ollama.com/install.sh | sh
4. Start Ollama: ollama serve
5. In a new terminal: ollama pull qwen3-vl:235b-cloud
"""
    else:
        return """
🐧 Linux Installation:
1. Install Ollama:
   curl -fsSL https://ollama.com/install.sh | sh
2. Start Ollama service:
   ollama serve
3. In a new terminal, install the model:
   ollama pull qwen3-vl:235b-cloud
"""

def main():
    """Main installation process."""
    print("🚀 Tea Leaf Analytics Installation")
    print("=" * 50)
    
    steps = [
        ("Installing Python Dependencies", install_python_dependencies),
        ("Creating Directories", create_directories),
        ("Checking Ollama Installation", check_ollama_installation),
        ("Checking Ollama Service", check_ollama_service),
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n🔧 {step_name}...")
        results[step_name] = step_func()
    
    # Handle Ollama model installation separately
    if results.get("Checking Ollama Service"):
        print(f"\n🔧 Installing Qwen3-VL Model...")
        results["Installing Qwen3-VL Model"] = install_ollama_model()
    
    # Print summary
    print("\n" + "=" * 50)
    print("📋 INSTALLATION SUMMARY")
    print("=" * 50)
    
    for step_name, result in results.items():
        status = "✅ SUCCESS" if result else "❌ FAILED"
        print(f"{status} {step_name}")
    
    # Provide next steps
    print(f"\n📝 NEXT STEPS:")
    
    if not results.get("Checking Ollama Installation"):
        print("1. Install Ollama:")
        print(get_ollama_install_instructions())
    
    if results.get("Checking Ollama Installation") and not results.get("Checking Ollama Service"):
        print("1. Start Ollama service:")
        if platform.system().lower() == "windows":
            print("   - Ollama should start automatically")
            print("   - If not, run: ollama serve")
        else:
            print("   ollama serve")
    
    if not results.get("Installing Qwen3-VL Model"):
        print("2. Install Qwen3-VL Vision model:")
        print("   ollama pull qwen3-vl:235b-cloud")
    
    print("3. Test the installation:")
    print("   python test_analytics.py")
    
    print("4. Start the application:")
    print("   python run.py")
    
    if all(results.values()):
        print("\n🎉 Installation completed successfully!")
        print("Analytics functionality is ready to use.")
    else:
        print("\n⚠️  Installation incomplete. Please follow the next steps above.")
    
    return all(results.values())

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)