#!/usr/bin/env python3
"""
Safe dependency installation script for RAG Document Q&A System
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status"""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ Success: {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {command}")
        print(f"Error: {e.stderr}")
        return False

def install_dependencies():
    """Install dependencies with compatibility fixes"""
    print("🔧 Installing dependencies with compatibility fixes...")
    
    # Core dependencies first
    core_deps = [
        "pip install --upgrade pip",
        "pip install numpy>=1.26.2",
        "pip install streamlit==1.28.1",
        "pip install google-generativeai==0.3.2",
        "pip install faiss-cpu==1.7.4",
        "pip install pandas==2.0.3",
        "pip install PyPDF2==3.0.1",
        "pip install python-docx==0.8.11",
        "pip install plotly==5.17.0",
        "pip install streamlit-option-menu==0.3.6",
        "pip install python-dotenv==1.0.0",
        "pip install scikit-learn>=1.0.0"
    ]
    
    # Install core dependencies
    for cmd in core_deps:
        if not run_command(cmd):
            print(f"⚠️ Warning: Failed to install {cmd}")
    
    # Try to install sentence-transformers with fallbacks
    print("\n🤖 Installing sentence-transformers...")
    
    # Try different versions/approaches
    transformers_options = [
        "pip install sentence-transformers>=2.7.0 transformers>=4.21.0 huggingface_hub>=0.16.0",
        "pip install sentence-transformers==2.6.1 transformers==4.33.0 huggingface_hub==0.17.0",
        "pip install --upgrade huggingface_hub && pip install sentence-transformers",
        "pip install sentence-transformers --no-deps && pip install transformers huggingface_hub tokenizers"
    ]
    
    transformers_installed = False
    for cmd in transformers_options:
        print(f"🔄 Trying: {cmd}")
        if run_command(cmd):
            transformers_installed = True
            break
        print("⚠️ Failed, trying next option...")
    
    if not transformers_installed:
        print("⚠️ Warning: sentence-transformers installation failed.")
        print("📝 The app will use TF-IDF embeddings as fallback.")
    else:
        print("✅ sentence-transformers installed successfully!")
    
    # Install torch separately if needed
    print("\n🔥 Checking PyTorch installation...")
    try:
        import torch
        print("✅ PyTorch already available")
    except ImportError:
        print("🔄 Installing PyTorch...")
        run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print("\n✅ Dependency installation complete!")
    print("📝 If you see warnings above, the app will use fallback methods.")
    return True

def main():
    print("🚀 RAG Document Q&A - Dependency Installation")
    print("=" * 50)
    
    try:
        install_dependencies()
        print("\n🎉 Installation completed!")
        print("📌 You can now run: streamlit run app.py")
        
        # Ask if user wants to run the app
        response = input("\n❓ Do you want to run the app now? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("🚀 Starting the app...")
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
            
    except Exception as e:
        print(f"❌ Installation failed: {str(e)}")
        print("📝 Please try manual installation with requirements.txt")

if __name__ == "__main__":
    main()