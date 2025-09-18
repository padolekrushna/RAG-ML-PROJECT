#!/usr/bin/env python3
"""
Run script for the RAG Document Q&A System
"""

import subprocess
import sys
import os
from pathlib import Path

#!/usr/bin/env python3
"""
Run script for the RAG Document Q&A System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages with better error handling"""
    try:
        print("🔄 Installing requirements...")
        
        # Try the custom installation script first
        try:
            exec(open('install_dependencies.py').read())
            return True
        except FileNotFoundError:
            pass
        
        # Fallback to requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Some packages may have failed to install: {e}")
        print("📝 The app will use fallback methods for any missing dependencies.")
        return True  # Continue anyway, app has fallbacks

def test_imports():
    """Test critical imports"""
    print("🧪 Testing critical imports...")
    
    try:
        import streamlit
        print("✅ Streamlit: OK")
    except ImportError:
        print("❌ Streamlit: FAILED - This is required!")
        return False
    
    try:
        import google.generativeai
        print("✅ Google GenerativeAI: OK")
    except ImportError:
        print("❌ Google GenerativeAI: FAILED - This is required!")
        return False
    
    try:
        import faiss
        print("✅ FAISS: OK")
    except ImportError:
        print("❌ FAISS: FAILED - This is required!")
        return False
    
    try:
        import sentence_transformers
        print("✅ Sentence Transformers: OK")
    except ImportError:
        print("⚠️ Sentence Transformers: Not available - Will use TF-IDF fallback")
    
    try:
        import sklearn
        print("✅ Scikit-learn: OK")
    except ImportError:
        print("⚠️ Scikit-learn: Not available - Limited fallback options")
    
    return True

def check_api_key():
    """Check if API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your_gemini_api_key_here':
        print("⚠️  Warning: Google Gemini API key not configured!")
        print("Please update the .env file with your actual API key or enter it in the app.")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'indexes', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("📁 Directories created successfully!")

def run_streamlit():
    """Run the Streamlit application"""
    try:
        print("🚀 Starting the RAG Document Q&A System...")
        print("📝 Open your browser and go to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server\n")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    print("🤖 RAG Document Q&A System Setup")
    print("=" * 40)
    
    # Install requirements
    if not install_requirements():
        print("❌ Critical dependencies failed to install. Exiting.")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Critical imports failed. Please check your installation.")
        return
    
    # Create directories
    create_directories()
    
    # Check API key
    check_api_key()
    
    print("\n🎯 All checks passed! Starting the application...")
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()