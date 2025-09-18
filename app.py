import streamlit as st
import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from document_processor import DocumentProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🤖 AI Document Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
    }
    
    .bot-message {
        background: #ffffff;
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False

def main():
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Document Q&A System</h1>
        <p>Upload documents and ask questions using advanced RAG with FAISS and Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration and document upload
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if api_key:
            os.environ['GOOGLE_API_KEY'] = api_key
            genai.configure(api_key=api_key)
        
        st.markdown("---")
        
        # Document upload section
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, TXT, or DOCX files"
        )
        
        # Process documents button
        if st.button("🚀 Process Documents", key="process_btn"):
            if not api_key:
                st.error("❌ Please provide your Gemini API key!")
                return
            
            if not uploaded_files:
                st.error("❌ Please upload at least one document!")
                return
            
            with st.spinner("🔄 Processing documents..."):
                try:
                    # Initialize components
                    doc_processor = DocumentProcessor()
                    vector_store = VectorStore(api_key)
                    
                    # Process documents
                    all_chunks = []
                    for file in uploaded_files:
                        chunks = doc_processor.process_document(file)
                        all_chunks.extend(chunks)
                    
                    if all_chunks:
                        # Create vector store
                        vector_store.create_index(all_chunks)
                        
                        # Initialize RAG pipeline
                        st.session_state.rag_pipeline = RAGPipeline(
                            vector_store=vector_store,
                            api_key=api_key
                        )
                        st.session_state.documents_processed = True
                        
                        st.success(f"✅ Successfully processed {len(all_chunks)} document chunks!")
                    else:
                        st.error("❌ No text could be extracted from the documents!")
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
        
        st.markdown("---")
        
        # System status
        if st.session_state.documents_processed:
            st.markdown("""
            <div class="success-box">
                <strong>✅ System Ready</strong><br>
                Documents processed and ready for Q&A!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="error-box">
                <strong>⏳ System Not Ready</strong><br>
                Please upload and process documents first.
            </div>
            """, unsafe_allow_html=True)
    
    # Main chat interface
    st.markdown("### 💬 Chat with Your Documents")
    
    # Create columns for layout after the chat input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="bot-message">
                    <strong>🤖 AI Assistant:</strong><br>{message["content"]}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📊 Quick Stats")
        
        if st.session_state.documents_processed and st.session_state.rag_pipeline:
            stats_container = st.container()
            with stats_container:
                st.metric("📄 Document Chunks", 
                         st.session_state.rag_pipeline.vector_store.total_chunks)
                st.metric("💬 Chat Messages", len(st.session_state.messages))
                
                # Clear chat button
                if st.button("🗑️ Clear Chat", key="clear_btn"):
                    st.session_state.messages = []
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ How to Use")
        st.markdown("""
        1. **Add API Key**: Enter your Google Gemini API key
        2. **Upload Documents**: Add PDF, TXT, or DOCX files
        3. **Process**: Click 'Process Documents' button
        4. **Chat**: Ask questions about your documents
        """)
    
    # Chat input - MUST be outside columns
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.documents_processed:
            st.error("❌ Please upload and process documents first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        with st.spinner("🤔 Thinking..."):
            try:
                response = st.session_state.rag_pipeline.get_response(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Rerun to show the new messages
                st.rerun()
                
            except Exception as e:
                error_msg = f"❌ Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()