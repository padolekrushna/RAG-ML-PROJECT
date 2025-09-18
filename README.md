#  RAG Document Q&A System

A modern, attractive document question-answering system built with **RAG (Retrieval Augmented Generation)**, **FAISS vector storage**, and **Google Gemini AI**. Upload your documents and chat with them using natural language!

## ✨ Features

- **📄 Multi-format Support**: PDF, TXT, and DOCX files
- **🔍 Advanced RAG Pipeline**: Combines document retrieval with AI generation
- **⚡ FAISS Vector Search**: Fast and efficient similarity search
- **🧠 Google Gemini Integration**: Powered by Google's advanced AI
- **🎨 Beautiful UI**: Modern, responsive Streamlit interface
- **💬 Chat Interface**: Intuitive conversation-style Q&A
- **📊 Real-time Stats**: Monitor system performance and usage

## 🚀 Quick Start

### 1. Clone or Download the Project
Download all the project files to your local directory.

### 2. Install Dependencies
```bash
# Option 1: Auto-install (recommended)
python run.py

# Option 2: Manual install
pip install -r requirements.txt
```

### 3. Get Your Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy your API key

### 4. Configure API Key
**Option A: Using .env file (recommended)**
1. Open `.env` file
2. Replace `your_gemini_api_key_here` with your actual API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

**Option B: Enter in the app**
- You can also enter your API key directly in the sidebar when running the app

### 5. Run the Application
```bash
# Option 1: Using run script (recommended)
python run.py

# Option 2: Direct Streamlit
streamlit run app.py
```

### 6. Open in Browser
Navigate to: `http://localhost:8501`

## 📁 Project Structure

```
rag-document-qa/
├── app.py                 # Main Streamlit application
├── document_processor.py  # Document processing and chunking
├── vector_store.py        # FAISS vector storage management
├── rag_pipeline.py        # RAG pipeline implementation
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── run.py               # Setup and run script
├── README.md            # This file
└── uploads/             # Document upload directory (auto-created)
└── indexes/             # FAISS indexes storage (auto-created)
```

## 🛠️ How to Use

### Step 1: Upload Documents
- Click "📄 Upload Documents" in the sidebar
- Select PDF, TXT, or DOCX files (multiple files supported)
- Click "🚀 Process Documents"

### Step 2: Wait for Processing
- Documents will be chunked and converted to embeddings
- FAISS index will be created for fast retrieval
- Progress bar shows processing status

### Step 3: Ask Questions
- Type your questions in the chat input
- The system will find relevant document chunks
- Gemini AI will generate comprehensive answers

### Step 4: Review Responses
- Get detailed answers based on your documents
- See real-time statistics
- Clear chat history when needed

## 🔧 Configuration

### Chunk Settings
You can modify chunking parameters in `document_processor.py`:
```python
chunk_size = 1000      # Size of each text chunk
chunk_overlap = 200    # Overlap between chunks
```

### Retrieval Settings
Adjust retrieval parameters in `rag_pipeline.py`:
```python
k = 5  # Number of similar chunks to retrieve
```

### Model Settings
Change the Gemini model in `rag_pipeline.py`:
```python
model_name = "gemini-pro"  # or other available models
```

## 🎨 UI Features

- **Modern Gradient Design**: Beautiful color schemes and animations
- **Responsive Layout**: Works on desktop and mobile
- **Real-time Updates**: Live statistics and progress indicators
- **Interactive Elements**: Hover effects and smooth transitions
- **Clear Visual Hierarchy**: Easy-to-follow information structure

## 🔍 Technical Details

### RAG Pipeline
1. **Document Processing**: Text extraction and chunking
2. **Embedding Generation**: Convert text to vector representations
3. **Vector Storage**: Store embeddings in FAISS index
4. **Query Processing**: Convert user questions to embeddings
5. **Retrieval**: Find most similar document chunks
6. **Generation**: Use Gemini to create comprehensive answers

### Vector Storage
- Uses **FAISS** (Facebook AI Similarity Search) for efficient vector operations
- Supports cosine similarity search
- Handles large document collections
- Persistent storage for reusing indexes

### AI Integration
- **Google Gemini Pro** for text generation
- **Sentence Transformers** for embedding generation (fallback)
- Intelligent context construction for better responses

## 🐛 Troubleshooting

### Common Issues

**1. API Key Error**
```
Error: API key not found
```
- Make sure you've set your Gemini API key in `.env` or the sidebar
- Verify the API key is valid and active

**2. Import Error**
```
ModuleNotFoundError: No module named 'streamlit'
```
- Run: `pip install -r requirements.txt`
- Or use: `python run.py` for auto-installation

**3. Document Processing Error**
```
Error processing document
```
- Check file format (PDF, TXT, DOCX only)
- Ensure file isn't corrupted or password-protected
- Try with a different file

**4. Memory Issues**
```
Out of memory error
```
- Reduce `chunk_size` in `document_processor.py`
- Process fewer documents at once
- Close other applications to free memory

### Performance Tips

- **Large Documents**: Use smaller chunk sizes for better processing
- **Multiple Files**: Process in batches if you have many documents
- **Query Optimization**: Be specific in your questions for better results
- **Index Persistence**: Save/load indexes to avoid reprocessing

## 📚 Dependencies

- **streamlit**: Web application framework
- **google-generativeai**: Google Gemini AI integration
- **faiss-cpu**: Vector similarity search
- **sentence-transformers**: Text embeddings
- **PyPDF2**: PDF processing
- **python-docx**: Word document processing
- **numpy**: Numerical operations
- **pandas**: Data manipulation

## 🤝 Support

If you encounter any issues:

1. Check the troubleshooting section above
2. Ensure all dependencies are installed correctly
3. Verify your API key is valid
4. Try with sample documents first

## 📄 License

This project is open source and available under the MIT License.

## 🎯 Future Enhancements

- [ ] Support for more document formats (PPTX, HTML, etc.)
- [ ] Multi-language document support
- [ ] Advanced search filters
- [ ] Document summarization
- [ ] Export chat conversations
- [ ] User authentication
- [ ] Database integration for persistent storage

---

**Happy Document Chatting! 🚀**
