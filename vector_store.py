import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
import streamlit as st

class VectorStore:
    """Handles vector storage and retrieval using FAISS"""
    
    def __init__(self, api_key: str, embedding_dim: int = 384):
        """
        Initialize vector store
        
        Args:
            api_key: Google Gemini API key
            embedding_dim: Dimension of embeddings
        """
        self.api_key = api_key
        self.embedding_dim = embedding_dim
        self.index = None
        self.documents = []
        self.embeddings = []
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Test API connection
        try:
            # Try to list models to verify API key works
            models = list(genai.list_models())
            if models:
                st.success(f"✅ API key verified! Found {len(models)} available models")
                # Log some available model names for debugging
                model_names = [m.name for m in models[:3]]
                st.info(f"📋 Sample models: {', '.join(model_names)}")
            else:
                st.warning("⚠️ API key works but no models found")
        except Exception as e:
            st.error(f"❌ API key verification failed: {str(e)}")
            st.error("Please check your Gemini API key and try again")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using TF-IDF only
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            # Use TF-IDF based embedding directly
            return self._create_tfidf_embedding(text)
            
        except Exception as e:
            st.error(f"Error generating TF-IDF embedding: {str(e)}")
            # Final fallback: create a simple hash-based embedding
            return self._create_simple_embedding(text)
    
    def _create_tfidf_embedding(self, text: str) -> List[float]:
        """Create TF-IDF based embedding"""
        try:
            if not hasattr(self, '_tfidf_vectorizer'):
                # If vectorizer not initialized, create a simple one
                st.warning("TF-IDF vectorizer not initialized. Creating simple embedding...")
                return self._create_simple_embedding(text)
            
            # Transform text to vector using pre-fitted vectorizer
            vector = self._tfidf_vectorizer.transform([text]).toarray()[0]
            
            # Ensure consistent dimension (384)
            target_dim = 384
            if len(vector) < target_dim:
                # Pad with zeros
                vector = list(vector) + [0.0] * (target_dim - len(vector))
            elif len(vector) > target_dim:
                # Truncate
                vector = vector[:target_dim]
            else:
                vector = list(vector)
            
            return vector
            
        except Exception as e:
            st.warning(f"⚠️ TF-IDF embedding failed: {str(e)}")
            return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create a simple hash-based embedding as final fallback"""
        try:
            # This is a very basic approach for demonstration
            # In production, use proper embedding models
            import hashlib
            
            # Create multiple hash values for better distribution
            text_bytes = text.encode('utf-8')
            
            # Create embedding vector using multiple hash functions
            embedding = []
            hash_funcs = [hashlib.md5, hashlib.sha1, hashlib.sha256]
            
            for i in range(self.embedding_dim):
                # Use different parts of different hashes
                hash_func = hash_funcs[i % len(hash_funcs)]
                hash_input = f"{text}_{i}".encode('utf-8')
                hash_val = int(hash_func(hash_input).hexdigest()[:8], 16)
                
                # Normalize to [-1, 1] range
                normalized_val = (hash_val % 2000 - 1000) / 1000.0
                embedding.append(normalized_val)
            
            return embedding
            
        except Exception as e:
            st.error(f"Even simple embedding failed: {str(e)}")
            # Absolute final fallback
            return [0.1] * self.embedding_dim
    
    def create_index(self, documents: List[str]) -> None:
        """
        Create FAISS index from documents
        
        Args:
            documents: List of document chunks
        """
        try:
            if not documents:
                raise ValueError("No documents provided")
            
            if len(documents) < 2:
                st.warning("⚠️ Only one document chunk found. Adding padding for TF-IDF to work properly.")
                # Add some padding documents to make TF-IDF work
                documents.extend([
                    "This is a padding document for TF-IDF processing.",
                    "Another padding document to ensure proper TF-IDF functionality."
                ])
            
            self.documents = documents
            
            # Initialize TF-IDF vectorizer with all documents at once
            self._initialize_tfidf_vectorizer(documents)
            
            # Generate embeddings for all documents
            embeddings = []
            progress_bar = st.progress(0)
            
            for i, doc in enumerate(documents):
                embedding = self.get_embedding(doc)
                embeddings.append(embedding)
                progress_bar.progress((i + 1) / len(documents))
            
            progress_bar.empty()
            
            self.embeddings = np.array(embeddings, dtype='float32')
            
            # Update embedding dimension based on actual embeddings
            if len(embeddings) > 0:
                self.embedding_dim = len(embeddings[0])
            
            # Create FAISS index
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(self.embeddings)
            
            # Add embeddings to index
            self.index.add(self.embeddings)
            
            st.success(f"✅ Created FAISS index with {len(documents)} documents (embedding dim: {self.embedding_dim})")
            
            # Show index statistics
            self._show_index_stats()
            
        except Exception as e:
            st.error(f"Error creating index: {str(e)}")
            raise
    
    def _initialize_tfidf_vectorizer(self, documents: List[str]) -> None:
        """Initialize TF-IDF vectorizer with all documents at once"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Create vectorizer with proper parameters
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=384,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b\w+\b',
                ngram_range=(1, 2),
                min_df=1,  # At least 1 document
                max_df=0.95,  # At most 95% of documents
                sublinear_tf=True,  # Use sublinear TF scaling
                norm='l2'  # L2 normalization
            )
            
            # Fit with all documents
            self._tfidf_vectorizer.fit(documents)
            self.embedding_dim = len(self._tfidf_vectorizer.get_feature_names_out())
            
            st.info(f"📊 Initialized TF-IDF with {len(documents)} documents, {self.embedding_dim} features")
            
        except Exception as e:
            st.error(f"Error initializing TF-IDF: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        try:
            if self.index is None:
                raise ValueError("Index not created. Call create_index first.")
            
            # Get query embedding
            query_embedding = np.array([self.get_embedding(query)], dtype='float32')
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.index.search(query_embedding, k)
            
            # Return results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append((self.documents[idx], float(score)))
            
            # Show search visualization in sidebar
            if hasattr(st, 'sidebar'):
                with st.sidebar:
                    if st.checkbox("🔍 Show Search Visualization", key=f"viz_{hash(query)}"):
                        self.show_search_visualization(query, results)
            
            return results
            
        except Exception as e:
            st.error(f"Error searching: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            'total_documents': len(self.documents),
            'embedding_dimension': self.embedding_dim,
            'index_size': self.index.ntotal if self.index else 0,
            'memory_usage': f"{len(self.embeddings) * self.embedding_dim * 4 / (1024*1024):.2f} MB" if len(self.embeddings) > 0 else "0 MB"
        }
    
    def save_index(self, filepath: str) -> None:
        """Save FAISS index and documents to file"""
        try:
            if self.index is None:
                raise ValueError("No index to save")
            
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.index")
            
            # Save documents and metadata
            metadata = {
                'documents': self.documents,
                'embeddings': self.embeddings,
                'embedding_dim': self.embedding_dim
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            st.success(f"✅ Index saved to {filepath}")
            
        except Exception as e:
            st.error(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str) -> None:
        """Load FAISS index and documents from file"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.index")
            
            # Load documents and metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.documents = metadata['documents']
            self.embeddings = metadata['embeddings']
            self.embedding_dim = metadata['embedding_dim']
            
            st.success(f"✅ Index loaded from {filepath}")
            
        except Exception as e:
            st.error(f"Error loading index: {str(e)}")
            raise
    
    @property
    def total_chunks(self) -> int:
        """Get total number of document chunks"""
        return len(self.documents)
    
    def _show_index_stats(self) -> None:
        """Show FAISS index statistics and visualization"""
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.decomposition import PCA
            
            if self.index is None or len(self.embeddings) == 0:
                return
                
            st.markdown("### 📊 FAISS Index Visualization")
            
            # Basic stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total Vectors", self.index.ntotal)
            with col2:
                st.metric("🔢 Vector Dimension", self.embedding_dim)
            with col3:
                st.metric("💾 Index Type", type(self.index).__name__)
            with col4:
                memory_size = self.embeddings.nbytes / (1024 * 1024)
                st.metric("🗄️ Memory Usage", f"{memory_size:.2f} MB")
            
            # Show sample vectors
            if st.checkbox("🔍 Show Sample Vectors"):
                self._show_sample_vectors()
            
            # Show 2D visualization
            if st.checkbox("📈 Show 2D Vector Plot") and len(self.embeddings) > 1:
                self._show_2d_plot()
                
        except Exception as e:
            st.warning(f"⚠️ Could not show index stats: {str(e)}")
    
    def _show_sample_vectors(self) -> None:
        """Show sample vectors from the index"""
        try:
            st.markdown("#### 🔢 Sample Vector Data")
            
            # Show first few vectors
            num_samples = min(3, len(self.embeddings))
            for i in range(num_samples):
                with st.expander(f"Vector {i+1} (First 20 dimensions)"):
                    vector = self.embeddings[i][:20]  # Show first 20 dimensions
                    doc_preview = self.documents[i][:100] + "..." if len(self.documents[i]) > 100 else self.documents[i]
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write("**Document Preview:**")
                        st.write(doc_preview)
                    with col2:
                        st.write("**Vector Values:**")
                        st.write(vector.tolist())
                        
        except Exception as e:
            st.error(f"Error showing sample vectors: {str(e)}")
    
    def _show_2d_plot(self) -> None:
        """Show 2D visualization of vectors using PCA"""
        try:
            import plotly.express as px
            from sklearn.decomposition import PCA
            import pandas as pd
            
            st.markdown("#### 📈 2D Vector Visualization (PCA)")
            
            # Apply PCA to reduce to 2D
            if len(self.embeddings) < 2:
                st.warning("Need at least 2 vectors for PCA visualization")
                return
                
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(self.embeddings)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'PC1': vectors_2d[:, 0],
                'PC2': vectors_2d[:, 1],
                'Document': [f"Doc {i+1}" for i in range(len(self.embeddings))],
                'Preview': [doc[:50] + "..." if len(doc) > 50 else doc for doc in self.documents]
            })
            
            # Create interactive plot
            fig = px.scatter(df, x='PC1', y='PC2', 
                           hover_name='Document', 
                           hover_data=['Preview'],
                           title='Document Vectors in 2D Space (PCA)',
                           labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                                  'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'})
            
            fig.update_traces(marker=dict(size=10, opacity=0.7))
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show explained variance
            st.info(f"📊 PCA Explained Variance: PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")
            
        except Exception as e:
            st.error(f"Error creating 2D plot: {str(e)}")
    
    def show_search_visualization(self, query: str, results: List[Tuple[str, float]]) -> None:
        """Show search results visualization"""
        try:
            st.markdown("#### 🎯 Search Results Visualization")
            
            if not results:
                st.warning("No search results to visualize")
                return
            
            # Create bar chart of similarity scores
            import plotly.graph_objects as go
            
            doc_names = [f"Doc {i+1}" for i in range(len(results))]
            scores = [score for _, score in results]
            
            fig = go.Figure(data=[
                go.Bar(x=doc_names, y=scores, 
                      text=[f"{score:.3f}" for score in scores],
                      textposition='auto')
            ])
            
            fig.update_layout(
                title=f'Similarity Scores for Query: "{query[:50]}..."',
                xaxis_title='Documents',
                yaxis_title='Similarity Score',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show detailed results
            st.markdown("#### 📋 Detailed Search Results")
            for i, (doc, score) in enumerate(results):
                with st.expander(f"Result {i+1} (Score: {score:.4f})"):
                    st.write(doc[:500] + "..." if len(doc) > 500 else doc)
                    
        except Exception as e:
            st.error(f"Error showing search visualization: {str(e)}")