import google.generativeai as genai
from vector_store import VectorStore
from typing import List, Dict, Any
import streamlit as st

class RAGPipeline:
    """RAG (Retrieval Augmented Generation) pipeline using Gemini and FAISS"""
    
    def __init__(self, vector_store: VectorStore, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: Vector store instance
            api_key: Google Gemini API key
            model_name: Gemini model name
        """
        self.vector_store = vector_store
        self.api_key = api_key
        self.model_name = model_name
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize the model with error handling
        try:
            # Try different model names in order of preference
            model_options = [
                "gemini-1.5-flash",
                "gemini-1.5-pro", 
                "gemini-pro",
                "models/gemini-1.5-flash",
                "models/gemini-pro"
            ]
            
            self.model = None
            for model_name_option in model_options:
                try:
                    self.model = genai.GenerativeModel(model_name_option)
                    self.model_name = model_name_option
                    st.success(f"✅ Successfully initialized model: {model_name_option}")
                    break
                except Exception as model_error:
                    st.warning(f"⚠️ Failed to load {model_name_option}: {str(model_error)}")
                    continue
            
            if self.model is None:
                raise Exception("No compatible Gemini model found")
                
        except Exception as e:
            st.error(f"Error initializing Gemini model: {str(e)}")
            st.error("Please check your API key and try again")
            raise
    
    def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant context from vector store
        
        Args:
            query: User query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant document chunks
        """
        try:
            results = self.vector_store.search(query, k=k)
            
            # Extract just the document text (without scores)
            context_docs = [doc for doc, score in results]
            
            return context_docs
            
        except Exception as e:
            st.error(f"Error retrieving context: {str(e)}")
            return []
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate response using Gemini with retrieved context
        
        Args:
            query: User query
            context: Retrieved context documents
            
        Returns:
            Generated response
        """
        try:
            # Create prompt with context
            context_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context)])
            
            prompt = f"""You are a helpful AI assistant that answers questions based on the provided documents. 
Use the following context to answer the user's question. If you cannot find the answer in the context, 
say so clearly.

Context:
{context_text}

Question: {query}

Answer: Please provide a comprehensive answer based on the context above. If the information is not available 
in the context, please mention that."""

            # Generate response using Gemini with safety settings
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1000,
                        top_p=0.8,
                        top_k=40
                    ),
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
                    ]
                )
                
                if response.text:
                    return response.text
                else:
                    return "I apologize, but I couldn't generate a response. The model might be experiencing issues or the content was blocked by safety filters."
                    
            except Exception as generation_error:
                # Try with simpler generation if the above fails
                st.warning(f"⚠️ Advanced generation failed: {str(generation_error)}")
                st.info("🔄 Trying simpler generation method...")
                
                response = self.model.generate_content(prompt)
                if response and hasattr(response, 'text') and response.text:
                    return response.text
                else:
                    return "I apologize, but I couldn't generate a response. Please try rephrasing your question or check your API key."
                
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            st.error(error_message)
            
            # Provide helpful error messages based on error type
            if "404" in str(e):
                st.error("🔧 Model not found. This might be due to:")
                st.error("• API key issues")
                st.error("• Model availability changes")
                st.error("• Regional restrictions")
            elif "401" in str(e) or "403" in str(e):
                st.error("🔐 Authentication issue. Please check your API key.")
            elif "quota" in str(e).lower():
                st.error("📊 API quota exceeded. Please check your usage limits.")
            
            return error_message
    
    def get_response(self, query: str, k: int = 5) -> str:
        """
        Get complete RAG response for a query
        
        Args:
            query: User query
            k: Number of context documents to retrieve
            
        Returns:
            Generated response
        """
        try:
            # Step 1: Retrieve relevant context
            context = self.retrieve_context(query, k=k)
            
            if not context:
                return "I couldn't find any relevant information in the uploaded documents to answer your question."
            
            # Step 2: Generate response with context
            response = self.generate_response(query, context)
            
            return response
            
        except Exception as e:
            error_message = f"Error in RAG pipeline: {str(e)}"
            st.error(error_message)
            return error_message
    
    def get_response_with_sources(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Get response with source information
        
        Args:
            query: User query
            k: Number of context documents to retrieve
            
        Returns:
            Dictionary with response and sources
        """
        try:
            # Step 1: Retrieve relevant context with scores
            search_results = self.vector_store.search(query, k=k)
            
            if not search_results:
                return {
                    "response": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "confidence": 0.0
                }
            
            # Extract context and sources
            context = [doc for doc, score in search_results]
            sources = []
            
            for i, (doc, score) in enumerate(search_results):
                # Extract source name from document if available
                source_info = {
                    "chunk_id": i + 1,
                    "similarity_score": float(score),
                    "preview": doc[:200] + "..." if len(doc) > 200 else doc
                }
                
                # Try to extract source name from document
                if doc.startswith('[Source:'):
                    source_line = doc.split('\n')[0]
                    source_name = source_line.replace('[Source:', '').replace(']', '').strip()
                    source_info["source_name"] = source_name
                
                sources.append(source_info)
            
            # Step 2: Generate response with context
            response = self.generate_response(query, context)
            
            # Calculate average confidence score
            avg_confidence = sum(score for _, score in search_results) / len(search_results)
            
            return {
                "response": response,
                "sources": sources,
                "confidence": float(avg_confidence),
                "num_sources": len(sources)
            }
            
        except Exception as e:
            error_message = f"Error in RAG pipeline: {str(e)}"
            st.error(error_message)
            return {
                "response": error_message,
                "sources": [],
                "confidence": 0.0
            }
    
    def evaluate_query_relevance(self, query: str) -> float:
        """
        Evaluate how well the query can be answered with available documents
        
        Args:
            query: User query
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Get top results
            results = self.vector_store.search(query, k=3)
            
            if not results:
                return 0.0
            
            # Calculate average similarity score
            avg_score = sum(score for _, score in results) / len(results)
            
            # Normalize to 0-1 range (assuming similarity scores are between -1 and 1)
            normalized_score = (avg_score + 1) / 2
            
            return min(max(normalized_score, 0.0), 1.0)
            
        except Exception as e:
            st.error(f"Error evaluating query relevance: {str(e)}")
            return 0.0
    
    def get_similar_questions(self, query: str, num_questions: int = 3) -> List[str]:
        """
        Generate similar questions based on the context
        
        Args:
            query: Original query
            num_questions: Number of similar questions to generate
            
        Returns:
            List of similar questions
        """
        try:
            # Get relevant context
            context = self.retrieve_context(query, k=3)
            
            if not context:
                return []
            
            context_text = "\n".join(context[:2])  # Use top 2 contexts
            
            prompt = f"""Based on the following context, suggest {num_questions} related questions that could be asked:

Context:
{context_text}

Original question: {query}

Please provide {num_questions} similar or related questions that could be answered using the same context. Format each question on a new line starting with a number."""

            response = self.model.generate_content(prompt)
            
            if response.text:
                # Extract questions from response
                lines = response.text.strip().split('\n')
                questions = []
                
                for line in lines:
                    line = line.strip()
                    # Remove numbering and clean up
                    if line and any(char.isalpha() for char in line):
                        # Remove common numbering patterns
                        line = line.lstrip('0123456789.-)( ')
                        if line and line.endswith('?'):
                            questions.append(line)
                        elif line:
                            questions.append(line + '?')
                
                return questions[:num_questions]
            
            return []
            
        except Exception as e:
            st.error(f"Error generating similar questions: {str(e)}")
            return []
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "model_name": self.model_name,
            "vector_store_stats": vector_stats,
            "total_documents": vector_stats.get("total_documents", 0),
            "embedding_dimension": vector_stats.get("embedding_dimension", 0)
        }