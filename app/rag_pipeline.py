"""
RAG Pipeline for GovInsight

This module orchestrates the complete RAG pipeline:
- PDF loading and text extraction
- Chunking with metadata enrichment
- Embedding generation
- Vector store indexing
- Semantic query and retrieval
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app.pdf_processor import PDFProcessor, chunk_text_with_metadata
    from app.utils import (
        initialize_vector_store,
        initialize_embeddings,
        embed_chunks,
        format_metadata_for_storage
    )
except ModuleNotFoundError:
    from pdf_processor import PDFProcessor, chunk_text_with_metadata
    from utils import (
        initialize_vector_store,
        initialize_embeddings,
        embed_chunks,
        format_metadata_for_storage
    )


# System prompt template for Gemini-2.5-Flash
SYSTEM_PROMPT_TEMPLATE = """You are GovInsight, an AI assistant specialized in analyzing Indian Union Budget documents. Your role is to provide accurate, citation-backed answers about government budget allocations, schemes, and expenditure.

**Critical Instructions:**

1. **Use Only Provided Context**: Base your answers EXCLUSIVELY on the retrieved document chunks provided below. DO NOT use external knowledge or make assumptions.

2. **Mandatory Citations**: For EVERY claim or figure you mention, cite the source using this format:
   - [Document: {document_title}, Page: {page_number}]
   
3. **No Hallucination**: If the provided context does not contain the information needed to answer the query:
   - Explicitly state: "Data not found in the provided budget documents."
   - Do NOT fabricate numbers, dates, or details.
   - You may suggest related information if available in the context.

4. **Structured Responses**:
   - Use clear, concise language
   - When presenting financial data, use Markdown tables with proper formatting
   - Include year-wise breakdowns when applicable
   - Separate different schemes or ministries clearly

5. **Table Format Example**:
   | Year | Allocation (₹ Crore) | Scheme | Ministry |
   |------|---------------------|--------|----------|
   | 2023-24 | 10,000 | Example | MoRTH |

6. **Handle Missing Data Gracefully**:
   - If data for specific years is missing, acknowledge it
   - If a scheme name has changed, mention both old and new names if available in context

**Retrieved Context:**

{context}

**User Query:**

{query}

**Your Response:**
"""



class RAGPipeline:
    """
    Complete RAG pipeline for indexing budget documents.
    """
    
    def __init__(
        self,
        pdf_directory: str = "data/raw_pdfs",
        vectorstore_directory: str = "vectorstore",
        embedding_model_type: str = "huggingface"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            pdf_directory: Directory containing PDF files
            vectorstore_directory: Directory for vector store persistence
            embedding_model_type: Type of embedding model ("huggingface")
        """
        self.pdf_directory = pdf_directory
        self.vectorstore_directory = vectorstore_directory
        self.embedding_model_type = embedding_model_type
        
        # Initialize components
        self.pdf_processor = None
        self.vector_store = None
        self.embedding_generator = None
    
    def index_documents(
        self,
        document_metadata: Optional[List[Dict]] = None,
        reset_vectorstore: bool = False
    ) -> None:
        """
        Index all PDF documents into the vector store.
        
        Args:
            document_metadata: Optional list of metadata dicts for each PDF
                               If not provided, extracts from filenames
            reset_vectorstore: If True, clear existing vector store before indexing
        """
        print("=" * 60)
        print("Starting GovInsight Document Indexing")
        print("=" * 60)
        
        # Step 1: Initialize PDF processor
        print("\n[1/5] Initializing PDF processor...")
        self.pdf_processor = PDFProcessor(pdf_directory=self.pdf_directory)
        pdf_files = self.pdf_processor.load_pdfs()
        print(f"Indexing {len(pdf_files)} PDFs...")
        
        # Step 2: Extract text from all PDFs
        print(f"\n[2/5] Extracting text from {len(pdf_files)} PDFs...")
        all_chunks = []
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"  Processing ({idx}/{len(pdf_files)}): {pdf_path.name}")
            
            # Extract text page by page
            pages_data = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Get or extract metadata for this PDF
            if document_metadata and idx - 1 < len(document_metadata):
                metadata = document_metadata[idx - 1]
            else:
                metadata = self.pdf_processor.extract_metadata_from_filename(pdf_path.name)
            
            # Extract metadata fields
            year = metadata.get('year', 'Unknown')
            ministry = metadata.get('ministry', 'Unknown')
            document_title = pdf_path.stem
            scheme = metadata.get('scheme', None)
            
            # Chunk the extracted text with metadata
            chunks = chunk_text_with_metadata(
                pages_data=pages_data,
                year=year,
                ministry=ministry,
                document_title=document_title,
                scheme=scheme,
                chunk_size=400,
                overlap=50
            )
            
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks created")
        
        print(f"\nExtracted {len(all_chunks)} chunks from {len(pdf_files)} documents")
        
        if not all_chunks:
            print("No chunks extracted. Exiting.")
            return
        
        # Step 3: Generate embeddings
        print(f"\n[3/5] Generating embeddings using {self.embedding_model_type}...")
        self.embedding_generator = initialize_embeddings(
            model_type=self.embedding_model_type
        )
        
        embeddings = embed_chunks(all_chunks, self.embedding_generator)
        print("Generated embeddings")
        
        # Step 4: Initialize vector store
        print(f"\n[4/5] Initializing ChromaDB vector store...")
        self.vector_store = initialize_vector_store(
            persist_directory=self.vectorstore_directory,
            reset=reset_vectorstore
        )
        
        # Step 5: Store in ChromaDB
        print(f"\n[5/5] Storing chunks in ChromaDB...")
        
        # Prepare data for storage
        documents = [chunk['text'] for chunk in all_chunks]
        metadatas = [format_metadata_for_storage(chunk) for chunk in all_chunks]
        
        # Add to vector store in batches
        batch_size = 100
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            batch_num = (i // batch_size) + 1
            
            print(f"  Batch {batch_num}/{total_batches}: Adding chunks {i+1}-{batch_end}...")
            
            self.vector_store.add_documents(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
        
        print("Stored in ChromaDB")
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("Indexing complete.")
        print("=" * 60)
        stats = self.vector_store.get_collection_stats()
        print(f"Total documents in vector store: {stats['total_documents']}")
        print(f"Collection name: {stats['collection_name']}")
        print(f"Persist directory: {stats['persist_directory']}")
        print("=" * 60)
    
    def query_documents(
        self,
        query: str,
        top_k: int = 5,
        year_filter: Optional[str] = None,
        ministry_filter: Optional[str] = None,
        scheme_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Query the vector store for relevant documents.
        
        Args:
            query: User query string
            top_k: Number of results to retrieve (default: 5)
            year_filter: Optional filter by year (e.g., "2023-24")
            ministry_filter: Optional filter by ministry name
            scheme_filter: Optional filter by scheme name
            
        Returns:
            List of dictionaries containing retrieved chunks and metadata:
            [
                {
                    'text': str,
                    'metadata': dict,
                    'distance': float
                }
            ]
        """
        # Initialize components if not already done
        if self.vector_store is None:
            print("Loading vector store...")
            self.vector_store = initialize_vector_store(
                persist_directory=self.vectorstore_directory,
                reset=False
            )
        
        if self.embedding_generator is None:
            print("Initializing embedding generator...")
            self.embedding_generator = initialize_embeddings(
                model_type=self.embedding_model_type
            )
        
        # Generate query embedding
        print(f"Generating query embedding for: '{query}'")
        query_embedding = self.embedding_generator.get_query_embedding(query)
        
        # Build metadata filter (where clause)
        where_filter = {}
        
        if year_filter:
            where_filter["year"] = year_filter
        
        if ministry_filter:
            where_filter["ministry"] = ministry_filter
        
        if scheme_filter:
            # Handle None scheme case
            if scheme_filter.lower() == "none":
                where_filter["scheme"] = "None"
            else:
                where_filter["scheme"] = scheme_filter
        
        # Query the vector store
        print(f"Retrieving top-{top_k} chunks from vector store...")
        
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": top_k
        }
        
        # Add where filter if any filters are specified
        if where_filter:
            query_params["where"] = where_filter
            print(f"Applying filters: {where_filter}")
        
        results = self.vector_store.collection.query(**query_params)
        
        # Format results
        retrieved_chunks = []
        
        if results and results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                chunk = {
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0
                }
                retrieved_chunks.append(chunk)
        
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        
        return retrieved_chunks
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Dict],
        temperature: float = 0.1
    ) -> Dict:
        """
        Generate an answer using Gemini-2.5-Flash based on retrieved chunks.
        
        Args:
            query: User query string
            retrieved_chunks: List of retrieved chunks from query_documents()
            temperature: Temperature for generation (default: 0.1 for deterministic)
            
        Returns:
            Dictionary containing:
            {
                'answer': str,
                'sources': List[Dict],
                'num_chunks_used': int
            }
        """
        import google.generativeai as genai
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Format context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk['text']
            metadata = chunk['metadata']
            
            # Format chunk with metadata
            chunk_text = f"[Chunk {i}]\n"
            chunk_text += f"Document: {metadata.get('document_title', 'Unknown')}\n"
            chunk_text += f"Page: {metadata.get('page_number', 'N/A')}\n"
            chunk_text += f"Year: {metadata.get('year', 'Unknown')}\n"
            chunk_text += f"Ministry: {metadata.get('ministry', 'Unknown')}\n"
            
            scheme = metadata.get('scheme', 'None')
            if scheme and scheme != 'None':
                chunk_text += f"Scheme: {scheme}\n"
            
            chunk_text += f"\nContent:\n{text}\n"
            
            context_parts.append(chunk_text)
            
            # Track sources
            sources.append({
                'document_title': metadata.get('document_title', 'Unknown'),
                'page_number': metadata.get('page_number', 'N/A'),
                'year': metadata.get('year', 'Unknown'),
                'ministry': metadata.get('ministry', 'Unknown')
            })
        
        context = "\n" + "="*60 + "\n\n".join(context_parts)
        
        # Format the prompt
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        
        print("Generating answer using Gemini-2.5-Flash...")
        
        # Generate response
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=2048,
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        answer = response.text
        
        print("Answer generated successfully")
        
        return {
            'answer': answer,
            'sources': sources,
            'num_chunks_used': len(retrieved_chunks)
        }




def index_pdfs_cli(
    pdf_directory: str = "data/raw_pdfs",
    vectorstore_directory: str = "vectorstore",
    embedding_model: str = "huggingface",
    reset: bool = False
) -> None:
    """
    CLI interface for indexing PDFs.
    
    Args:
        pdf_directory: Directory containing PDF files
        vectorstore_directory: Directory for vector store persistence
        embedding_model: Type of embedding model to use
        reset: Whether to reset the vector store before indexing
    """
    pipeline = RAGPipeline(
        pdf_directory=pdf_directory,
        vectorstore_directory=vectorstore_directory,
        embedding_model_type=embedding_model
    )
    
    pipeline.index_documents(reset_vectorstore=reset)


def query_rag(
    query: str,
    top_k: int = 5,
    year: Optional[str] = None,
    ministry: Optional[str] = None,
    scheme: Optional[str] = None,
    vectorstore_directory: str = "vectorstore",
    embedding_model: str = "huggingface"
) -> List[Dict]:
    """
    Query the RAG system for relevant budget information.
    
    Args:
        query: User query string
        top_k: Number of results to retrieve (default: 5)
        year: Optional filter by year
        ministry: Optional filter by ministry
        scheme: Optional filter by scheme
        vectorstore_directory: Directory containing vector store
        embedding_model: Embedding model type
        
    Returns:
        List of retrieved chunks with metadata
    """
    pipeline = RAGPipeline(
        vectorstore_directory=vectorstore_directory,
        embedding_model_type=embedding_model
    )
    
    results = pipeline.query_documents(
        query=query,
        top_k=top_k,
        year_filter=year,
        ministry_filter=ministry,
        scheme_filter=scheme
    )
    
    return results


def complete_query(
    query: str,
    top_k: int = 5,
    year: Optional[str] = None,
    ministry: Optional[str] = None,
    scheme: Optional[str] = None,
    vectorstore_directory: str = "vectorstore",
    embedding_model: str = "huggingface",
    temperature: float = 0.1
) -> Dict:
    """
    Complete RAG query: retrieve chunks and generate answer.
    
    Args:
        query: User query string
        top_k: Number of chunks to retrieve
        year: Optional filter by year
        ministry: Optional filter by ministry
        scheme: Optional filter by scheme
        vectorstore_directory: Vector store directory
        embedding_model: Embedding model type
        temperature: LLM temperature (default: 0.1 for deterministic)
        
    Returns:
        Dictionary with answer, sources, and metadata
    """
    pipeline = RAGPipeline(
        vectorstore_directory=vectorstore_directory,
        embedding_model_type=embedding_model
    )
    
    # Retrieve relevant chunks
    retrieved_chunks = pipeline.query_documents(
        query=query,
        top_k=top_k,
        year_filter=year,
        ministry_filter=ministry,
        scheme_filter=scheme
    )
    
    # Generate answer
    result = pipeline.generate_answer(
        query=query,
        retrieved_chunks=retrieved_chunks,
        temperature=temperature
    )
    
    return result




def main():
    """
    Main entry point for CLI usage.
    """
    parser = argparse.ArgumentParser(
        description="GovInsight RAG Pipeline - Index budget documents"
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Run indexing pipeline to process PDFs and build vector store"
    )
    
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="data/raw_pdfs",
        help="Directory containing PDF files (default: data/raw_pdfs)"
    )
    
    parser.add_argument(
        "--vectorstore-dir",
        type=str,
        default="vectorstore",
        help="Directory for vector store persistence (default: vectorstore)"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="huggingface",
        help="Embedding model type (default: huggingface)"
    )
    
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset vector store before indexing (WARNING: deletes existing data)"
    )
    
    args = parser.parse_args()
    
    if args.index:
        index_pdfs_cli(
            pdf_directory=args.pdf_dir,
            vectorstore_directory=args.vectorstore_dir,
            embedding_model=args.embedding_model,
            reset=args.reset
        )
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python app/rag_pipeline.py --index")
        print("  python app/rag_pipeline.py --index --reset")
        print("  python app/rag_pipeline.py --index --embedding-model huggingface")


if __name__ == "__main__":
    main()

