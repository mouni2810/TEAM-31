"""
RAG Pipeline for GovInsight

This module orchestrates the complete RAG pipeline:
- PDF loading and text extraction
- Chunking with manual metadata (no automatic extraction)
- Embedding generation
- Vector store indexing
- Semantic query and retrieval

NOTE: All metadata is manually configured in metadata_config.py
to ensure consistent and complete metadata for every chunk.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from app.pdf_processor import PDFProcessor, chunk_text_with_metadata
from app.metadata_config import get_metadata_for_document, validate_metadata
from app.utils import (
    initialize_vector_store,
    initialize_embeddings,
    embed_chunks,
    format_metadata_for_storage
)


def normalize_year_filter(year: Optional[str]) -> Optional[str]:
    """
    Normalize year format to match metadata format (YYYY-YY).
    
    Args:
        year: Year string in various formats (2024, 2024-25, etc.)
        
    Returns:
        Normalized year string (YYYY-YY format) or original if already valid
    """
    if not year:
        return None
    
    # If already in YYYY-YY format, return as-is
    if re.match(r'^\d{4}-\d{2}$', year):
        return year
    
    # If in YYYY format, convert to YYYY-YY
    if re.match(r'^\d{4}$', year):
        year_num = int(year)
        next_year = str(year_num + 1)[-2:]
        normalized = f"{year}-{next_year}"
        print(f"  Normalized year filter: '{year}' → '{normalized}'")
        return normalized
    
    # If in YYYY-YYYY format, convert to YYYY-YY
    match = re.match(r'^(\d{4})-(\d{4})$', year)
    if match:
        start_year = match.group(1)
        end_year = match.group(2)[-2:]
        normalized = f"{start_year}-{end_year}"
        print(f"  Normalized year filter: '{year}' → '{normalized}'")
        return normalized
    
    # Otherwise return as-is and warn
    print(f"  ⚠️  Warning: Unusual year format '{year}', using as-is")
    return year


def preprocess_query(query: str, year_filter: Optional[str] = None) -> str:
    """
    Preprocess query to improve semantic matching.
    
    Expands abbreviations and adds context for better retrieval.
    
    Args:
        query: Original user query
        year_filter: Optional year filter for context
        
    Returns:
        Preprocessed query string
    """
    # Expand common abbreviations
    expansions = {
        r'\bedu\b': 'education',
        r'\bdef\b': 'defence defense',
        r'\bfin\b': 'finance financial',
        r'\bMoRTH\b': 'Ministry of Road Transport Highways',
        r'\bNITI\b': 'NITI Aayog',
        r'\bWCD\b': 'Women Child Development',
        r'\bMoHFW\b': 'Ministry of Health Family Welfare',
        r'\balloc\b': 'allocation allocated',
        r'\bbudg\b': 'budget',
    }
    
    processed = query
    for pattern, expansion in expansions.items():
        processed = re.sub(pattern, expansion, processed, flags=re.IGNORECASE)
    
    # Add year context if provided
    if year_filter:
        processed = f"{processed} fiscal year {year_filter}"
    
    # If query changed, log it
    if processed != query:
        print(f"  Query preprocessing: '{query}' → '{processed}'")
    
    return processed



# System prompt template for Gemini
SYSTEM_PROMPT_TEMPLATE = """You are GovInsight, an AI assistant specialized in analyzing Indian Union Budget documents. Your role is to provide accurate, citation-backed answers about government budget allocations, schemes, and expenditure.

**CRITICAL RULES - STRICTLY ENFORCE:**

1. **ONLY USE PROVIDED CONTEXT**: You MUST base your answers EXCLUSIVELY on the retrieved document chunks provided below. 
   - DO NOT use any external knowledge, general knowledge, or information not explicitly in the provided chunks.
   - DO NOT make assumptions or inferences beyond what is directly stated in the context.
   - If information is not in the provided chunks, you MUST say so explicitly.

2. **MANDATORY CITATIONS**: For EVERY claim, figure, number, or statement you make, you MUST cite the source using this exact format:
   - [Year: {{year}}, Ministry: {{ministry}}, Page: {{page_number}}]
   - Multiple citations should be separated by semicolons: [Year: 2024-25, Ministry: Finance, Page: 10]; [Year: 2023-24, Ministry: Finance, Page: 15]
   
3. **ZERO TOLERANCE FOR HALLUCINATION**: 
   - If the provided context does not contain the information needed to answer the query, you MUST state: "The requested information is not available in the provided budget documents."
   - DO NOT fabricate, estimate, approximate, or guess any numbers, dates, names, or details.
   - DO NOT combine information from different sources unless explicitly stated in the context.
   - If you're uncertain, state your uncertainty clearly.

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



def rerank_chunks_by_content_quality(
    chunks: List[Dict],
    query: str,
    top_k: int = 7
) -> List[Dict]:
    """
    Rerank retrieved chunks based on content quality metrics.
    
    Prioritizes chunks that contain more informative content by considering:
    - Token count (longer chunks may have more context)
    - Number density (budget docs with numbers are more informative)
    - Semantic distance (original retrieval score)
    
    The final score is a weighted combination:
    - 60% semantic similarity (inverted distance)
    - 25% number density (for budget-relevant content)
    - 15% token count (normalized)
    
    Args:
        chunks: List of retrieved chunks with metadata
        query: Original user query (for potential query-aware reranking)
        top_k: Number of chunks to return after reranking
        
    Returns:
        Reranked list of chunks (best first)
    """
    if not chunks:
        return []
    
    # Calculate normalization factors
    max_token_count = max(c.get('metadata', {}).get('token_count', 1) or 1 for c in chunks)
    max_number_density = max(c.get('metadata', {}).get('number_density', 0.1) or 0.1 for c in chunks)
    max_distance = max(c.get('distance', 1.0) or 1.0 for c in chunks)
    
    # Check if query is asking for numerical/budget information
    numerical_query_keywords = [
        'allocation', 'budget', 'crore', 'lakh', 'expenditure', 'amount',
        'how much', 'total', 'percentage', '%', 'growth', 'increase', 
        'decrease', 'compare', 'trend', 'fiscal', 'spending', 'cost'
    ]
    query_lower = query.lower()
    is_numerical_query = any(kw in query_lower for kw in numerical_query_keywords)
    
    # Adjust weights based on query type
    if is_numerical_query:
        # For numerical queries, heavily weight number density
        weight_semantic = 0.45
        weight_numbers = 0.40
        weight_tokens = 0.15
        print("  Reranking: Numerical query detected, prioritizing data-rich chunks")
    else:
        # For general queries, balance semantic and content
        weight_semantic = 0.60
        weight_numbers = 0.25
        weight_tokens = 0.15
    
    # Score each chunk
    scored_chunks = []
    for chunk in chunks:
        metadata = chunk.get('metadata', {}) or {}
        distance = chunk.get('distance', 1.0) or 1.0
        
        # Semantic score (inverted and normalized distance - lower distance = higher score)
        semantic_score = 1.0 - (distance / max_distance) if max_distance > 0 else 0.5
        
        # Token count score (normalized)
        token_count = metadata.get('token_count', 0) or 0
        token_score = token_count / max_token_count if max_token_count > 0 else 0.5
        
        # Number density score (normalized)
        number_density = metadata.get('number_density', 0.0) or 0.0
        number_score = number_density / max_number_density if max_number_density > 0 else 0.0
        
        # Bonus for chunks that have numbers (especially for budget queries)
        has_numbers = metadata.get('has_numbers', False)
        number_bonus = 0.05 if has_numbers and is_numerical_query else 0.0
        
        # Calculate weighted final score
        final_score = (
            (weight_semantic * semantic_score) +
            (weight_numbers * number_score) +
            (weight_tokens * token_score) +
            number_bonus
        )
        
        scored_chunks.append({
            **chunk,
            'rerank_score': round(final_score, 4),
            'score_breakdown': {
                'semantic': round(semantic_score, 3),
                'number_density': round(number_score, 3),
                'token_count': round(token_score, 3)
            }
        })
    
    # Sort by final score (descending)
    scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
    
    # Log reranking results
    print(f"  Reranking complete. Top chunk scores:")
    for i, chunk in enumerate(scored_chunks[:3]):
        meta = chunk.get('metadata', {})
        print(f"    [{i+1}] Score: {chunk['rerank_score']:.3f} | "
              f"Tokens: {meta.get('token_count', 0)} | "
              f"NumDensity: {meta.get('number_density', 0):.1f}% | "
              f"Distance: {chunk.get('distance', 0):.3f}")
    
    return scored_chunks[:top_k]


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
        document_metadata: Optional[Dict[str, Dict]] = None,
        reset_vectorstore: bool = False
    ) -> None:
        """
        Index all PDF documents into the vector store.
        
        Metadata is obtained from:
        1. The document_metadata parameter (if provided) - dictionary keyed by filename
        2. The metadata_config.py configuration file
        3. Default values to ensure complete metadata schema
        
        Args:
            document_metadata: Optional dictionary mapping filenames to metadata dicts.
                               If not provided, uses metadata_config.py configuration.
            reset_vectorstore: If True, clear existing vector store before indexing
        """
        print("=" * 60)
        print("Starting GovInsight Document Indexing")
        print("=" * 60)
        print("\n📋 NOTE: Using manual metadata configuration (no auto-extraction)")
        print("   Configure metadata in app/metadata_config.py\n")
        
        # Step 1: Initialize PDF processor
        print("[1/5] Initializing PDF processor...")
        self.pdf_processor = PDFProcessor(pdf_directory=self.pdf_directory)
        pdf_files = self.pdf_processor.load_pdfs()
        print(f"Found {len(pdf_files)} PDFs to index")
        
        # Step 2: Extract text from all PDFs
        print(f"\n[2/5] Extracting text from {len(pdf_files)} PDFs...")
        all_chunks = []
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"  Processing ({idx}/{len(pdf_files)}): {pdf_path.name}")
            
            # Extract text page by page
            pages_data = self.pdf_processor.extract_text_from_pdf(pdf_path)
            
            # Get metadata for this PDF from configuration
            # Priority: document_metadata param > metadata_config.py > defaults
            if document_metadata and pdf_path.name in document_metadata:
                metadata = validate_metadata(document_metadata[pdf_path.name])
                print(f"    ✓ Using provided metadata")
            else:
                metadata = get_metadata_for_document(pdf_path.name)
            
            # Log the metadata being used
            print(f"    → Year: {metadata.get('year', 'Unknown')}, "
                  f"Ministry: {metadata.get('ministry', 'Unknown')}, "
                  f"Scheme: {metadata.get('scheme', 'General')}")
            
            # Chunk the extracted text with complete metadata
            chunks = chunk_text_with_metadata(
                pages_data=pages_data,
                metadata=metadata,
                chunk_size=1000,
                overlap=200
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
        ids = [chunk['id'] for chunk in all_chunks]
        
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
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
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
        top_k: int = 7,
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
        
        # Preprocess query for better matching
        processed_query = preprocess_query(query, year_filter)
        
        # Generate query embedding
        print(f"Generating query embedding for: '{processed_query}'")
        query_embedding = self.embedding_generator.get_query_embedding(processed_query)
        
        # Build metadata filter (where clause) with normalization
        where_filter = {}
        
        if year_filter:
            # Normalize year format (e.g., "2024" → "2024-25")
            normalized_year = normalize_year_filter(year_filter)
            if normalized_year:
                where_filter["year"] = normalized_year
        
        if ministry_filter:
            where_filter["ministry"] = ministry_filter
        
        if scheme_filter:
            # Handle None scheme case
            if scheme_filter.lower() == "none":
                where_filter["scheme"] = "None"
            else:
                where_filter["scheme"] = scheme_filter
        
        # Retrieve more candidates for reranking (2x top_k)
        retrieval_count = top_k * 2
        print(f"Retrieving top-{retrieval_count} candidates for reranking...")
        
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": retrieval_count
        }
        
        # Add where filter if any filters are specified
        if where_filter:
            query_params["where"] = where_filter
            print(f"Applying filters: {where_filter}")
        
        results = self.vector_store.collection.query(**query_params)
        
        
        # Format results with relevance filtering
        retrieved_chunks = []
        
        # CRITICAL: Filter by relevance threshold to exclude irrelevant chunks
        # Lower distance = more similar (cosine distance)
        RELEVANCE_THRESHOLD = 1.2  # Only include chunks with distance <= 1.2
        
        if results and results['documents'] and results['documents'][0]:
            total_retrieved = len(results['documents'][0])
            
            for i in range(total_retrieved):
                # Ensure metadata is always a dict, never None
                metadata = results['metadatas'][0][i] if (results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0])) else {}
                if metadata is None:
                    metadata = {}
                
                distance = results['distances'][0][i] if (results.get('distances') and results['distances'][0] and i < len(results['distances'][0])) else 0.0
                
                # Filter by relevance threshold
                if distance <= RELEVANCE_THRESHOLD:
                    chunk = {
                        'text': results['documents'][0][i] if results['documents'][0][i] else '',
                        'metadata': metadata,
                        'distance': distance
                    }
                    retrieved_chunks.append(chunk)
                else:
                    print(f"  Filtered out chunk {i+1} (distance: {distance:.3f} > threshold: {RELEVANCE_THRESHOLD})")
             
            print(f"Retrieved {len(retrieved_chunks)} relevant chunks (from {total_retrieved} total)")
        else:
            print("No results returned from vector store query")
        
        if not retrieved_chunks:
            print("No relevant chunks found.")
            return []
        
        # ---------------------------------------------------------
        # Rerank chunks by content quality (token count, number density)
        # ---------------------------------------------------------
        print("Reranking chunks by content quality...")
        retrieved_chunks = rerank_chunks_by_content_quality(
            chunks=retrieved_chunks,
            query=query,
            top_k=top_k
        )

        # ---------------------------------------------------------
        # Context Expansion: Fetch Neighbor Chunks
        # ---------------------------------------------------------
        print("Expanding context with neighbors...")
        
        # Collect all neighbor IDs to fetch in batch
        neighbor_ids_to_fetch = set()
        chunk_map = {} # map id -> chunk data
        
        for chunk in retrieved_chunks:
            metadata = chunk['metadata']
            doc_name = metadata.get('document_name')
            chunk_index = metadata.get('chunk_index')
            
            # If we have index info, look for neighbors
            if doc_name and chunk_index is not None:
                # Get surrounding +/- 1 chunk
                prev_id = f"{doc_name}_chunk_{chunk_index - 1}"
                next_id = f"{doc_name}_chunk_{chunk_index + 1}"
                
                neighbor_ids_to_fetch.add(prev_id)
                neighbor_ids_to_fetch.add(next_id)
        
        if neighbor_ids_to_fetch:
            try:
                # Batch fetch from Chroma
                neighbor_results = self.vector_store.collection.get(
                    ids=list(neighbor_ids_to_fetch)
                )
                
                if neighbor_results and neighbor_results['ids']:
                    for i, nid in enumerate(neighbor_results['ids']):
                        if neighbor_results['documents'][i]:
                            chunk_map[nid] = {
                                'text': neighbor_results['documents'][i],
                                'metadata': neighbor_results['metadatas'][i],
                                'is_neighbor': True
                            }
            except Exception as e:
                print(f"Error fetching neighbors: {e}")
        
        # Now merge neighbors into retrieved chunks
        final_chunks = []
        
        for chunk in retrieved_chunks:
            meta = chunk['metadata']
            doc_name = meta.get('document_name')
            c_idx = meta.get('chunk_index')
            
            if doc_name is not None and c_idx is not None:
                # Check for Previous Neighbor
                prev_id = f"{doc_name}_chunk_{c_idx - 1}"
                if prev_id in chunk_map:
                    prev_chunk = chunk_map[prev_id]
                    chunk['text'] = f"{prev_chunk['text']}\n\n{chunk['text']}"
                
                # Check for Next Neighbor
                next_id = f"{doc_name}_chunk_{c_idx + 1}"
                if next_id in chunk_map:
                    next_chunk = chunk_map[next_id]
                    chunk['text'] = f"{chunk['text']}\n\n{next_chunk['text']}"
            
            final_chunks.append(chunk)

        print(f"Retrieved {len(final_chunks)} chunks after context expansion")
        
        return final_chunks
    
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
        # Handle empty chunks
        if not retrieved_chunks or len(retrieved_chunks) == 0:
            return {
                'answer': "No relevant documents were found matching your query and filters. Please try:\n\n1. Removing or adjusting the year/ministry/scheme filters\n2. Using a broader search query\n3. Checking if documents have been indexed with the correct metadata",
                'sources': [],
                'num_chunks_used': 0
            }
        
        import google.generativeai as genai
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Initialize model - using gemini-2.5-flash
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Format context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            text = chunk.get('text', '')
            # Ensure metadata is always a dictionary, never None
            metadata = chunk.get('metadata', {})
            if metadata is None:
                metadata = {}
            
            # Format chunk with metadata - all fields now guaranteed to exist
            chunk_text = f"[Chunk {i}]\n"
            chunk_text += f"Page: {metadata.get('page_number', 0)}\n"
            chunk_text += f"Year: {metadata.get('year', 'Unknown')}\n"
            chunk_text += f"Ministry: {metadata.get('ministry', 'Unknown')}\n"
            
            scheme = metadata.get('scheme', 'General')
            if scheme and scheme != 'General' and scheme != 'None':
                chunk_text += f"Scheme: {scheme}\n"
            
            budget_category = metadata.get('budget_category', 'General')
            if budget_category and budget_category != 'General':
                chunk_text += f"Category: {budget_category}\n"
            
            chunk_text += f"\nContent:\n{text}\n"
            
            context_parts.append(chunk_text)
            
            # Track sources - ensure page_number is an int
            page_num = metadata.get('page_number', 0)
            if not isinstance(page_num, int):
                try:
                    page_num = int(page_num) if page_num else 0
                except (ValueError, TypeError):
                    page_num = 0
            
            sources.append({
                'page_number': page_num,
                'year': metadata.get('year', 'Unknown'),
                'ministry': metadata.get('ministry', 'Unknown')
            })
        
        # Properly format context with separators between chunks
        separator = "\n" + "="*60 + "\n\n"
        context = separator + separator.join(context_parts)
        
        # Format the prompt
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )
        
        print("Generating answer using Gemini-2.5-Flash...")
        
        # Generate response with improved parameters
        generation_config = genai.types.GenerationConfig(
            temperature=max(0.3, temperature),  # Minimum 0.3 for better reasoning
            top_p=0.95,
            top_k=40,
            max_output_tokens=4096,  # Increased for longer, more complete answers
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
    top_k: int = 7,
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
    top_k: int = 7,
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
    
    # Check if any chunks were retrieved
    if not retrieved_chunks or len(retrieved_chunks) == 0:
        # Build filter message
        filter_parts = []
        if year:
            filter_parts.append(f"year: {year}")
        if ministry:
            filter_parts.append(f"ministry: {ministry}")
        if scheme:
            filter_parts.append(f"scheme: {scheme}")
        
        filter_msg = f" with filters ({', '.join(filter_parts)})" if filter_parts else ""
        
        return {
            'answer': f"No documents were found matching your query{filter_msg}. This could mean:\n\n1. **No documents match the specified filters** - Try removing or adjusting the year/ministry/scheme filters\n2. **Documents may not be indexed** - Ensure PDFs have been processed with: `python app/rag_pipeline.py --index`\n3. **Metadata mismatch** - The documents may have different year/ministry values than the filter\n\n**Suggestions:**\n- Remove the year filter to search across all years\n- Try a broader query without specific filters\n- Check if documents exist for the selected year",
            'sources': [],
            'num_chunks_used': 0
        }
    
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

