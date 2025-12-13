"""
FastAPI Application for GovInsight

Provides REST API endpoints for querying Indian budget documents.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_pipeline import complete_query


# Initialize FastAPI app
app = FastAPI(
    title="GovInsight API",
    description="RAG-based API for querying Indian Union Budget documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class QueryFilters(BaseModel):
    """Optional filters for query"""
    year: Optional[str] = Field(None, description="Budget year (e.g., '2023-24')")
    ministry: Optional[str] = Field(None, description="Ministry name")
    scheme: Optional[str] = Field(None, description="Scheme name")
    top_k: Optional[int] = Field(5, ge=1, le=10, description="Number of chunks to retrieve")


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: str = Field(..., min_length=1, description="User query about budget documents")
    filters: Optional[QueryFilters] = Field(None, description="Optional metadata filters")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="LLM temperature")


class SourceDocument(BaseModel):
    """Source document metadata"""
    document_title: str
    page_number: int
    year: str
    ministry: str


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="Generated answer with citations")
    sources: List[SourceDocument] = Field(..., description="Source documents used")
    num_chunks_used: int = Field(..., description="Number of chunks retrieved")
    query: str = Field(..., description="Original query")


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI"""
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to API info if HTML not found
        return """
        <html>
            <body>
                <h1>GovInsight API</h1>
                <p>Frontend not found. Please check if index.html exists.</p>
                <p>API Documentation: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "name": "GovInsight API",
        "version": "1.0.0",
        "description": "RAG system for Indian Union Budget documents",
        "endpoints": {
            "/query": "POST - Query budget documents",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "GovInsight"
    }


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query Indian budget documents using RAG.
    
    Args:
        request: QueryRequest with query string and optional filters
        
    Returns:
        QueryResponse with answer, sources, and metadata
    """
    try:
        # Extract filters
        filters = request.filters or QueryFilters()
        
        # Execute RAG query
        result = complete_query(
            query=request.query,
            top_k=filters.top_k or 5,
            year=filters.year,
            ministry=filters.ministry,
            scheme=filters.scheme,
            temperature=request.temperature
        )
        
        # Format sources
        sources = [
            SourceDocument(
                document_title=source['document_title'],
                page_number=source['page_number'],
                year=source['year'],
                ministry=source['ministry']
            )
            for source in result['sources']
        ]
        
        # Return response
        return QueryResponse(
            answer=result['answer'],
            sources=sources,
            num_chunks_used=result['num_chunks_used'],
            query=request.query
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store not found. Please run indexing first: {str(e)}"
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Run with: uvicorn app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

