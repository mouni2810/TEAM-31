# GovInsight – Indian Government Budget RAG System

## Project Overview

GovInsight is a Retrieval-Augmented Generation (RAG) system designed to enable semantic querying and analysis of Indian Union Budget documents. The system addresses the complexity of manually analyzing hundreds of pages of budget PDFs spread across multiple years, ministries, and schemes.

### Problem Statement

Indian government budget documents contain critical information about public expenditure, infrastructure investments, and policy priorities. However, extracting insights from these documents is challenging due to:

- **Volume and Fragmentation**: Budget documents span multiple PDFs across different ministries and years
- **Structural Inconsistency**: Scheme names, categorizations, and formats change across budget cycles
- **Limited Searchability**: Traditional keyword search fails to understand semantic queries or synthesize cross-document patterns
- **Manual Overhead**: Analysts spend significant time locating and aggregating data across documents

### Why RAG?

While Google or keyword search can locate specific terms, they cannot:
- Synthesize multi-year trends from fragmented data
- Answer comparative queries across schemes and ministries
- Provide contextual, citation-backed answers
- Handle semantic variations in terminology

RAG combines vector-based semantic search with large language model reasoning to enable natural language querying over structured budget data.

### Scope

This system focuses on:
- **Document Type**: Union Budget of India (federal/central government)
- **Focus Areas**: Infrastructure, roads, transport, capital expenditure
- **Analysis Dimensions**: Sector-wise and scheme-wise spending breakdown
- **Temporal Coverage**: Multi-year budget analysis (static dataset, no real-time updates)

**Explicitly Out of Scope**:
- Real-time budget data ingestion
- Autonomous policy recommendations
- State-level budgets (can be extended)
- Predictive forecasting or simulations

---

## Data Sources

The system uses **only official, publicly available government PDFs**. No proprietary or restricted data is included.

### Primary Data Sources

#### 1. Union Budget of India (Multiple Years)

**Source**: [https://www.indiabudget.gov.in](https://www.indiabudget.gov.in)

**Documents Used**:
- Budget at a Glance
- Expenditure Budget (Volume I & II)
- Demand for Grants (selected ministries)

**Focus Areas**:
- Infrastructure and Roads
- Capital Expenditure
- National Development Schemes
- Transport, Housing, and Urban Development

#### 2. NITI Aayog Reports

**Source**: [https://www.niti.gov.in](https://www.niti.gov.in)

**Documents Used**:
- Infrastructure sector analysis reports
- Public expenditure studies
- Development scheme evaluations

#### 3. Ministry-Level Reports (Selected)

**Ministries Included**:
- Ministry of Road Transport & Highways
- Ministry of Housing and Urban Affairs (infrastructure-related sections)

**Document Types**:
- Annual reports (infrastructure expenditure summaries)
- Scheme performance reports

#### 4. Parliamentary Documents (Limited)

**Source**: Lok Sabha / Rajya Sabha Q&A archives

**Usage**: Validation only (not primary data source)

### Dataset Characteristics

| Attribute | Value |
|-----------|-------|
| Total PDFs | 12–15 |
| Years Covered | 2019–2024 (approximate) |
| Primary Focus | Infrastructure, Roads, CapEx |
| Dataset Size | Intentionally kept small for clean RAG performance |

**Rationale for Limited Dataset**: A focused, high-quality dataset ensures accurate retrieval and reduces noise. The architecture is designed to scale to hundreds of documents.

---

## System Architecture

The system follows a standard RAG pipeline architecture:

```
User Query  
    ↓
Query Embedding Generation
    ↓
Vector Database Search (ChromaDB)
    ↓
Top-K Relevant Chunk Retrieval (k=5)
    ↓
Context Injection + Prompt Construction
    ↓
LLM Answer Generation (Gemini-2.5-Flash)
    ↓
Response Formatting (Natural Language + Tables + Citations)
    ↓
Optional Visualization (matplotlib)
```

### Component Breakdown

1. **Query Embedding**: User query is converted to a vector using the same embedding model used during indexing
2. **Vector Search**: ChromaDB performs cosine similarity search to retrieve top-k most relevant chunks
3. **Chunk Retrieval**: Retrieved chunks include metadata (year, ministry, page number) for citation
4. **LLM Generation**: Gemini-2.5-Flash generates a natural language answer using retrieved context
5. **Response Formatting**: Answer includes year-wise tables, scheme breakdowns, and source citations
6. **Visualization**: Optional matplotlib charts for trend analysis

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **RAG Framework** | LlamaIndex |
| **LLM** | Gemini-2.5-Flash (Google) |
| **Embeddings** | Gemini embeddings (or HuggingFace compatible model) |
| **Vector Database** | ChromaDB |
| **PDF Parsing** | PyMuPDF (fitz) or pdfplumber |
| **Backend API** | FastAPI (minimal REST endpoints) |
| **Visualization** | matplotlib (optional) |

### Explicitly NOT Included

- ❌ OpenAI APIs
- ❌ Jupyter notebooks (all code in `.py` files)
- ❌ Frontend (CLI/API only for judging)

---

## RAG Pipeline (Step-by-Step)

### Ingestion and Indexing

1. **PDF Ingestion**: Load PDFs from `data/raw_pdfs/` directory
2. **Text Extraction**: Extract text using PyMuPDF, preserving structure where possible
3. **Cleaning**: Remove headers, footers, page numbers, and formatting artifacts
4. **Chunking**: Split documents into 300–500 character chunks with 50-character overlap
5. **Metadata Enrichment**: Attach metadata to each chunk:
   - `year`: Budget year (e.g., 2023-24)
   - `ministry`: Source ministry (e.g., MoRTH)
   - `scheme`: Specific scheme name (if applicable)
   - `page_number`: Original page for citation
   - `document_title`: Source document name
6. **Embedding Generation**: Generate vector embeddings using LlamaIndex embedding model
7. **Vector Storage**: Store embeddings and metadata in ChromaDB with persistent storage

### Query and Retrieval

8. **Semantic Retrieval**: Convert user query to embedding and retrieve top-k=5 chunks using cosine similarity
9. **Answer Generation**: Construct prompt with retrieved chunks and system instructions, then query Gemini-2.5-Flash
10. **Response Formatting**: Structure response as:
    - Natural language summary
    - Year-wise or scheme-wise tables (Markdown format)
    - Citations with document name and page number
11. **Visualization (Optional)**: Generate matplotlib charts for trend queries

### Example Metadata Structure

```python
{
    "text": "Ministry of Road Transport & Highways was allocated Rs. 2,70,435 crore...",
    "metadata": {
        "year": "2024-25",
        "ministry": "MoRTH",
        "scheme": "Bharatmala Pariyojana",
        "page_number": 145,
        "document_title": "Expenditure Budget Vol II 2024-25"
    }
}
```

---

## Folder Structure

```
govinsight/
│
├── data/
│   ├── raw_pdfs/              # Original budget PDFs
│   ├── processed_chunks/      # Cleaned and chunked text (optional cache)
│
├── embeddings/                # Cached embeddings (optional)
│
├── vectorstore/               # ChromaDB persistent storage
│
├── app/
│   ├── main.py                # FastAPI app entry point
│   ├── rag_pipeline.py        # Core RAG logic (indexing + querying)
│   ├── pdf_processor.py       # PDF parsing and cleaning
│   ├── utils.py               # Helper functions
│
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── .env.example               # Environment variable template
```

### Directory Purposes

- **`data/raw_pdfs/`**: Place all source PDF files here before running the indexing pipeline
- **`vectorstore/`**: Persistent ChromaDB storage (automatically created)
- **`app/`**: All application code
- **`requirements.txt`**: Pinned dependencies for reproducibility

---

## Features Implemented

| Feature | Description |
|---------|-------------|
| **Natural Language Queries** | Ask questions in plain English about budget allocations |
| **Year-wise Breakdown** | Retrieve spending data across multiple budget years |
| **Scheme-wise Analysis** | Compare allocations for specific schemes (e.g., Bharatmala, PMGSY) |
| **Infrastructure Focus** | Specialized retrieval for roads, transport, and capital expenditure |
| **Citation-backed Answers** | Every answer includes source document and page number |
| **Missing Data Handling** | Gracefully returns "data not found" when information is unavailable |
| **Trend Visualization** | Optional matplotlib charts for multi-year comparisons |
| **Metadata Filtering** | Filter results by year, ministry, or scheme |

### Non-Features (Honest Scope)

- ❌ Autonomous AI insights or recommendations
- ❌ Predictive modeling
- ❌ Real-time data updates
- ❌ Unverified claims synthesis

---

## Example Queries

| Query | Expected Output |
|-------|-----------------|
| "How much did India allocate for road infrastructure in the last 5 years?" | Year-wise table with MoRTH allocations + citations |
| "Capital expenditure trend in Union Budget" | Multi-year CapEx breakdown + optional line chart |
| "Which year had the highest transport budget allocation?" | Single year + amount + source citation |
| "Compare Bharatmala and PMGSY allocations in 2023-24" | Scheme-wise comparison table |
| "Total infrastructure spending in Budget 2024-25" | Aggregated value + sector breakdown |
| "Ministry of Road Transport & Highways budget breakdown" | Scheme-wise allocations for MoRTH |

---

## How to Run

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd govinsight
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_api_key_here
```

### Step 5: Add PDF Documents

Place budget PDFs in the `data/raw_pdfs/` directory.

### Step 6: Build Vector Store (One-time Indexing)

```bash
python app/rag_pipeline.py --index
```

**Expected Output**:
```
Indexing 15 PDFs...
Extracted 1,247 chunks
Generated embeddings
Stored in ChromaDB
Indexing complete.
```

### Step 7: Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

**Server starts at**: `http://localhost:8000`

### Step 8: Query the System

**Via API**:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Road infrastructure budget for 2023-24"}'
```

**Via Python Script**:
```python
from app.rag_pipeline import query_rag

result = query_rag("How much was allocated to Bharatmala in 2024?")
print(result)
```

---

## Evaluation Notes for Judges

### Strengths

- **Reproducible**: All dependencies pinned, clear setup instructions
- **Transparent**: Citation-backed answers with source page numbers
- **Focused**: Intentionally limited dataset for clean performance
- **Production-ready Architecture**: Uses industry-standard RAG components
- **Extensible**: Can scale to 100+ PDFs and additional ministries

### Known Limitations

#### 1. PDF Quality Dependency
- Some budget PDFs contain scanned tables or low-quality text
- OCR-based PDFs may have character recognition errors
- Tabular data extraction is format-dependent

#### 2. Scheme Name Inconsistencies
- Government schemes are renamed or merged across years (e.g., PMGSY I → PMGSY II)
- The system may not automatically detect scheme relationships without explicit metadata

#### 3. Static Dataset
- Data is not updated in real-time from government websites
- Requires manual re-indexing when new budgets are published

#### 4. Aggregation Accuracy
- Multi-year aggregations depend on consistent categorization in source PDFs
- Manual validation recommended for critical financial analysis

#### 5. LLM Hallucination Risk
- While citation requirements reduce hallucination, LLM may occasionally misinterpret ambiguous text
- Cross-reference with source PDFs for critical use cases

---

## Future Enhancements (Not Implemented)

- Automated PDF scraping from government websites with version tracking
- OCR improvement pipeline for scanned documents
- Graph-based entity resolution for scheme name variations
- Multi-modal support for chart/table extraction
- State budget integration
- API rate limiting and authentication for production deployment

---

## Conclusion

GovInsight demonstrates how Retrieval-Augmented Generation can make complex government financial data accessible through natural language interfaces. By combining semantic search with structured metadata and citation-backed answers, the system reduces the manual effort required to analyze Union Budget documents.

### Technical Contributions

1. **Domain-Specific RAG Pipeline**: Tailored chunking and metadata enrichment for budget documents
2. **Hybrid Retrieval**: Combines vector similarity with metadata filtering for precise results
3. **Reproducible Architecture**: Clean separation of ingestion, retrieval, and generation layers

### Scalability Path

The current architecture supports:
- Expansion to 100+ documents without re-architecture
- Addition of state budgets and ministry-specific reports
- Integration with parliamentary Q&A archives
- Multi-lingual support (Hindi budget documents)

### Reliability

By requiring citations and gracefully handling missing data, GovInsight prioritizes accuracy over speculative answers. This makes it suitable for analysts, journalists, and researchers who need verifiable insights.

---

## Technical Support

For questions or issues during evaluation, refer to:
- System logs in `app/logs/` (auto-generated)
- ChromaDB admin interface: `http://localhost:8000/admin` (if enabled)
- Sample queries in `examples/` directory

---

**Project Type**: GenAI Hackathon Submission  
**Domain**: Public Finance & Policy Analysis  
**Architecture**: RAG (Retrieval-Augmented Generation)  
**Evaluation Priority**: Reproducibility, Technical Correctness, Citation Accuracy
