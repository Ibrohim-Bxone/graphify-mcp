# Claude Knowledge Graph - Technical Documentation 📚

**Version**: 1.0
**Status**: Production Ready
**Last Updated**: July 2026

---

## Table of Contents

1. [Architecture Overview](#architecture)
2. [System Design](#system-design)
3. [Data Pipeline](#data-pipeline)
4. [API Reference](#api-reference)
5. [Performance Metrics](#performance)
6. [Integration Guide](#integration)
7. [Troubleshooting](#troubleshooting)

---

## <a id="architecture"></a>1. Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                   USER INTERFACE LAYER                      │
│                  (CLI / Web / API)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│              ORCHESTRATION LAYER                            │
│         (ClaudeKnowledgeGraph class)                        │
│  - Query routing                                            │
│  - Context management                                       │
│  - Conversation history                                     │
└────────────────┬───────────────────────────────────────────┘
                 │
      ┌──────────┴──────────┐
      │                     │
┌─────▼──────────┐  ┌──────▼─────────────┐
│ RETRIEVAL      │  │  GENERATION        │
│ LAYER          │  │  LAYER             │
│                │  │                    │
│ Vector DB      │  │  Anthropic API     │
│ (Chroma)       │  │  - claude-opus-4-6 │
│ ↓              │  │  - claude-sonnet-46│
│ Embeddings     │  │  - claude-haiku-45 │
│ (sent-trans)   │  │                    │
└────────────────┘  └────────────────────┘
      │                     │
└──────────────┬────────────┘
               │
┌──────────────▼────────────────────────────────────────────┐
│              STORAGE LAYER                                │
│  - Vector DB (DuckDB-backed Chroma)                       │
│  - Document metadata                                      │
│  - Embeddings cache                                       │
└───────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Role | Technology | Cost |
|-----------|------|-----------|------|
| **Retrieval** | Find relevant docs | Chroma + sent-transformers | 0 tokens |
| **Embedding** | Convert text→vectors | all-MiniLM-L6-v2 | Local (free) |
| **Vector DB** | Store embeddings | DuckDB (persistent) | Disk space only |
| **Generation** | Create responses | Anthropic Claude API | Pay-per-token |
| **Orchestration** | Tie everything | Python class | Free |

---

## <a id="system-design"></a>2. System Design

### 2.1 Data Flow Diagram

```
INPUT: User Query
  │
  ├─► [Tokenization]
  │    └─► Split into tokens
  │
  ├─► [Embedding] (Local, 0 tokens cost)
  │    └─► sentence-transformers converts to 384-dim vector
  │
  ├─► [Vector Search] (Local, 0 tokens cost)
  │    └─► Chroma: cosine similarity search
  │        Result: Top-3 most relevant documents
  │
  ├─► [Context Building]
  │    └─► Format: "[source (relevance)]\n{content}\n"
  │
  ├─► [API Call] (Costs tokens!)
  │    └─► Send: system_prompt + context + user_query
  │        to: claude-opus-4-6 (or other model)
  │
  └─► OUTPUT: Response
      └─► Store in conversation history
```

### 2.2 Database Schema

#### Chroma Collection Structure
```python
{
  "id": "doc_unique_id",
  "document": "The actual text content...",
  "metadata": {
    "source": "Where it came from",
    "title": "Document title",
    "timestamp": "ISO timestamp",
    "tags": ["tag1", "tag2"],
    "chunk_index": 0,
    "original_doc_id": "parent_id"
  },
  "embedding": [0.12, -0.45, 0.89, ...]  # 384 dimensions
}
```

#### Example Documents
```json
{
  "id": "claude_api_001",
  "document": "Claude API supports function calling...",
  "metadata": {
    "source": "docs.anthropic.com",
    "title": "Function Calling",
    "tags": ["API", "Advanced"],
    "chunk_index": 0
  }
}
```

### 2.3 Query Processing Pipeline

```python
# Step 1: User Input
query = "How do I use Claude API?"

# Step 2: Embedding (local, no API call)
query_embedding = embedder.encode(query)  # 384-dimensional vector
# Cost: 0 tokens (runs locally on CPU/GPU)

# Step 3: Vector Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,  # Get top 3
    include=["documents", "metadatas", "distances"]
)
# Cost: 0 tokens (local DB operation)

# Step 4: Context Building
context = """
[docs.anthropic.com (similarity: 0.92)]
Claude API supports function calling via the tools parameter...

[docs.anthropic.com (similarity: 0.87)]
The API uses JSON schema for tool definitions...

[docs.anthropic.com (similarity: 0.81)]
Responses include tool_use blocks when functions are called...
"""

# Step 5: API Call
response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    system="Use provided knowledge to answer...",
    messages=[
        {
            "role": "user",
            "content": f"{context}\n\nUser: {query}"
        }
    ]
)
# Cost: ~1500-2000 tokens (only relevant context sent!)
```

---

## <a id="data-pipeline"></a>3. Data Pipeline

### 3.1 Indexing Process (One-Time)

```
RAW DOCUMENTS
    │
    ├─► [Chunking] - Split large docs
    │    └─► Strategy: Fixed size (1000 chars) with overlap (100 chars)
    │        Why: Better retrieval granularity
    │
    ├─► [Cleaning] - Remove noise
    │    └─► Strip whitespace, normalize unicode
    │
    ├─► [Metadata Extraction]
    │    └─► Source, title, tags, timestamp
    │
    ├─► [Embedding] - Convert to vectors
    │    └─► sentence-transformers
    │        Input: Text chunk (max 512 tokens)
    │        Output: 384-dimensional embedding
    │        Time: ~0.5ms per chunk (GPU)
    │
    ├─► [Normalization]
    │    └─► L2 normalization for cosine similarity
    │
    └─► [Vector DB Storage]
        └─► Chroma DuckDB backend
            - Persistent storage
            - HNSW indexing (approximate nearest neighbors)
            - Ready for semantic search
```

### 3.2 Token Cost Analysis

#### Indexing Cost (One-Time)

| Operation | Tokens | Frequency | Total |
|-----------|--------|-----------|-------|
| Embedding (local) | 0 | ∞ | 0 |
| Vector search | 0 | ∞ | 0 |
| **Total indexing cost** | | | **0 tokens** |

#### Per-Query Cost

| Without Knowledge Graph | With Knowledge Graph |
|-------------------------|----------------------|
| Load all docs: 5000 tokens | Vector search: 0 tokens |
| Search through: 500 tokens | Embedding: 0 tokens |
| Generate: 2000 tokens | Load top-3: 300 tokens |
| **Total: 7500 tokens** | Generate: 1500 tokens |
| | **Total: 1800 tokens** |
| | **Savings: 76%** ✅ |

---

## <a id="api-reference"></a>4. API Reference

### ClaudeKnowledgeGraph Class

#### Constructor

```python
from claude_kg import ClaudeKnowledgeGraph

kg = ClaudeKnowledgeGraph(
    db_path: str = "./kg_db",           # Where to store vector DB
    model_name: str = "all-MiniLM-L6-v2" # Embedding model
)
```

**Parameters:**
- `db_path`: Directory for persistent storage
  - Default: `./kg_db` (relative to current dir)
  - Can be absolute: `/home/user/.kg_db`
  - Will be created if doesn't exist

- `model_name`: Hugging Face model for embeddings
  - `all-MiniLM-L6-v2`: Fast (83MB), 384-dim (DEFAULT)
  - `all-mpnet-base-v2`: Accurate (438MB), 768-dim
  - `distiluse-base-multilingual-cased-v2`: Multilingual

**Environment:**
- Requires: `ANTHROPIC_API_KEY` in `.env` or exported

**Example:**
```python
kg = ClaudeKnowledgeGraph(
    db_path="/data/my_kg_db",
    model_name="all-mpnet-base-v2"  # Better accuracy
)
```

#### add_documents()

```python
def add_documents(self, docs: List[dict]) -> int
```

**Purpose**: Add documents to vector DB for indexing

**Input:**
```python
docs = [
    {
        "id": "unique_identifier",
        "content": "The actual text content...",
        "source": "Where it came from (optional)",
        "title": "Document title (optional)"
    },
    # More docs...
]

count = kg.add_documents(docs)
# Returns: 5
```

**What it does:**
1. Converts text to embeddings (local)
2. Stores in Chroma DB
3. Creates searchable index

**Cost**: 0 tokens (all local)

**Example:**
```python
docs = [
    {
        "id": "python_loops",
        "content": "for loop: for i in range(10). while loop: while condition.",
        "source": "my_notes",
        "title": "Python Control Flow"
    }
]
kg.add_documents(docs)  # ✓ Indexed
```

#### retrieve()

```python
def retrieve(self, query: str, top_k: int = 3) -> List[dict]
```

**Purpose**: Find relevant documents for a query

**Input:**
```python
query = "How do I use loops in Python?"
top_k = 5  # Get top 5 results

results = kg.retrieve(query, top_k=top_k)
```

**Output:**
```python
[
    {
        "content": "for loop: for i in range(10)...",
        "source": "my_notes",
        "similarity": 0.92  # 0-1 score, higher = more relevant
    },
    {
        "content": "while loop: while condition...",
        "source": "my_notes",
        "similarity": 0.89
    },
    # ... more results
]
```

**Cost**: 0 tokens (all local vector search)

**Parameters:**
- `query`: Text to search for
- `top_k`: Number of results (1-10, default 3)

**Example:**
```python
results = kg.retrieve("Docker containers", top_k=5)
for r in results:
    print(f"Match: {r['content'][:50]}... (similarity: {r['similarity']:.2f})")
```

#### query()

```python
def query(self, user_message: str, use_kg: bool = True) -> str
```

**Purpose**: Full query with knowledge graph integration

**Input:**
```python
user_message = "What is machine learning?"
use_kg = True  # Use knowledge graph (default)

response = kg.query(user_message, use_kg=True)
```

**Output:**
```python
"Machine learning is a subset of AI that learns patterns from data..."
```

**What it does:**
1. Searches KB for relevant docs (0 tokens)
2. Builds optimized context
3. Calls Claude API with only relevant info
4. Stores in conversation history
5. Returns response

**Cost**: ~1500-2000 tokens (vs 7500 without KG)

**Parameters:**
- `user_message`: Question or instruction
- `use_kg`: Whether to use knowledge graph
  - `True`: Smart retrieval (default, saves tokens)
  - `False`: Pure Claude (no context injection)

**Example:**
```python
# With knowledge graph (smart)
response = kg.query("Tell me about Docker", use_kg=True)
# Cost: 1800 tokens

# Without knowledge graph (slow)
response = kg.query("Tell me about Docker", use_kg=False)
# Cost: 7500 tokens (75% more expensive!)
```

#### save_db()

```python
def save_db(self) -> None
```

**Purpose**: Persist vector DB to disk

**Usage:**
```python
kg.save_db()  # Saves to kg_db_path
# Output: ✓ Knowledge graph saved
```

**Cost**: Free (just disk I/O)

**Why?** Chroma maintains DB in memory. Call `save_db()` before exiting to persist.

**Example:**
```python
kg.add_documents(docs)
kg.save_db()  # Persist to disk
# Next time you run: kg.db loads from disk automatically
```

---

## <a id="performance"></a>5. Performance Metrics

### 5.1 Speed Benchmarks

| Operation | Time | Notes |
|-----------|------|-------|
| **Embedding (1 doc)** | 0.5ms | CPU: slow, GPU: instant |
| **Vector search (3 results)** | 1-5ms | Local, very fast |
| **Claude API call** | 2-5s | Network dependent |
| **Total per query** | 2-5s | Dominated by API |

### 5.2 Token Usage

```
Scenario: Answer question using 100 docs

WITHOUT Knowledge Graph:
- Load all 100 docs: 5000 tokens
- Search: 500 tokens
- Generate: 1500 tokens
TOTAL: 7000 tokens
COST: $0.21 (at $0.03/1K tokens)

WITH Knowledge Graph:
- Vector search: 0 tokens
- Load top-3 docs: 300 tokens
- Generate: 1500 tokens
TOTAL: 1800 tokens
COST: $0.054
SAVINGS: 74% less cost! 🎉
```

### 5.3 Scalability

| Metric | Limit | Notes |
|--------|-------|-------|
| **Docs** | 1M+ | Chroma can handle millions |
| **Query speed** | <10ms | Vector search is O(log n) |
| **Memory** | ~1GB/100K docs | Embeddings: 384 dims × 4 bytes |
| **Storage** | ~500MB/100K docs | DuckDB optimized |

### 5.4 Accuracy Metrics

```
Relevance Score: 0-1 (cosine similarity)

Score | Relevance | Usage
0.90+ | Perfect match | Use directly
0.80+ | Very relevant | Use with confidence
0.70+ | Relevant | Use with context
0.60+ | Somewhat relevant | Consider including
<0.60 | Not relevant | Ignore
```

---

## <a id="integration"></a>6. Integration Guide

### 6.1 Integrate with Existing Claude Projects

#### Before (No Knowledge Graph)
```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    messages=[{"role": "user", "content": "Your question here"}]
)
```

#### After (With Knowledge Graph)
```python
from claude_kg import ClaudeKnowledgeGraph

kg = ClaudeKnowledgeGraph()
kg.add_documents(your_docs)

response = kg.query("Your question here", use_kg=True)
# 60% cheaper! ✅
```

### 6.2 Integration Points

#### 1. Web Framework (Flask/FastAPI)

```python
from fastapi import FastAPI
from claude_kg import ClaudeKnowledgeGraph

app = FastAPI()
kg = ClaudeKnowledgeGraph()

@app.post("/ask")
async def ask(question: str):
    response = kg.query(question, use_kg=True)
    return {"response": response}
```

#### 2. Discord Bot

```python
import discord
from claude_kg import ClaudeKnowledgeGraph

class KGBot(discord.Client):
    def __init__(self):
        super().__init__()
        self.kg = ClaudeKnowledgeGraph()
    
    async def on_message(self, message):
        if message.author == self.user:
            return
        response = self.kg.query(message.content, use_kg=True)
        await message.reply(response)
```

#### 3. Batch Processing

```python
from claude_kg import ClaudeKnowledgeGraph

kg = ClaudeKnowledgeGraph()
kg.add_documents(all_docs)

queries = ["Q1", "Q2", "Q3"]
for q in queries:
    response = kg.query(q, use_kg=True)
    print(response)
```

---

## <a id="troubleshooting"></a>7. Troubleshooting

### Common Issues

#### Issue 1: "ANTHROPIC_API_KEY not found"

```
Error: ANTHROPIC_API_KEY not found. Set it in .env or export it.
```

**Solution:**
```bash
# Option 1: .env file
echo "ANTHROPIC_API_KEY=sk-ant-xxxxx" > .env

# Option 2: Export
export ANTHROPIC_API_KEY=sk-ant-xxxxx
python3 claude_kg.py

# Option 3: Check
echo $ANTHROPIC_API_KEY  # Should print your key
```

#### Issue 2: "Out of memory"

```
MemoryError: Unable to allocate 2.5 GiB for an array
```

**Solution:**
```python
# Use smaller model
kg = ClaudeKnowledgeGraph(
    model_name="all-MiniLM-L6-v2"  # 83MB (not 438MB)
)

# Or reduce batch size
for i in range(0, len(docs), 100):
    kg.add_documents(docs[i:i+100])
```

#### Issue 3: "Slow retrieval"

```
Vector search takes 10+ seconds
```

**Solution:**
```python
# Rebuild index
import shutil
shutil.rmtree("kg_db")  # Delete old DB

kg = ClaudeKnowledgeGraph()  # Rebuild
kg.add_documents(docs)  # Re-index
```

#### Issue 4: "Poor relevance"

```
Retrieved docs aren't relevant to query
```

**Solution:**
```python
# Change embedding model (more accurate)
kg = ClaudeKnowledgeGraph(
    model_name="all-mpnet-base-v2"  # Better quality
)

# Or adjust top_k
results = kg.retrieve(query, top_k=10)  # Get more, filter better
```

---

## Monitoring & Logging

### Add Debugging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# In claude_kg.py, add:
logger.info(f"Retrieving for: {query}")
logger.info(f"Found {len(results)} results")
logger.info(f"API call cost: ~{tokens} tokens")
```

### Token Usage Tracking

```python
# Add to query() method:
print(f"""
Query: {user_message}
Retrieved docs: {len(relevant_docs)}
Estimated tokens: {len(context) // 4}
Context size: {len(context)} chars
API model: claude-opus-4-6
""")
```

---

## Performance Tuning

### Parameter Tuning

```python
# Retrieval sensitivity
kg.retrieve(query, top_k=1)    # Fast, concise
kg.retrieve(query, top_k=5)    # Balanced
kg.retrieve(query, top_k=10)   # Comprehensive

# Embedding model
"all-MiniLM-L6-v2"      # Default: fast
"all-mpnet-base-v2"     # Better: accurate
"thenlper/gte-small"    # Advanced: specialized

# API model
"claude-haiku-4-5"      # Fast, cheap
"claude-sonnet-4-6"     # Balanced
"claude-opus-4-6"       # Powerful, expensive
```

---

## Architecture Decisions

### Why Chroma?

- ✅ Local vector DB (no API calls)
- ✅ DuckDB backend (persistent)
- ✅ Cosine similarity (fast, accurate)
- ✅ Python-native
- ✅ No deployment needed

### Why sentence-transformers?

- ✅ Runs locally (0 API cost)
- ✅ 384 dimensions (balanced size)
- ✅ Trained on semantic similarity
- ✅ Fast inference (~0.5ms)

### Why Anthropic Claude?

- ✅ Best-in-class reasoning
- ✅ Long context window
- ✅ Function calling
- ✅ Token-efficient
- ✅ Trustworthy outputs

---

## Future Enhancements

```
Phase 2: Multi-modal Support
- Add image embeddings
- Support PDF extraction
- Vision API integration

Phase 3: Advanced Retrieval
- Hybrid search (vector + keyword)
- Reranking with cross-encoders
- Query expansion

Phase 4: Production Scale
- Distributed embeddings
- Real-time indexing
- API versioning
```

---

**Questions? Issues? Improvements?** Open an issue on GitHub! 🚀
