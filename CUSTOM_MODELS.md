# Custom Models Integration Guide 🚀

How to integrate **custom embedding models** (like Fable 5) with Claude Knowledge Graph.

---

## Quick Decision Tree

```
Do you have Fable 5?
    │
    ├─ YES → What type?
    │   ├─ Embedding model? → Use as custom embedder
    │   ├─ LLM/Generation? → Use alongside Claude API
    │   └─ Hybrid? → Combine both
    │
    └─ NO → Which custom model?
        ├─ OpenAI embeddings? → Using openai library
        ├─ Open-source LLM? → Using ollama/llamacpp
        └─ Custom API? → Using requests library
```

---

## Supported Integration Patterns

### Pattern 1: Replace Embedding Model (Custom Embedder)

**Use Case**: Use Fable 5 for embeddings instead of sentence-transformers

#### Step 1: Create Wrapper

```python
# embedders.py
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbedderBase(ABC):
    @abstractmethod
    def encode(self, texts: list) -> np.ndarray:
        pass

class SentenceTransformerEmbedder(EmbedderBase):
    """Default: sentence-transformers"""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def encode(self, texts):
        return self.model.encode(texts)

class Fable5Embedder(EmbedderBase):
    """Fable 5 custom embedder"""
    def __init__(self, api_key=None, model_name="fable-5"):
        self.api_key = api_key
        self.model_name = model_name
        # Initialize Fable 5 client
        # ...
    
    def encode(self, texts):
        # Call Fable 5 API/local model
        # Return embeddings as numpy array
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def _get_embedding(self, text: str):
        # Your Fable 5 implementation
        # Should return: [0.12, -0.45, 0.89, ...]
        pass
```

#### Step 2: Modify claude_kg.py

```python
from embedders import SentenceTransformerEmbedder, Fable5Embedder

class ClaudeKnowledgeGraph:
    def __init__(self, 
                 db_path: str = "./kg_db",
                 embedder_type: str = "sentence-transformers",
                 embedder_config: dict = None):
        """
        embedder_type: "sentence-transformers", "fable5", "openai", etc.
        embedder_config: Additional params for embedder
        """
        
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize embedder based on type
        if embedder_type == "sentence-transformers":
            model_name = embedder_config.get("model_name", "all-MiniLM-L6-v2")
            self.embedder = SentenceTransformerEmbedder(model_name)
        
        elif embedder_type == "fable5":
            self.embedder = Fable5Embedder(
                api_key=embedder_config.get("api_key"),
                model_name=embedder_config.get("model_name", "fable-5")
            )
        
        elif embedder_type == "openai":
            self.embedder = OpenAIEmbedder(
                api_key=embedder_config.get("api_key")
            )
        
        else:
            raise ValueError(f"Unknown embedder: {embedder_type}")
        
        # Rest of initialization...
        settings = Settings(...)
        self.db = chromadb.Client(settings)
        self.collection = self.db.get_or_create_collection(...)
```

#### Step 3: Usage

```python
# Using Fable 5
kg = ClaudeKnowledgeGraph(
    embedder_type="fable5",
    embedder_config={
        "api_key": "your-fable5-key",
        "model_name": "fable-5"
    }
)

kg.add_documents(docs)
kg.save_db()
```

---

### Pattern 2: Use Fable 5 as Generation Model (Alongside Claude)

**Use Case**: Use Fable 5 for response generation instead of Claude

#### Step 1: Create Fable5 Client

```python
# fable5_client.py

class Fable5Client:
    def __init__(self, api_key=None, model_name="fable-5"):
        self.api_key = api_key or os.getenv("FABLE5_API_KEY")
        self.model_name = model_name
        # Initialize client
        # This depends on how Fable 5 is provided (API/local)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response using Fable 5"""
        # Call Fable 5 API or local model
        response = self._call_fable5(
            prompt=prompt,
            max_tokens=max_tokens
        )
        return response["text"]
    
    def _call_fable5(self, prompt, max_tokens):
        # Your implementation
        pass
```

#### Step 2: Modify query() method

```python
class ClaudeKnowledgeGraph:
    def __init__(self, ..., generator_type="claude"):
        # ...
        self.generator_type = generator_type
        
        if generator_type == "claude":
            self.client = Anthropic(api_key=self.api_key)
        elif generator_type == "fable5":
            self.fable5_client = Fable5Client()
        else:
            raise ValueError(f"Unknown generator: {generator_type}")
    
    def query(self, user_message: str, use_kg: bool = True) -> str:
        # Retrieve relevant docs
        relevant_docs = []
        if use_kg:
            relevant_docs = self.retrieve(user_message, top_k=3)
        
        # Build context
        context = self._build_context(relevant_docs)
        full_message = f"{context}\nUser: {user_message}"
        
        # Generate response
        if self.generator_type == "claude":
            response = self._query_claude(full_message)
        elif self.generator_type == "fable5":
            response = self.fable5_client.generate(full_message)
        
        return response
```

#### Step 3: Usage

```python
# Use Fable 5 for generation
kg = ClaudeKnowledgeGraph(generator_type="fable5")
kg.add_documents(docs)

response = kg.query("Your question", use_kg=True)
```

---

### Pattern 3: Hybrid (Fable 5 for Embeddings + Fable 5 for Generation)

**Use Case**: Complete replacement with Fable 5

```python
kg = ClaudeKnowledgeGraph(
    embedder_type="fable5",
    embedder_config={"api_key": "..."},
    generator_type="fable5"
)

# Everything uses Fable 5
kg.add_documents(docs)  # Uses Fable 5 embeddings
kg.query("Question")    # Uses Fable 5 generation
```

---

### Pattern 4: Ensemble (Multiple Models)

**Use Case**: Use best of both - Claude for reasoning, Fable 5 for speed

```python
class HybridKG:
    def __init__(self):
        self.embedder = Fable5Embedder()  # Fast embeddings
        self.claude = Anthropic()           # Best generation
        self.fable5 = Fable5Client()        # Fallback
    
    def query(self, message: str, fast_mode=False):
        context = self.retrieve(message)  # Fable 5 embeddings
        
        if fast_mode:
            return self.fable5.generate(f"{context}\n{message}")
        else:
            return self._query_claude(f"{context}\n{message}")
```

---

## Implementation Templates

### Template 1: Fable 5 API Integration

```python
import requests
import os

class Fable5EmbedderAPI(EmbedderBase):
    """Call Fable 5 via REST API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    def encode(self, texts: list) -> np.ndarray:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"texts": texts, "model": "fable-5"}
        
        response = requests.post(
            f"{self.base_url}/embed",
            json=payload,
            headers=headers
        )
        
        embeddings = response.json()["embeddings"]
        return np.array(embeddings)
```

### Template 2: Fable 5 Local Model

```python
from transformers import AutoTokenizer, AutoModel
import torch

class Fable5EmbedderLocal(EmbedderBase):
    """Run Fable 5 locally"""
    
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def encode(self, texts: list) -> np.ndarray:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()
```

### Template 3: Fable 5 Generation

```python
class Fable5Generator:
    """Fable 5 as LLM"""
    
    def __init__(self, model_path: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0])
```

---

## Performance Comparison

### Embedding Models

| Model | Speed | Quality | Size | Cost |
|-------|-------|---------|------|------|
| **sentence-transformers** | Fast | Good | 83MB | Free |
| **Fable 5** | ? | ? | ? | TBD |
| **OpenAI Ada** | Slow | Best | - | $0.10/1M |

### Generation Models

| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| **Claude Haiku** | Fast | Good | $0.80/1M |
| **Claude Sonnet** | Balanced | Great | $3/1M |
| **Claude Opus** | Slow | Best | $15/1M |
| **Fable 5** | ? | ? | ? |

---

## Configuration Examples

### Config A: Fable 5 Embeddings + Claude Generation

```python
config = {
    "embedder": {
        "type": "fable5",
        "config": {
            "api_key": os.getenv("FABLE5_KEY"),
            "base_url": "https://api.fable5.ai"
        }
    },
    "generator": {
        "type": "claude",
        "model": "claude-opus-4-6"
    },
    "retrieval": {
        "top_k": 3
    }
}
```

### Config B: Fable 5 Everything

```python
config = {
    "embedder": {
        "type": "fable5",
        "config": {"local": True}  # Run locally
    },
    "generator": {
        "type": "fable5",
        "model": "fable-5-large"
    }
}
```

### Config C: Hybrid Ensemble

```python
config = {
    "primary": {
        "embedder": "fable5",
        "generator": "claude"
    },
    "fallback": {
        "embedder": "sentence-transformers",
        "generator": "fable5"
    }
}
```

---

## Testing & Validation

### Test Embedding Quality

```python
def test_embeddings():
    embedder = Fable5Embedder()
    
    # Test semantic similarity
    texts = [
        "What is machine learning?",
        "How does ML work?",  # Should be similar
        "Cats are fluffy"      # Should be different
    ]
    
    embeddings = embedder.encode(texts)
    
    # Compute similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    print(f"Q1-Q2 similarity: {sim_matrix[0][1]:.2f}")  # Should be high
    print(f"Q1-Cat similarity: {sim_matrix[0][2]:.2f}") # Should be low
```

### Test Generation Quality

```python
def test_generation():
    kg = ClaudeKnowledgeGraph(generator_type="fable5")
    
    test_queries = [
        "Explain machine learning",
        "How do I use Docker?",
        "What is Python?"
    ]
    
    for query in test_queries:
        response = kg.query(query)
        print(f"Q: {query}")
        print(f"A: {response}\n")
```

---

## Troubleshooting

### Custom Embedder Issues

```python
# Error: Shape mismatch
# Solution: Ensure embeddings are 1D array
embeddings = embedder.encode(texts)
assert embeddings.shape == (len(texts), embedding_dim)

# Error: API timeout
# Solution: Add retry logic
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3))
def encode_with_retry(texts):
    return embedder.encode(texts)
```

---

## Next Steps

1. **Clarify Fable 5 type**: Embedding model? LLM? Both?
2. **Get API/Model details**: How to access Fable 5?
3. **Choose integration pattern**: Which template above?
4. **Implement wrapper**: Modify embedders.py
5. **Test thoroughly**: Validate accuracy & speed

**Tell me about Fable 5!** 🎯
