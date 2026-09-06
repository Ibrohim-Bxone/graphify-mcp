"""
Token-based chunking for Graphify.
"""
import re
from tokenizers import Tokenizer
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

_ef = ONNXMiniLM_L6_V2()
MAX_TOKENS = _ef.max_tokens()

# Create a non-truncating, non-padding tokenizer
tokenizer = Tokenizer.from_str(_ef.tokenizer.to_str())
tokenizer.no_truncation()
tokenizer.no_padding()

PREFIX_BUDGET = 32
# 2 special tokens: [CLS] and [SEP]
CONTENT_BUDGET = MAX_TOKENS - PREFIX_BUDGET - 2


def count_tokens(text: str) -> int:
    # exclude special tokens if we just want raw length, but wait:
    # tokenizer.encode().ids includes [CLS] and [SEP]. So len(ids) = actual + 2
    # but here we just need a consistent measure.
    # We will use len(ids) - 2 for content tokens, or just len(ids) to be safe.
    if not text:
        return 0
    return len(tokenizer.encode(text).ids) - 2


def chunk_by_tokens(text: str, *, max_tokens: int = MAX_TOKENS, overlap_tokens: int = 40, parent_id: str = "") -> list[dict]:
    """
    Chunks text by tokens.
    Returns a list of dicts: {"text": str, "token_count": int, "parent_id": str, "chunk_index": int, "chunk_total": int}
    """
    text = text.strip()
    if not text:
        return []

    # Safe budget for content
    # If the user overrides max_tokens, we need to recalculate budget
    # The budget is for the content string itself (without special tokens, as we subtract 2)
    budget = max_tokens - PREFIX_BUDGET - 2
    if budget <= 0:
        raise ValueError("max_tokens too small to accommodate PREFIX_BUDGET and special tokens")

    # Fast path if entire text fits
    total_tokens = count_tokens(text)
    if total_tokens <= budget:
        return [{
            "text": text,
            "token_count": total_tokens,
            "parent_id": parent_id,
            "chunk_index": 0,
            "chunk_total": 1,
        }]

    # We need to split. We'll split the text into a list of words/spaces to preserve exact text.
    # We can split by paragraphs/sentences first, but word-level token building is most accurate.
    # Let's split by regex that keeps separators:
    # \n\n, \n, . , space
    
    parts = re.split(r'(\n\n|\n|\. |\s+)', text)
    # parts will be [word, sep, word, sep, ...]
    
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    i = 0
    while i < len(parts):
        part = parts[i]
        if not part:
            i += 1
            continue
            
        part_tokens = count_tokens(part)
        
        # If a single part is larger than budget (very rare, e.g. a huge string with no spaces)
        if part_tokens > budget:
            # We have to slice it by characters, measuring as we go
            # Simple binary search or linear build
            if current_chunk:
                chunks.append((current_chunk.strip(), current_tokens))
                current_chunk = ""
                current_tokens = 0
            
            # Sub-split the huge part
            # To be simple and safe, we add char by char (or larger steps)
            temp = ""
            for char in part:
                temp_new = temp + char
                if count_tokens(temp_new) > budget:
                    chunks.append((temp.strip(), count_tokens(temp.strip())))
                    temp = char
                else:
                    temp = temp_new
            if temp:
                current_chunk = temp
                current_tokens = count_tokens(temp)
            i += 1
            continue

        if current_tokens + part_tokens > budget and current_chunk.strip():
            chunks.append((current_chunk.strip(), count_tokens(current_chunk.strip())))
            
            # Start new chunk with overlap
            # overlap needs to go back by 'overlap_tokens'
            # Let's do a simple approach: if we push to chunks, we look at current_chunk
            # and take the last few words to satisfy overlap_tokens.
            
            overlap_text = ""
            if overlap_tokens > 0:
                # Find the suffix of current_chunk that fits in overlap_tokens
                # We can split current_chunk by words and take from the end
                overlap_parts = re.split(r'(\s+)', current_chunk)
                suffix = ""
                for p in reversed(overlap_parts):
                    new_suffix = p + suffix
                    if count_tokens(new_suffix) > overlap_tokens:
                        break
                    suffix = new_suffix
                overlap_text = suffix.lstrip() # don't start overlap with spaces
            
            current_chunk = overlap_text + part
            current_tokens = count_tokens(current_chunk)
        else:
            current_chunk += part
            current_tokens = count_tokens(current_chunk)
            
        i += 1
        
    if current_chunk.strip():
        chunks.append((current_chunk.strip(), count_tokens(current_chunk.strip())))
        
    # Convert to dicts
    result = []
    chunk_total = len(chunks)
    for idx, (c_text, c_tokens) in enumerate(chunks):
        result.append({
            "text": c_text,
            "token_count": c_tokens,
            "parent_id": parent_id,
            "chunk_index": idx,
            "chunk_total": chunk_total
        })
        
    return result
