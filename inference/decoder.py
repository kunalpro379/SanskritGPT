
import sentencepiece as spm
from typing import List, Optional
def decode_tokens(tokens: List[int], model_path: Optional[str] = None) -> str:
    if model_path is None:
        model_path = 'sanskrit_spm.model'
    
    try:
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        return sp.decode(tokens)
    except Exception as e:
        print(f"Decoding failed: {e}")
        return f"Token IDs: {tokens[:20]}..."