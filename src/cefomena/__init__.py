"""Neural-CEFOMENA: modelo conversacional ligero con arquitectura híbrida bioinspirada."""

from cefomena.config import CefomenaConfig
from cefomena.model.model import CefomenaLM
from cefomena.tokenizer.bpe import BPETokenizer

__all__ = ["BPETokenizer", "CefomenaConfig", "CefomenaLM"]
__version__ = "0.1.0"
