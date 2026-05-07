import hashlib
import re
from typing import Iterable, List

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-/\.]+")


def normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text).lower()))


def simple_tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(str(text).lower())


def stable_hash(text: str, modulo: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    h = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(h, 16) % modulo


def hash_tokens(tokens: Iterable[str], vocab_size: int, bos: int = 1, eos: int = 2, pad: int = 0) -> List[int]:
    ids = [bos]
    for tok in tokens:
        ids.append(3 + stable_hash(tok, max(vocab_size - 3, 1)))
    ids.append(eos)
    return ids
