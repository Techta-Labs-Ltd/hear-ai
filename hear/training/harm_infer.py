import logging
import os

import torch
from sentence_transformers import SentenceTransformer

from hear.training.categorizer_train import CHECKPOINT_DIR, ClassifierHead, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder = None
_entry = None
_entry_mtime = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            local_files_only=True,
        )
    return _embedder

def _load() -> dict | None:
    global _entry, _entry_mtime
    path = os.path.join(CHECKPOINT_DIR, "harm_classifier.pt")
    if not os.path.exists(path):
        _entry = None
        _entry_mtime = None
        return None
    checkpoint_mtime = os.path.getmtime(path)
    if _entry is not None and _entry_mtime == checkpoint_mtime:
        return _entry
    ckpt = torch.load(path, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClassifierHead(ckpt["in_dim"], len(ckpt["label_names"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _entry = {"model": model, "label_names": ckpt["label_names"], "device": device}
    _entry_mtime = checkpoint_mtime
    logger.info("harm_infer loaded labels=%s", ckpt["label_names"])
    return _entry
    return _entry


def predict(text: str) -> float:
    entry = _load()
    if entry is None or not text or not text.strip():
        return -1.0
    try:
        embedder = _get_embedder()
        emb = embedder.encode([text[:2000]], convert_to_numpy=True, normalize_embeddings=True)
        with torch.no_grad():
            logits = entry["model"](torch.from_numpy(emb).float().to(entry["device"]))
            scores = torch.softmax(logits, dim=-1)[0]
        harmful_idx = entry["label_names"].index("harmful")
        return float(scores[harmful_idx])
    except Exception:
        logger.warning("harm_infer predict failed", exc_info=True)
        return -1.0
