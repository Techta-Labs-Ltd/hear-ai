import logging
import os

import torch
from sentence_transformers import SentenceTransformer

from hear.training.categorizer_train import CHECKPOINT_DIR, ClassifierHead, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder = None
_classifiers: dict[str, dict | None] = {}


def invalidate_classifier(target: str | None = None) -> None:
    """Force the next prediction to load the newly trained checkpoint."""
    if target is None:
        _classifiers.clear()
    else:
        _classifiers.pop(target, None)


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


def _load_classifier(target: str) -> dict | None:
    path = os.path.join(CHECKPOINT_DIR, f"{target}_classifier.pt")
    if not os.path.exists(path):
        _classifiers[target] = None
        return None
    checkpoint_mtime = os.path.getmtime(path)
    cached = _classifiers.get(target)
    if cached is not None and cached.get("checkpoint_mtime") == checkpoint_mtime:
        return cached
    ckpt = torch.load(path, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClassifierHead(ckpt["in_dim"], len(ckpt["label_names"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    entry = {
        "model": model,
        "label_names": ckpt["label_names"],
        "multi_label": ckpt.get("multi_label", False),
        "device": device,
        "checkpoint_mtime": checkpoint_mtime,
    }
    _classifiers[target] = entry
    logger.info("categorizer_infer loaded target=%s labels=%d", target, len(entry["label_names"]))
    return entry


def predict(target: str, text: str) -> dict[str, float]:
    """Returns {label: score} from the trained checkpoint, or {} if none exists yet."""
    entry = _load_classifier(target)
    if entry is None or not text or not text.strip():
        return {}
    try:
        embedder = _get_embedder()
        emb = embedder.encode([text[:2000]], convert_to_numpy=True, normalize_embeddings=True)
        with torch.no_grad():
            logits = entry["model"](torch.from_numpy(emb).float().to(entry["device"]))
            scores = torch.sigmoid(logits)[0] if entry["multi_label"] else torch.softmax(logits, dim=-1)[0]
        return {name: float(scores[i]) for i, name in enumerate(entry["label_names"])}
    except Exception:
        logger.warning("categorizer_infer predict failed target=%s", target, exc_info=True)
        return {}
