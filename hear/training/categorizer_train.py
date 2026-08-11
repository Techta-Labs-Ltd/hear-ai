import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset

from hear.config import settings
from hear.models.database import CategoryTrainingExample, SessionLocal

logger = logging.getLogger(__name__)

MIN_EXAMPLES = 50
EMBEDDING_MODEL = settings.RESOLVER_SEMANTIC_MODEL
CHECKPOINT_DIR = settings.TRAINING_CHECKPOINT_DIR

_embedder = None


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


def load_training_examples(target: str = "category") -> tuple[list[str], list]:
    db = SessionLocal()
    try:
        if target == "category":
            rows = (
                db.query(CategoryTrainingExample)
                .filter(CategoryTrainingExample.category.isnot(None))
                .all()
            )
            return [r.text for r in rows], [r.category for r in rows]
        if target == "harm":
            rows = (
                db.query(CategoryTrainingExample)
                .filter(CategoryTrainingExample.label.in_(["harmful", "safe"]))
                .all()
            )
            return [r.text for r in rows], [r.label for r in rows]
        rows = db.query(CategoryTrainingExample).filter(CategoryTrainingExample.tags.isnot(None)).all()
        texts, tag_lists = [], []
        for r in rows:
            if r.tags:
                texts.append(r.text)
                tag_lists.append(list(r.tags))
        return texts, tag_lists
    finally:
        db.close()


class _EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, targets: np.ndarray):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.targets = torch.from_numpy(targets).float() if targets.ndim == 2 else torch.from_numpy(targets).long()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.targets[idx]


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _train_direct(config: dict) -> dict:
    texts = config["texts"]
    targets = np.array(config["targets"], dtype=np.float32 if config["multi_label"] else np.int64)
    num_classes = config["num_classes"]
    epochs = config["epochs"]
    multi_label = config["multi_label"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = _get_embedder()
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)

    dataset = _EmbeddingDataset(embeddings, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ClassifierHead(embeddings.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info("train %s epoch=%d loss=%.4f", config.get("checkpoint_name", ""), epoch + 1, total_loss / max(len(loader), 1))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{config['checkpoint_name']}.pt")
    temporary_path = f"{checkpoint_path}.{os.getpid()}.tmp"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_names": config["label_names"],
            "embedding_model": EMBEDDING_MODEL,
            "in_dim": embeddings.shape[1],
            "multi_label": multi_label,
        },
        temporary_path,
    )
    os.replace(temporary_path, checkpoint_path)

    eval_metrics = {}
    eval_texts = config.get("eval_texts", [])
    eval_targets_raw = config.get("eval_targets", [])
    if eval_texts and eval_targets_raw:
        eval_targets_arr = np.array(eval_targets_raw, dtype=np.float32 if multi_label else np.int64)
        eval_emb = embedder.encode(eval_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
        eval_loader = DataLoader(_EmbeddingDataset(eval_emb, eval_targets_arr), batch_size=32, shuffle=False)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in eval_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                if multi_label:
                    all_preds.append((torch.sigmoid(logits) > 0.5).int().cpu().numpy())
                else:
                    all_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                all_labels.append(batch_y.cpu().numpy() if not multi_label else batch_y.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        if multi_label:
            eval_metrics = {
                "eval_accuracy": float(accuracy_score(y_true.ravel(), y_pred.ravel())),
                "eval_precision_micro": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
                "eval_recall_micro": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
                "eval_f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            }
        else:
            eval_metrics = {
                "eval_accuracy": float(accuracy_score(y_true, y_pred)),
                "eval_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            }
    return eval_metrics


def train_categorizer(target: str = "category", epochs: int = 20) -> dict:
    if target not in ("category", "tags", "harm"):
        raise ValueError("target must be 'category', 'tags', or 'harm'")
    multi_label = target == "tags"

    texts, labels = load_training_examples(target)
    if len(texts) < MIN_EXAMPLES:
        return {"target": target, "error": f"only {len(texts)} examples (need >= {MIN_EXAMPLES})"}

    zipped = list(zip(texts, labels))
    rng = np.random.RandomState(42)
    rng.shuffle(zipped)
    split = int(len(zipped) * 0.8)
    train_pairs = zipped[:split]
    eval_pairs = zipped[split:]
    train_texts, train_labels_raw = zip(*train_pairs) if train_pairs else ([], [])
    eval_texts, eval_labels_raw = zip(*eval_pairs) if eval_pairs else ([], [])
    train_texts, train_labels_raw = list(train_texts), list(train_labels_raw)
    eval_texts, eval_labels_raw = list(eval_texts), list(eval_labels_raw)

    if multi_label:
        label_names = sorted({tag for tags in train_labels_raw for tag in tags})
        targets = np.zeros((len(train_labels_raw), len(label_names)), dtype=np.float32)
        for i, tags in enumerate(train_labels_raw):
            for tag in tags:
                targets[i, label_names.index(tag)] = 1.0
        eval_targets = np.zeros((len(eval_labels_raw), len(label_names)), dtype=np.float32)
        for i, tags in enumerate(eval_labels_raw):
            for tag in tags:
                if tag in label_names:
                    eval_targets[i, label_names.index(tag)] = 1.0
    else:
        label_names = sorted(set(train_labels_raw))
        targets = np.array([label_names.index(l) for l in train_labels_raw], dtype=np.int64)
        eval_targets = np.array([label_names.index(l) if l in label_names else 0 for l in eval_labels_raw], dtype=np.int64)

    eval_metrics = _train_direct({
        "texts": train_texts,
        "targets": targets.tolist(),
        "label_names": label_names,
        "num_classes": len(label_names),
        "epochs": epochs,
        "multi_label": multi_label,
        "checkpoint_name": f"{target}_classifier",
        "eval_texts": eval_texts,
        "eval_targets": eval_targets.tolist() if hasattr(eval_targets, "tolist") and eval_targets.size > 0 else [],
    })

    return {
        "target": target,
        "eval_metrics": eval_metrics,
        "num_train": len(train_texts),
        "num_eval": len(eval_texts),
        "num_classes": len(label_names),
    }


def ray_train_categorizer(target: str = "category", epochs: int = 20) -> dict:
    """Run one classifier update as an isolated, cluster-deduplicated Ray Train worker."""
    from ray import train
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    from sqlalchemy import text

    from hear.models.database import get_engine

    texts, _ = load_training_examples(target)
    if len(texts) < MIN_EXAMPLES:
        return {"target": target, "error": f"only {len(texts)} examples (need >= {MIN_EXAMPLES})"}

    lock_name = f"hear-ai-training-{target}"
    connection = get_engine().connect()
    locked = bool(
        connection.execute(
            text("SELECT pg_try_advisory_lock(hashtext(:name))"), {"name": lock_name}
        ).scalar()
    )
    if not locked:
        connection.close()
        return {"target": target, "error": "training already in progress"}

    def train_loop(config: dict) -> None:
        result = train_categorizer(target=config["target"], epochs=config["epochs"])
        train.report({"result_json": json.dumps(result, default=str)})

    try:
        trainer = TorchTrainer(
            train_loop_per_worker=train_loop,
            train_loop_config={"target": target, "epochs": epochs},
            scaling_config=ScalingConfig(
                num_workers=1,
                use_gpu=False,
                resources_per_worker={"CPU": 1},
            ),
        )
        metrics = trainer.fit().metrics
        payload = metrics.get("result_json", "{}")
        return json.loads(payload) if isinstance(payload, str) else dict(payload)
    finally:
        connection.execute(
            text("SELECT pg_advisory_unlock(hashtext(:name))"), {"name": lock_name}
        )
        connection.close()


def train_all_targets(epochs: int = 20) -> dict:
    results = {}
    for t in ("harm", "category", "tags"):
        results[t] = train_categorizer(t, epochs)
    return results


if __name__ == "__main__":
    start = time.time()
    _target = sys.argv[1] if len(sys.argv) > 1 else None
    if _target:
        print(json.dumps(train_categorizer(target=_target), default=str))
    else:
        print(json.dumps(train_all_targets(), default=str))
    print(f"done in {time.time() - start:.1f}s", file=sys.stderr)
