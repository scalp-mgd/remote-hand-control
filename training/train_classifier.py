"""Train MLP gesture classifier and export to ONNX."""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dataset(path: str):
    """Load keypoints CSV → (X, y) arrays."""
    X, y = [], []
    with open(path, "r") as f:
        for row in csv.reader(f):
            if not row:
                continue
            y.append(int(row[0]))
            X.append([float(v) for v in row[1:]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    data_path = "models/keypoints.csv"
    model_path = "models/gesture_classifier.onnx"

    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Run collect_data.py first.")
        sys.exit(1)

    X, y = load_dataset(data_path)
    num_classes = int(y.max()) + 1
    print(f"Dataset: {len(X)} samples, {num_classes} classes")
    for c in range(num_classes):
        print(f"  Class {c}: {(y == c).sum()} samples")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    # Model
    model = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(42, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, num_classes),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # Training loop
    best_val_acc = 0.0
    best_state = None

    for epoch in range(200):
        model.train()
        # Mini-batch
        indices = torch.randperm(len(X_train_t))
        total_loss = 0.0
        for i in range(0, len(indices), 64):
            batch_idx = indices[i:i + 64]
            xb = X_train_t[batch_idx]
            yb = y_train_t[batch_idx]

            out = model(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_preds = val_out.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/200 — loss: {total_loss:.4f}, val_acc: {val_acc:.1%}")

    # Load best model
    model.load_state_dict(best_state)
    model.eval()
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")

    # Export to ONNX
    dummy_input = torch.randn(1, 42)
    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11,
    )
    print(f"Model exported to: {model_path}")

    # Final validation with ONNX
    import onnxruntime as ort
    session = ort.InferenceSession(model_path)
    onnx_out = session.run(None, {"input": X_val})[0]
    onnx_preds = onnx_out.argmax(axis=1)
    onnx_acc = (onnx_preds == y_val).mean()
    print(f"ONNX model validation accuracy: {onnx_acc:.1%}")


if __name__ == "__main__":
    main()
