"""Name That LLM - TF-IDF + LinearSVC baseline (Kaggle Notebook, no internet).

Data layout was not visible on the Overview page, so train file / text column /
label column are auto-detected. Check the printed detection result and
override TRAIN_PATH / TEXT_COL / LABEL_COL below if it guessed wrong.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import LinearSVC

DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/competitions/name-that-llm")
TRAIN_PATH = None   # e.g. f"{DATA_DIR}/train.csv"
TEXT_COL = None     # e.g. "text"
LABEL_COL = None    # e.g. "model"
GROUP_COL = None    # e.g. "prompt_id" if the train file has one

LABELS = {
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
}


def read_any(path):
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


def find_train():
    if TRAIN_PATH:
        return read_any(TRAIN_PATH), TRAIN_PATH
    for p in sorted(glob.glob(f"{DATA_DIR}/**/*.*", recursive=True)):
        if not p.endswith((".csv", ".parquet")) or os.path.basename(p).startswith(("test", "sample")):
            continue
        df = read_any(p)
        for c in df.columns:
            if df[c].isin(LABELS).mean() > 0.9:
                return df, p
    raise FileNotFoundError(
        f"No labelled train file found under {DATA_DIR}. "
        "Check the Data tab and set TRAIN_PATH / TEXT_COL / LABEL_COL.")


train, train_path = find_train()
label_col = LABEL_COL or next(c for c in train.columns if train[c].isin(LABELS).mean() > 0.9)
text_col = TEXT_COL or max(
    (c for c in train.columns if c != label_col and not pd.api.types.is_numeric_dtype(train[c])),
    key=lambda c: train[c].astype(str).str.len().mean())
group_col = GROUP_COL or next((c for c in train.columns if "prompt" in c.lower() and c != text_col), None)
print(f"train={train_path} rows={len(train)} text={text_col} label={label_col} group={group_col}")
print(train[label_col].value_counts().to_string())

test = read_any(f"{DATA_DIR}/test.csv") if os.path.exists(f"{DATA_DIR}/test.csv") else \
    read_any(glob.glob(f"{DATA_DIR}/test.*")[0])
test_text_col = text_col if text_col in test.columns else \
    max((c for c in test.columns if not pd.api.types.is_numeric_dtype(test[c]) and c != "id"),
        key=lambda c: test[c].astype(str).str.len().mean())

X_text = train[text_col].astype(str).values
y = train[label_col].values


def build_features(fit_text, *other):
    # case and punctuation are kept on purpose: they carry style signal
    char = TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False,
                           sublinear_tf=True, min_df=3, max_features=400_000)
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), lowercase=True,
                           token_pattern=r"(?u)\b\w+\b|[^\w\s]", sublinear_tf=True,
                           min_df=2, max_features=300_000)
    Xf = hstack([char.fit_transform(fit_text), word.fit_transform(fit_text)]).tocsr()
    others = [hstack([char.transform(t), word.transform(t)]).tocsr() for t in other]
    return (Xf, *others)


def make_model():
    return LinearSVC(C=0.5, class_weight="balanced")  # balanced -> matches per-class-recall metric


# ---- cross-validation (grouped by prompt if a prompt column exists) ----
if group_col:
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X_text, y, train[group_col])
else:
    splits = StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(X_text, y)
scores = []
for tr, va in splits:
    Xtr, Xva = build_features(X_text[tr], X_text[va])
    pred = make_model().fit(Xtr, y[tr]).predict(Xva)
    scores.append(balanced_accuracy_score(y[va], pred))
    print(f"fold balanced_acc={scores[-1]:.4f}")
print(f"CV balanced accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# ---- final fit + submission ----
Xtr, Xte = build_features(X_text, test[test_text_col].astype(str).values)
preds = make_model().fit(Xtr, y).predict(Xte)

sub = pd.DataFrame({"id": test["id"], "model": preds})
assert sub["id"].is_unique and sub["model"].isin(LABELS).all() and sub.notna().all().all()
sub.to_parquet("submission.parquet", index=False)
print(sub.head())
