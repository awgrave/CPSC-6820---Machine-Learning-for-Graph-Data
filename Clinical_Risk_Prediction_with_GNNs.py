#Aidan Graves
#3/25/26
#CPSC 6820 - Machine Learning for Graph Data
"""
Builds patient features from MIMIC-III database, creates a kNN patient graph, and compares LR vs GNN risk models.
Goal: Improve patient diagnosis and risk prediction accuracy by leveraging patient similarity graphs.
"""

#IMPORT LIBRARIES
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.transforms import RandomNodeSplit


#CONFIG ----------------------------------------------------------------------------------------------------------------------------------------------------------------
DATA_DIR = "mimic-iii-clinical-database-1.4"

#outputs
FEATURES_PKL = "patient_features.pkl"
GRAPH_PT = "patient_graph_data.pt"
EVAL_ARRAYS_NPZ = "evaluation_arrays.npz"
SPLIT_SUMMARY_CSV = "split_summary.csv"
TABLE_ONE_CSV = "table_one_demographics.csv"

#cohort + chunking
N_PATIENTS = 35000
RANDOM_SEED = 45
CHUNKSIZE_EVENTS = 200_000
CHUNKSIZE_DIAGNOSES = 200_000

#tunable to help control processing time
MAX_CHARTEVENTS_CHUNKS: Optional[int] = None
MAX_LABEVENTS_CHUNKS: Optional[int] = None
MAX_DIAGNOSES_CHUNKS: Optional[int] = None

#params
K_NEIGHBORS = 10
HIDDEN_DIM = 64
LR = 0.01
EPOCHS = 300

#itemids - from D_ITEMS
HR_ITEMIDS = [211, 220045]
SBP_ITEMIDS = [51, 442, 455, 220050]
DBP_ITEMIDS = [8368, 8440, 8441, 8555, 220051]
TEMP_ITEMIDS = [223761, 223762]
RR_ITEMIDS = [618, 615, 220210]
SPO2_ITEMIDS = [646, 220277]

#lab events IDs - from D_LABITEMS
SODIUM_ITEMIDS = [50824]
CREATININE_ITEMIDS = [50912]
WBC_ITEMIDS = [51300, 51301]

FEATURE_COLS = [
    "age",
    "gender",
    "ethnicity",
    "hr_mean",
    "sbp_mean",
    "dbp_mean",
    "temp_mean",
    "rr_mean",
    "spo2_mean",
    "sodium",
    "creatinine",
    "wbc",
    "diag_count",
]


#UTILS ----------------------------------------------------------------------------------------------------------------------------------------------------------------
#safe divide - avoid division by 0
def safe_divide(sum_arr: np.ndarray, count_arr: np.ndarray) -> np.ndarray:
    """Elementwise divide with NaN for zero-count entries"""
    out = np.full(sum_arr.shape, np.nan, dtype=float)
    mask = count_arr > 0
    out[mask] = sum_arr[mask] / count_arr[mask]
    return out

def save_confusion_matrix(y_true, y_pred, title: str, out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def save_feature_corr_heatmap(x_tensor: torch.Tensor, feature_names: List[str], out_path: str) -> None:
    X = x_tensor.detach().cpu().numpy()
    corr = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        xticklabels=feature_names,
        yticklabels=feature_names
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def report_elapsed(step_name: str, start_t: float) -> float:
    elapsed = time.perf_counter() - start_t
    print(f"{step_name}: {elapsed:.2f} seconds")
    return elapsed

def logistic_feature_importance(
    lr_model: LogisticRegression,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 3,
) -> pd.DataFrame:
    """Estimates LR feat importance via permutation AUC drop"""

    coef = lr_model.coef_[0]
    coef_abs = np.abs(coef)

    baseline_proba = lr_model.predict_proba(X_test)[:, 1]
    baseline_auc = roc_auc_score(y_test, baseline_proba)

    perm_drops = []
    rng = np.random.default_rng(RANDOM_SEED)
    for j in range(X_test.shape[1]):
        drops_j = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            auc_perm = roc_auc_score(y_test, lr_model.predict_proba(X_perm)[:, 1])
            drops_j.append(baseline_auc - auc_perm)
        perm_drops.append(float(np.mean(drops_j)))

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "lr_coef": coef,
            "lr_coef_abs": coef_abs,
            "perm_auc_drop": perm_drops,
        }
    )
    imp_df = imp_df.sort_values("perm_auc_drop", ascending=False).reset_index(drop=True)
    return imp_df

def save_feature_importance_plot(imp_df: pd.DataFrame, out_path: str) -> None:
    top_df = imp_df.sort_values("perm_auc_drop", ascending=False)
    plt.figure(figsize=(8, max(5, 0.35 * len(top_df))))
    sns.barplot(data=top_df, x="perm_auc_drop", y="feature", orient="h")
    plt.title("Feature Importance (Permutation AUC Drop, LR Baseline)")
    plt.xlabel("AUC drop when feature is permuted")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def compute_class_weights_from_train(y_tensor: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    '''Computes inverse-frequency class weights from training labels only
    returns weights for classes [0, 1] for nn.CrossEntropyLoss'''
    y_train = y_tensor[train_mask].detach().cpu().numpy().astype(int)
    counts = np.bincount(y_train, minlength=2).astype(float)
    counts = np.maximum(counts, 1.0)  #for edge cases
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float)

def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Computes key binary classification metrics for imbalanced data settings"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
    }

def save_roc_curve(y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    for model_name, y_prob in prob_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def save_precision_recall_curve(y_true: np.ndarray, prob_dict: Dict[str, np.ndarray], out_path: str) -> None:
    plt.figure(figsize=(6, 5))
    base_rate = float(np.mean(y_true))
    plt.hlines(base_rate, 0, 1, colors="k", linestyles="--", linewidth=1, label=f"Base rate={base_rate:.3f}")
    for model_name, y_prob in prob_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def summarize_splits_and_table_one(raw_features_df: pd.DataFrame, data: Data) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create split outcome summary and simple Table One demographics summary"""
    split_masks = {
        "train": data.train_mask.detach().cpu().numpy().astype(bool),
        "val": data.val_mask.detach().cpu().numpy().astype(bool),
        "test": data.test_mask.detach().cpu().numpy().astype(bool),
        "all": np.ones(data.num_nodes, dtype=bool),
    }

    split_rows = []
    for split_name, mask in split_masks.items():
        split_df = raw_features_df.loc[mask]
        n = len(split_df)
        deaths = int(split_df["label"].sum())
        split_rows.append(
            {
                "split": split_name,
                "n_patients": n,
                "n_mortality": deaths,
                "mortality_rate": deaths / n if n else np.nan,
            }
        )
    split_summary = pd.DataFrame(split_rows)

    table_rows = []
    for split_name, mask in split_masks.items():
        split_df = raw_features_df.loc[mask]
        n = len(split_df)
        row = {
            "split": split_name,
            "n_patients": n,
            "age_mean": float(split_df["age"].mean()) if n else np.nan,
            "age_std": float(split_df["age"].std()) if n else np.nan,
            "male_pct": float((split_df["gender"] == 1).mean()) if n else np.nan,
            "ethnicity_white_pct": float((split_df["ethnicity"] == 0).mean()) if n else np.nan,
            "ethnicity_black_pct": float((split_df["ethnicity"] == 1).mean()) if n else np.nan,
            "ethnicity_hispanic_pct": float((split_df["ethnicity"] == 2).mean()) if n else np.nan,
            "ethnicity_asian_pct": float((split_df["ethnicity"] == 3).mean()) if n else np.nan,
            "ethnicity_other_pct": float((split_df["ethnicity"] == 4).mean()) if n else np.nan,
        }
        table_rows.append(row)
    table_one = pd.DataFrame(table_rows)
    return split_summary, table_one

#FEATURE EXTRACTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_cohort(admissions: pd.DataFrame, patients: pd.DataFrame, n_patients: int, seed: int) -> pd.DataFrame:
    """Select first admission per patient and derive static demographics + outcome label."""
    merged = admissions.merge(patients, on="SUBJECT_ID", suffixes=("", "_PAT"))
    merged = merged.sort_values(["SUBJECT_ID", "ADMITTIME"])
    first_adm = merged.groupby("SUBJECT_ID", as_index=False).head(1).copy()

    dob_dt = pd.to_datetime(first_adm["DOB"], errors="coerce")
    adm_dt = pd.to_datetime(first_adm["ADMITTIME"], errors="coerce")
    first_adm["age"] = (adm_dt - dob_dt).dt.days / 365.25

    first_adm["gender"] = first_adm["GENDER"].map({"M": 1.0, "F": 0.0})

    # WHITE=0, BLACK=1, HISPANIC=2, ASIAN=3, OTHER=4
    eth = first_adm["ETHNICITY"].astype(str).str.upper()

    def _eth_group(s: str) -> str:
        if "WHITE" in s:
            return "WHITE"
        if "BLACK" in s or "AFRICAN" in s:
            return "BLACK"
        if "HISPANIC" in s or "LATINO" in s:
            return "HISPANIC"
        if "ASIAN" in s:
            return "ASIAN"
        return "OTHER"

    eth_group = eth.map(_eth_group)
    eth_map = {"WHITE": 0.0, "BLACK": 1.0, "HISPANIC": 2.0, "ASIAN": 3.0, "OTHER": 4.0}
    first_adm["ethnicity"] = eth_group.map(eth_map).astype(float)

    #target label
    first_adm["label"] = first_adm["HOSPITAL_EXPIRE_FLAG"].astype(int)

    cohort = first_adm[["SUBJECT_ID", "HADM_ID", "age", "gender", "ethnicity", "label"]].copy()
    cohort = cohort.rename(columns={"SUBJECT_ID": "patient_id", "HADM_ID": "hadm_id"})
    cohort = cohort.sample(n=min(n_patients, len(cohort)), random_state=seed).reset_index(drop=True)
    return cohort

#accumulates means - from CHARTEVENTS
def accumulate_means_from_events(
    path: str,
    hadm_ids_set: set,
    hadm_to_idx: Dict[int, int],
    specs: List[Tuple[str, List[int]]],
    chunksize: int,
    max_chunks: Optional[int],
) -> Dict[str, np.ndarray]:
    """Chunk through events/labs table and compute per-admission means for selected ITEMIDs."""
    n = len(hadm_to_idx)
    sum_by_feat: Dict[str, np.ndarray] = {name: np.zeros(n, dtype=float) for name, _ in specs}
    cnt_by_feat: Dict[str, np.ndarray] = {name: np.zeros(n, dtype=int) for name, _ in specs}

    usecols = ["HADM_ID", "ITEMID", "VALUENUM"]
    reader = pd.read_csv(path, chunksize=chunksize, usecols=usecols)

    for chunk_idx, chunk in enumerate(reader):
        if max_chunks is not None and chunk_idx >= max_chunks:
            print(f"  Max Chunks:{max_chunks}")
            break

        chunk = chunk[chunk["HADM_ID"].isin(hadm_ids_set)]
        if chunk.empty:
            continue

        chunk["VALUENUM"] = pd.to_numeric(chunk["VALUENUM"], errors="coerce")
        chunk = chunk.dropna(subset=["VALUENUM"])
        if chunk.empty:
            continue

        for feat_name, itemids in specs:
            sub = chunk[chunk["ITEMID"].isin(itemids)]
            if sub.empty:
                continue

            grp = sub.groupby("HADM_ID")["VALUENUM"].agg(["sum", "count"]).reset_index()
            idxs = np.fromiter((hadm_to_idx[int(h)] for h in grp["HADM_ID"].values), dtype=int)
            sum_by_feat[feat_name][idxs] += grp["sum"].values
            cnt_by_feat[feat_name][idxs] += grp["count"].values.astype(int)

        if chunk_idx % 10 == 0:
            print(f"  Processed chunk {chunk_idx}")

    means = {feat_name: safe_divide(sum_arr, cnt_by_feat[feat_name]) for feat_name, sum_arr in sum_by_feat.items()}
    return means

#accumulates diagnosis counts - from DIAGNOSES_ICD
def accumulate_diag_counts(
    path: str,
    hadm_ids_set: set,
    hadm_to_idx: Dict[int, int],
    chunksize: int,
    max_chunks: Optional[int],
) -> np.ndarray:
    """Chunk through diagnosis table and count unique ICD codes per admission."""
    #count unique ICD9_CODE per HADM_ID
    n = len(hadm_to_idx)
    code_sets: Dict[int, set] = {int(h): set() for h in hadm_to_idx.keys()}

    usecols = ["HADM_ID", "ICD9_CODE"]
    reader = pd.read_csv(path, chunksize=chunksize, usecols=usecols)

    for chunk_idx, chunk in enumerate(reader):
        if max_chunks is not None and chunk_idx >= max_chunks:
            print(f" Max Chunks: {max_chunks}")
            break

        chunk = chunk[chunk["HADM_ID"].isin(hadm_ids_set)]
        if chunk.empty:
            continue

        chunk["ICD9_CODE"] = chunk["ICD9_CODE"].astype(str)
        for hadm_id, grp in chunk.groupby("HADM_ID")["ICD9_CODE"]:
            code_sets[int(hadm_id)].update(c for c in grp.values if c and c != "nan")

        if chunk_idx % 10 == 0:
            print(f"  Processed chunk {chunk_idx}")

    diag_count = np.zeros(n, dtype=int)
    for hadm_id, idx in hadm_to_idx.items():
        diag_count[idx] = len(code_sets[int(hadm_id)])
    return diag_count

def extract_features() -> Dict:
    """Build feat mat and labels from core demographics, vitals, labs, and diagnosis"""
    t0 = time.perf_counter()
    print("Loading core tables...")
    admissions = pd.read_csv(os.path.join(DATA_DIR, "ADMISSIONS.csv"))
    patients = pd.read_csv(os.path.join(DATA_DIR, "PATIENTS.csv"))
    report_elapsed("Load ADMISSIONS/PATIENTS", t0)

    t1 = time.perf_counter()
    cohort = build_cohort(admissions, patients, N_PATIENTS, RANDOM_SEED)
    hadm_ids = cohort["hadm_id"].astype(int).tolist()
    hadm_ids_set = set(hadm_ids)
    hadm_to_idx = {h: i for i, h in enumerate(hadm_ids)}

    print(f"Cohort size: {len(cohort)} patients")
    report_elapsed("Build cohort", t1)

    # CHARTEVENTS (chunked)
    t2 = time.perf_counter()
    print("Extracting vitals from CHARTEVENTS...")
    vitals_specs = [
        ("hr_mean", HR_ITEMIDS),
        ("sbp_mean", SBP_ITEMIDS),
        ("dbp_mean", DBP_ITEMIDS),
        ("temp_mean", TEMP_ITEMIDS),
        ("rr_mean", RR_ITEMIDS),
        ("spo2_mean", SPO2_ITEMIDS),
    ]
    vitals_means = accumulate_means_from_events(
        path=os.path.join(DATA_DIR, "CHARTEVENTS.csv"),
        hadm_ids_set=hadm_ids_set,
        hadm_to_idx=hadm_to_idx,
        specs=vitals_specs,
        chunksize=CHUNKSIZE_EVENTS,
        max_chunks=MAX_CHARTEVENTS_CHUNKS,
    )
    report_elapsed("Extract vitals (CHARTEVENTS)", t2)

    # LABEVENTS (chunked)
    t3 = time.perf_counter()
    print("Extracting labs from LABEVENTS...")
    labs_specs = [
        ("sodium", SODIUM_ITEMIDS),
        ("creatinine", CREATININE_ITEMIDS),
        ("wbc", WBC_ITEMIDS),
    ]
    labs_means = accumulate_means_from_events(
        path=os.path.join(DATA_DIR, "LABEVENTS.csv"),
        hadm_ids_set=hadm_ids_set,
        hadm_to_idx=hadm_to_idx,
        specs=labs_specs,
        chunksize=CHUNKSIZE_EVENTS,
        max_chunks=MAX_LABEVENTS_CHUNKS,
    )
    report_elapsed("Extract labs (LABEVENTS)", t3)

    # DIAGNOSES_ICD (chunked)
    t4 = time.perf_counter()
    print("Extracting diagnosis count from DIAGNOSES_ICD...")
    diag_count = accumulate_diag_counts(
        path=os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv"),
        hadm_ids_set=hadm_ids_set,
        hadm_to_idx=hadm_to_idx,
        chunksize=CHUNKSIZE_DIAGNOSES,
        max_chunks=MAX_DIAGNOSES_CHUNKS,
    )
    report_elapsed("Extract diag_count (DIAGNOSES_ICD)", t4)

    t5 = time.perf_counter()
    features_df = cohort.copy()
    features_df["diag_count"] = diag_count
    for feat_name, _ in vitals_specs:
        features_df[feat_name] = vitals_means[feat_name]
    for feat_name, _ in labs_specs:
        features_df[feat_name] = labs_means[feat_name]

    X_df = features_df[FEATURE_COLS].copy()

    #drop all NaN cols if any
    nonempty_mask = X_df.notna().any(axis=0)
    feature_names = X_df.columns[nonempty_mask].tolist()
    X_df = X_df.loc[:, feature_names]

    X = X_df.to_numpy(dtype=float)
    y = features_df["label"].to_numpy(dtype=int)
    patient_ids = features_df["patient_id"].tolist()

    print(f"Raw feature matrix shape: {X.shape}")

    #impute + normalize
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_imputed)

    print(f"Normalized feature matrix shape: {X_normalized.shape}")
    print(f"Missing values after imputation: {np.isnan(X_normalized).sum()}")
    report_elapsed("Assemble/impute/normalize features", t5)

    t6 = time.perf_counter()
    data_dict = {
        "X": X_normalized,
        "y": y,
        "patient_ids": patient_ids,
        "feature_names": feature_names,
        "imputer": imputer,
        "scaler": scaler,
        "raw_features_df": features_df[["patient_id", "hadm_id", "label"] + feature_names].copy(),
    }

    with open(FEATURES_PKL, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Saved extracted features to {FEATURES_PKL}")
    report_elapsed("Save feature pickle", t6)

    return data_dict

#GRAPH BUILDING ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def build_graph_from_features(X: np.ndarray, y: np.ndarray, k: int = 10) -> Data:
    """Construct directed kNN patient graph and add train/val/test node masks"""
    print("\nBuilding similarity graph...")
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    distances, indices = nbrs.kneighbors(X)

    edges = []
    for i in range(len(indices)):
        for j in indices[i][1:]:
            edges.append([i, j])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y_tensor)
    data = RandomNodeSplit(num_val=0.15, num_test=0.15)(data)

    torch.save(data, GRAPH_PT)
    print(f"Saved graph data to {GRAPH_PT}")
    print(f"Graph: nodes={data.num_nodes}, edges={data.num_edges}, features={data.num_node_features}")

    return data

#MODELS ----------------------------------------------------------------------------------------------------------------------------------------------------------------
class GCN(nn.Module):
    """2-layer GCN"""
    def __init__(self, num_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    """2-layer GraphSAGE."""
    def __init__(self, num_features: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def train_eval_gnn(
    model,
    data,
    epochs: int,
    lr: float,
    class_weights: Optional[torch.Tensor] = None,
):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(data.x.device))
    else:
        criterion = nn.CrossEntropyLoss()

    def train_one():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate():
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            proba = F.softmax(out, dim=1)[:, 1]

            train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum() / data.train_mask.sum()
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum() / data.val_mask.sum()
            test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum() / data.test_mask.sum()

            y_test_proba = proba[data.test_mask].cpu().numpy()
            y_test_true = data.y[data.test_mask].cpu().numpy()
            y_test_pred = pred[data.test_mask].cpu().numpy()
            test_auc = roc_auc_score(y_test_true, y_test_proba)

        return (
            train_acc.item(),
            val_acc.item(),
            test_acc.item(),
            test_auc,
            y_test_true,
            y_test_pred,
            y_test_proba,
        )

    for epoch in range(epochs):
        loss = train_one()
        if epoch % 40 == 0:
            tr, va, te, auc, _, _, _ = evaluate()
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Train: {tr:.4f} | Val: {va:.4f} | Test: {te:.4f} | Test AUC: {auc:.4f}")

    return evaluate()

#MAIN FUNCTION ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def main():
    """feat extraction, graph construction, training, and reporting"""
    runtime_log = []
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    #feature extraction
    t = time.perf_counter()
    feat_data = extract_features()
    runtime_log.append(("Feature extraction total", report_elapsed("Feature extraction total", t)))
    X = feat_data["X"]
    y = feat_data["y"]
    feature_names = feat_data["feature_names"]
    raw_features_df = feat_data["raw_features_df"]

    #graph building
    t = time.perf_counter()
    data = build_graph_from_features(X, y, k=K_NEIGHBORS)
    runtime_log.append(("Graph build total", report_elapsed("Graph build total", t)))

    # Compute train-split class weights once for weighted GNN loss
    class_weights = compute_class_weights_from_train(data.y, data.train_mask)
    print(f"Class weights [w0, w1]: {class_weights.tolist()}")

    print("\nData splits:")
    print(f"  Train: {data.train_mask.sum().item()} nodes")
    print(f"  Val:   {data.val_mask.sum().item()} nodes")
    print(f"  Test:  {data.test_mask.sum().item()} nodes")
    print("\nNote on time cutoff for mortality:")
    print(
        "Current implementation does NOT apply a pre-mortality censoring window (t_TOD - x)."
        "Features aggregate available events for the selected admission."
        "If you need strict prospective setup, add an event-time filter before feature aggregation."
    )

    #demographics and class balance summary by split
    split_summary_df, table_one_df = summarize_splits_and_table_one(raw_features_df, data)
    split_summary_df.to_csv(SPLIT_SUMMARY_CSV, index=False)
    table_one_df.to_csv(TABLE_ONE_CSV, index=False)
    print(f"\nSaved split outcome summary: {SPLIT_SUMMARY_CSV}")
    print(split_summary_df.to_string(index=False))
    print(f"\nSaved demographics Table One summary: {TABLE_ONE_CSV}")
    print(table_one_df.to_string(index=False))

    #baseline LogReg
    print("\n" + "=" * 60)
    print("BASELINE: Logistic Regression")
    print("=" * 60)

    t = time.perf_counter()
    X_train = data.x[data.train_mask].cpu().numpy()
    y_train = data.y[data.train_mask].cpu().numpy()
    X_test = data.x[data.test_mask].cpu().numpy()
    y_test = data.y[data.test_mask].cpu().numpy()

    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

    baseline_acc = accuracy_score(y_test, y_pred_lr)
    baseline_auc = roc_auc_score(y_test, y_proba_lr)
    baseline_precision = precision_score(y_test, y_pred_lr, zero_division=0)
    baseline_recall = recall_score(y_test, y_pred_lr, zero_division=0)
    runtime_log.append(("Baseline LR train/eval", report_elapsed("Baseline LR train/eval", t)))

    print(f"Accuracy: {baseline_acc:.4f}")
    print(f"ROC-AUC:  {baseline_auc:.4f}")
    print(f"Precision:{baseline_precision:.4f}")
    print(f"Recall:   {baseline_recall:.4f}")

    save_confusion_matrix(y_test, y_pred_lr, "Confusion Matrix - Logistic Regression", "confusion_matrix_lr.png")
    save_feature_corr_heatmap(data.x, feature_names, "feature_corr_heatmap.png")

    # feat importance for LR baseline
    t = time.perf_counter()
    imp_df = logistic_feature_importance(lr_model, X_test, y_test, feature_names, n_repeats=3)
    imp_df.to_csv("feature_importance_lr.csv", index=False)
    save_feature_importance_plot(imp_df, "feature_importance_lr.png")
    print("\nTop LR features by permutation AUC drop:")
    print(imp_df[["feature", "perm_auc_drop", "lr_coef", "lr_coef_abs"]].head(10).to_string(index=False))
    runtime_log.append(("Feature importance (LR)", report_elapsed("Feature importance (LR)", t)))

    #GCN
    print("\n" + "=" * 60)
    print("GCN MODEL")
    print("=" * 60)

    t = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_device = data.to(device)
    gcn = GCN(num_features=data.num_node_features, hidden_dim=HIDDEN_DIM, num_classes=2).to(device)
    gcn_tr, gcn_va, gcn_te, gcn_auc, gcn_true, gcn_pred, gcn_proba = train_eval_gnn(
        gcn,
        data_device,
        EPOCHS,
        LR,
        class_weights=class_weights,
    )
    runtime_log.append(("GCN train/eval", report_elapsed("GCN train/eval", t)))

    print("\nGCN Final:")
    print(f"  Test Accuracy: {gcn_te:.4f}")
    print(f"  Test ROC-AUC:  {gcn_auc:.4f}")
    print(f"  Test Precision:{precision_score(gcn_true, gcn_pred, zero_division=0):.4f}")
    print(f"  Test Recall:   {recall_score(gcn_true, gcn_pred, zero_division=0):.4f}")
    save_confusion_matrix(gcn_true, gcn_pred, "Confusion Matrix - GCN", "confusion_matrix_gcn.png")

    #GraphSAGE
    print("\n" + "=" * 60)
    print("GRAPHSAGE MODEL")
    print("=" * 60)

    t = time.perf_counter()
    sage = GraphSAGE(num_features=data.num_node_features, hidden_dim=HIDDEN_DIM, num_classes=2).to(device)
    sage_tr, sage_va, sage_te, sage_auc, sage_true, sage_pred, sage_proba = train_eval_gnn(
        sage,
        data_device,
        EPOCHS,
        LR,
        class_weights=class_weights,
    )
    runtime_log.append(("GraphSAGE train/eval", report_elapsed("GraphSAGE train/eval", t)))

    print("\nGraphSAGE Final:")
    print(f"  Test Accuracy: {sage_te:.4f}")
    print(f"  Test ROC-AUC:  {sage_auc:.4f}")
    print(f"  Test Precision:{precision_score(sage_true, sage_pred, zero_division=0):.4f}")
    print(f"  Test Recall:   {recall_score(sage_true, sage_pred, zero_division=0):.4f}")
    save_confusion_matrix(sage_true, sage_pred, "Confusion Matrix - GraphSAGE", "confusion_matrix_sage.png")

    # Save ROC and precision-recall curves across all models
    model_probabilities = {
        "Logistic Regression": y_proba_lr,
        "GCN": gcn_proba,
        "GraphSAGE": sage_proba,
    }
    save_roc_curve(y_test, model_probabilities, "roc_curves_all_models.png")
    save_precision_recall_curve(y_test, model_probabilities, "precision_recall_curves_all_models.png")

    np.savez(
        EVAL_ARRAYS_NPZ,
        y_true=y_test.astype(np.int64),
        y_pred_lr=y_pred_lr.astype(np.int64),
        y_prob_lr=y_proba_lr.astype(np.float64),
        y_pred_gcn=gcn_pred.astype(np.int64),
        y_prob_gcn=gcn_proba.astype(np.float64),
        y_pred_sage=sage_pred.astype(np.int64),
        y_prob_sage=sage_proba.astype(np.float64),
        auc_lr=np.float64(baseline_auc),
        auc_gcn=np.float64(gcn_auc),
        auc_sage=np.float64(sage_auc),
        acc_lr=np.float64(baseline_acc),
        acc_gcn=np.float64(gcn_te),
        acc_sage=np.float64(sage_te),
    )
    print(f"Saved test predictions for figure generation: {EVAL_ARRAYS_NPZ}")

    summary = (
        "\n" + "=" * 60 + "\n"
        "SUMMARY\n"
        + "=" * 60 + "\n"
        f"Baseline (LR) - Accuracy: {baseline_acc:.4f}, AUC: {baseline_auc:.4f},"
        f" Precision: {baseline_precision:.4f}, Recall: {baseline_recall:.4f}\n"
        f"GCN           - Accuracy: {gcn_te:.4f}, AUC: {gcn_auc:.4f}, "
        f"Precision: {precision_score(gcn_true, gcn_pred, zero_division=0):.4f}, "
        f"Recall: {recall_score(gcn_true, gcn_pred, zero_division=0):.4f}\n"
        f"GraphSAGE     - Accuracy: {sage_te:.4f}, AUC: {sage_auc:.4f}, "
        f"Precision: {precision_score(sage_true, sage_pred, zero_division=0):.4f}, "
        f"Recall: {recall_score(sage_true, sage_pred, zero_division=0):.4f}\n"
    )
    print(summary)

    total_time = sum(sec for _, sec in runtime_log)
    lines = ["Runtime summary (s):"]
    for name, sec in runtime_log:
        lines.append(f"- {name}: {sec:.2f}")
    lines.append(f"- Total tracked runtime: {total_time:.2f}")
    timing_text = "\n".join(lines) + "\n"
    print("\n" + timing_text)

if __name__ == "__main__":
    main()
