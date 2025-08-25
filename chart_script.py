#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Confidence score evaluation for multi-label email classification (dummy dataset),
with fully labeled plots, non-interactive backend for autosaving, and a
report-style "8.2 Raw Metric Results" console block.

- Dataset: 100 emails, 3 labels, ~2 labels/email, 25% mismatches (default)
- Confidence: N(0.78, 0.12) if correct; N(0.42, 0.15) if incorrect; clipped [0.01, 0.99]
- Metrics: ECE, MCE, Brier, NLL, ROC-AUC, PR-AUC, Cohen's d, point-biserial r,
          Margin, Entropy, Mutual Information (ensembles), Coverage–Risk + AURC
- Calibration: Temperature scaling, Platt scaling, Isotonic regression
- Plots: KDE + Violin distributions, Reliability (raw & calibrated), ROC, PR,
         Risk–Coverage, Per-label calibration heatmap (all with labeled axes & titles)

Aligned with the attached report’s definitions and presentation. 
"""

from __future__ import annotations
import argparse
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

# Force a non-interactive backend so savefig works headless (CLI/servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid, logit
from scipy.stats import pointbiserialr
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    brier_score_loss,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

import seaborn as sns
sns.set(style="whitegrid")


# ---------- I/O helpers ----------

def ensure_outdir(outdir: str) -> str:
    out_abs = os.path.abspath(outdir)
    os.makedirs(out_abs, exist_ok=True)
    return out_abs

def save_fig(path: str, dpi: int = 180):
    path_abs = os.path.abspath(path)
    try:
        plt.tight_layout()
    except Exception:
        pass
    try:
        plt.savefig(path_abs, dpi=dpi, bbox_inches="tight")
        print(f"[saved] {path_abs}")
    except Exception as e:
        print(f"[save error] {path_abs}: {e}")
    finally:
        plt.close()


# ---------- Data generation ----------

def set_seed(seed: int):
    np.random.seed(seed)

def gen_multilabel_truth(n_samples: int, n_labels: int, avg_labels: float = 2.0) -> np.ndarray:
    # Approximate 2 of 3: 70% -> 2 labels, 15% -> 1 label, 15% -> 3 labels
    probs = np.array([0.15, 0.70, 0.15])
    sizes = np.array([
        max(1, int(round(avg_labels - 1))),
        int(round(avg_labels)),
        min(n_labels, int(round(avg_labels + 1))),
    ])
    sizes = np.clip(sizes, 1, n_labels)
    Z = np.zeros((n_samples, n_labels), dtype=int)
    for i in range(n_samples):
        k = np.random.choice(len(sizes), p=probs)
        s = int(np.clip(sizes[k], 1, n_labels))
        pos = np.random.choice(n_labels, size=s, replace=False)
        Z[i, pos] = 1
    return Z

def gen_predictions_with_mismatch(Y_true: np.ndarray, mismatch_rate: float = 0.25) -> np.ndarray:
    flips = (np.random.rand(*Y_true.shape) < mismatch_rate).astype(int)
    return (Y_true ^ flips).astype(int)

def synthesize_confidences(Y_true: np.ndarray, Y_pred: np.ndarray,
                           mu_correct: float = 0.78, sd_correct: float = 0.12,
                           mu_incorrect: float = 0.42, sd_incorrect: float = 0.15,
                           eps: float = 1e-3) -> np.ndarray:
    A = (Y_true == Y_pred).astype(int)
    C = np.empty_like(Y_true, dtype=float)
    mask_c = A == 1
    mask_i = ~mask_c
    C[mask_c] = np.random.normal(mu_correct, sd_correct, size=mask_c.sum())
    C[mask_i] = np.random.normal(mu_incorrect, sd_incorrect, size=mask_i.sum())
    C = np.clip(C, eps, 1 - eps)
    return C


# ---------- Metrics ----------

def ece_mce(c: np.ndarray, a: np.ndarray, n_bins: int = 10) -> Tuple[float, float, pd.DataFrame]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(c, bins) - 1
    ece = 0.0
    mce = 0.0
    rows = []
    N = len(c)
    for b in range(n_bins):
        idx = inds == b
        n_b = idx.sum()
        if n_b == 0:
            rows.append((b, bins[b], bins[b+1], 0, np.nan, np.nan, 0.0))
            continue
        conf_b = c[idx].mean()
        acc_b = a[idx].mean()
        gap = abs(acc_b - conf_b)
        ece += (n_b / N) * gap
        mce = max(mce, gap)
        rows.append((b, bins[b], bins[b+1], n_b, conf_b, acc_b, gap))
    df = pd.DataFrame(rows, columns=["bin", "lo", "hi", "count", "mean_conf", "mean_acc", "gap"])
    return float(ece), float(mce), df

def brier(c: np.ndarray, a: np.ndarray) -> float:
    return float(brier_score_loss(a, c))

def nll(c: np.ndarray, a: np.ndarray, eps: float = 1e-12) -> float:
    c = np.clip(c, eps, 1 - eps)
    return float(log_loss(a, c))

def roc_pr_auc(c: np.ndarray, a: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    roc = roc_auc_score(a, c)
    prec, rec, thr = precision_recall_curve(a, c)
    pr = auc(rec, prec)
    return float(roc), float(pr), prec, rec, thr

def cohens_d(c: np.ndarray, a: np.ndarray) -> float:
    c1 = c[a == 1]
    c0 = c[a == 0]
    if len(c1) < 2 or len(c0) < 2:
        return 0.0
    m1, m0 = c1.mean(), c0.mean()
    s1, s0 = c1.std(ddof=1), c0.std(ddof=1)
    sp = math.sqrt((s1**2 + s0**2) / 2.0)
    if sp == 0:
        return 0.0
    return float((m1 - m0) / sp)

def point_biserial(c: np.ndarray, a: np.ndarray) -> float:
    r, _ = pointbiserialr(a, c)
    return float(r)

def margin_per_sample(C: np.ndarray) -> np.ndarray:
    sorted_C = np.sort(C, axis=1)[:, ::-1]
    if C.shape[1] == 1:
        return np.zeros(C.shape)
    return (sorted_C[:, 0] - sorted_C[:, 1])

def entropy_per_sample(C: np.ndarray, normalize: str = "sum") -> np.ndarray:
    eps = 1e-12
    if normalize == "softmax":
        S = np.exp((C - C.max(axis=1, keepdims=True)))
        P = S / (S.sum(axis=1, keepdims=True) + eps)
    else:
        P = C / (C.sum(axis=1, keepdims=True) + eps)
    H = -(P * np.log(P + eps)).sum(axis=1)
    return H

def ensemble_mi(C: np.ndarray, n_ensembles: int = 8, noise_sd: float = 0.05) -> np.ndarray:
    eps = 1e-12
    Ps = []
    for _ in range(n_ensembles):
        noisy = np.clip(C + np.random.normal(0, noise_sd, size=C.shape), eps, 1 - eps)
        P = noisy / (noisy.sum(axis=1, keepdims=True) + eps)
        Ps.append(P)
    Ps = np.stack(Ps, axis=0)   # (E, N, K)
    P_bar = Ps.mean(axis=0)     # (N, K)
    H_bar = -(Ps * np.log(Ps + eps)).sum(axis=2).mean(axis=0)   # mean H[p] over ensembles per sample
    H_of_bar = -(P_bar * np.log(P_bar + eps)).sum(axis=1)       # H[E p] per sample
    MI = H_of_bar - H_bar
    return MI

def risk_coverage(c: np.ndarray, a: np.ndarray, n_points: int = 200) -> Tuple[pd.DataFrame, float]:
    taus = np.linspace(0.0, 1.0, n_points)
    rows = []
    for t in taus:
        idx = c >= t
        cov = idx.mean()
        risk = np.nan if idx.sum() == 0 else 1.0 - a[idx].mean()
        rows.append((t, cov, risk))
    df = pd.DataFrame(rows, columns=["tau", "coverage", "risk"]).dropna()
    if df.empty:
        return df, float("nan")
    df = df.sort_values("coverage")
    aurc = np.trapz(df["risk"].values, df["coverage"].values)
    return df, float(aurc)

def cov_risk_at(c: np.ndarray, a: np.ndarray, tau: float) -> Tuple[float, float]:
    idx = c >= tau
    cov = idx.mean()
    risk = np.nan if idx.sum() == 0 else 1.0 - a[idx].mean()
    return float(cov), float(risk)


# ---------- Calibrators ----------

@dataclass
class Calibrator:
    name: str
    fitted_: bool = False
    T_: Optional[float] = None
    lr_: Optional[LogisticRegression] = None
    iso_: Optional[IsotonicRegression] = None

    def fit(self, c_train: np.ndarray, a_train: np.ndarray, method: str = "temperature"):
        method = method.lower()
        if method == "temperature":
            Ts = np.linspace(0.5, 5.0, 120)
            best_T, best_loss = None, float("inf")
            l = logit(np.clip(c_train, 1e-8, 1 - 1e-8))
            for T in Ts:
                p = sigmoid(l / T)
                loss = log_loss(a_train, p, labels=[0, 1])
                if loss < best_loss:
                    best_loss, best_T = loss, T
            self.T_ = float(best_T)
            self.fitted_ = True
        elif method == "platt":
            X = logit(np.clip(c_train, 1e-8, 1 - 1e-8)).reshape(-1, 1)
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X, a_train)
            self.lr_ = lr
            self.fitted_ = True
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(c_train, a_train)
            self.iso_ = iso
            self.fitted_ = True
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def transform(self, c: np.ndarray, method: str = "temperature") -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Calibrator not fitted")
        method = method.lower()
        if method == "temperature":
            l = logit(np.clip(c, 1e-8, 1 - 1e-8))
            return sigmoid(l / self.T_)
        elif method == "platt":
            X = logit(np.clip(c, 1e-8, 1 - 1e-8)).reshape(-1, 1)
            return self.lr_.predict_proba(X)[:, 1]
        elif method == "isotonic":
            return self.iso_.predict(c)
        else:
            raise ValueError(f"Unknown calibration method: {method}")


# ---------- Plotting (labeled axes & titles) ----------

def reliability_plot(df_bins: pd.DataFrame, title: str, path: str):
    plt.figure(figsize=(6.5, 6))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    valid = df_bins.dropna()
    ax.plot(valid["mean_conf"], valid["mean_acc"], marker='o', label="Empirical")
    ax.set_xlabel("Mean confidence (bin)")
    ax.set_ylabel("Empirical accuracy (bin)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(path)

def kde_dist_plot(c: np.ndarray, a: np.ndarray, out_png: str):
    plt.figure(figsize=(7.2, 5))
    ax = plt.gca()
    sns.kdeplot(c[a == 1], fill=True, label="Correct")
    sns.kdeplot(c[a == 0], fill=True, label="Incorrect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distributions (Correct vs Incorrect)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(out_png)

def violin_dist_plot(c: np.ndarray, a: np.ndarray, out_png: str):
    plt.figure(figsize=(6.8, 5))
    ax = plt.gca()
    df = pd.DataFrame({"confidence": c, "correct": np.where(a == 1, "Correct", "Incorrect")})
    sns.violinplot(data=df, x="correct", y="confidence", inner="quartile", cut=0)
    sns.boxplot(data=df, x="correct", y="confidence", width=0.2, boxprops={"zorder": 3}, showcaps=False)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence violin plot by outcome")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(out_png)

def pr_plot(c: np.ndarray, a: np.ndarray, out_png: str):
    prec, rec, _ = precision_recall_curve(a, c)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(6.2, 5))
    ax = plt.gca()
    ax.plot(rec, prec, label=f"PR curve (AUC={pr_auc:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    save_fig(out_png)

def roc_plot(c: np.ndarray, a: np.ndarray, out_png: str):
    fpr, tpr, _ = roc_curve(a, c)
    roc_auc = roc_auc_score(a, c)
    plt.figure(figsize=(6.2, 5))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.2f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    save_fig(out_png)

def risk_coverage_plot(df_rc: pd.DataFrame, aurc: float, out_png: str, title_suffix: str):
    plt.figure(figsize=(6.2, 5))
    ax = plt.gca()
    ax.plot(df_rc["coverage"], df_rc["risk"], marker='o')
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 - accuracy)")
    ax.set_title(f"Risk–Coverage Curve (AURC={aurc:.2f}) — {title_suffix}")
    ax.grid(True, alpha=0.3)
    save_fig(out_png)

def per_label_heatmap(C: np.ndarray, A: np.ndarray, n_bins: int, out_png: str):
    n_labels = C.shape[1]
    gaps = []
    for k in range(n_labels):
        _, _, dfb = ece_mce(C[:, k], A[:, k], n_bins=n_bins)
        gaps.append(dfb["gap"].fillna(0).values[:n_bins])
    G = np.stack(gaps, axis=0)  # (K, bins)
    plt.figure(figsize=(9, 3 + 0.35 * n_labels))
    ax = plt.gca()
    sns.heatmap(G, annot=False, cmap="Reds", cbar=True, cbar_kws={"label": "|acc - conf|"})
    ax.set_yticks(np.arange(n_labels) + 0.5)
    ax.set_yticklabels([f"Label {i}" for i in range(n_labels)], rotation=0)
    ax.set_xticks(np.arange(n_bins) + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_bins)], rotation=0)
    ax.set_xlabel("Calibration bins")
    ax.set_ylabel("Labels")
    ax.set_title("Per-label calibration gaps per bin")
    save_fig(out_png)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--n-labels", type=int, default=3)
    ap.add_argument("--avg-labels", type=float, default=2.0)
    ap.add_argument("--mismatch-rate", type=float, default=0.25)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--n-ensembles", type=int, default=8)
    ap.add_argument("--calibrator", type=str, default="temperature", choices=["temperature", "platt", "isotonic"])
    ap.add_argument("--tau-target", type=float, default=0.7)
    ap.add_argument("--outdir", type=str, default="plots")
    args = ap.parse_args()

    args.outdir = ensure_outdir(args.outdir)
    print(f"[outdir] {args.outdir}")

    set_seed(args.seed)

    # Data
    Y_true = gen_multilabel_truth(args.n_samples, args.n_labels, avg_labels=args.avg_labels)
    Y_pred = gen_predictions_with_mismatch(Y_true, mismatch_rate=args.mismatch_rate)
    A = (Y_true == Y_pred).astype(int)  # agreement per (i,k)
    C = synthesize_confidences(Y_true, Y_pred)

    # Flatten for pairwise metrics
    a = A.ravel()
    c = C.ravel()
    overall_accuracy = a.mean()
    N_pairs = c.size

    # Hold-out split for calibration evaluation
    c_tr, c_te, a_tr, a_te = train_test_split(c, a, test_size=0.5, random_state=args.seed, stratify=a)

    # Pre-calibration metrics
    ece_raw, mce_raw, bins_raw = ece_mce(c_te, a_te, n_bins=args.n_bins)
    brier_raw = brier(c_te, a_te)
    nll_raw = nll(c_te, a_te)
    roc_raw, pr_raw, prec_raw, rec_raw, thr_pr = roc_pr_auc(c_te, a_te)
    d_raw = cohens_d(c_te, a_te)
    rpb_raw = point_biserial(c_te, a_te)
    margins = margin_per_sample(C)
    margin_mean = float(np.mean(margins))
    ent = entropy_per_sample(C)
    entropy_mean = float(np.mean(ent))
    mi = ensemble_mi(C, n_ensembles=args.n_ensembles)
    mi_mean = float(np.mean(mi))
    df_rc_raw, aurc_raw = risk_coverage(c_te, a_te)

    # Coverage & risk at tau (raw)
    tau = args.tau_target
    mask_tau_raw = c_te >= tau
    cov_tau_raw = float(mask_tau_raw.mean())
    risk_tau_raw = float(1.0 - a_te[mask_tau_raw].mean()) if mask_tau_raw.any() else float("nan")

    # Calibration
    cal = Calibrator(name=args.calibrator)
    cal.fit(c_tr, a_tr, method=args.calibrator)
    c_cal = cal.transform(c_te, method=args.calibrator)

    # Post-cal metrics
    ece_cal, mce_cal, bins_cal = ece_mce(c_cal, a_te, n_bins=args.n_bins)
    brier_cal = brier(c_cal, a_te)
    nll_cal = nll(c_cal, a_te)
    roc_cal, pr_cal, *_ = roc_pr_auc(c_cal, a_te)
    df_rc_cal, aurc_cal = risk_coverage(c_cal, a_te)

    # Coverage & risk at tau (calibrated)
    mask_tau_cal = c_cal >= tau
    cov_tau_cal = float(mask_tau_cal.mean())
    risk_tau_cal = float(1.0 - a_te[mask_tau_cal].mean()) if mask_tau_cal.any() else float("nan")

    # Per-label ECE/AUC
    per_label = []
    for k in range(args.n_labels):
        e_k, _, _ = ece_mce(C[:, k], A[:, k], n_bins=args.n_bins)
        try:
            auc_k = roc_auc_score(A[:, k], C[:, k])
        except ValueError:
            auc_k = np.nan
        per_label.append((k, e_k, auc_k))
    df_per_label = pd.DataFrame(per_label, columns=["label", "ECE", "ROC_AUC"])

    # Save CSVs
    summary = {
        "Samples": args.n_samples,
        "Labels": args.n_labels,
        "Avg_labels_per_email": args.avg_labels,
        "Mismatch_rate": args.mismatch_rate,
        "Pairwise_accuracy": overall_accuracy,
        "ECE_raw": ece_raw,
        "MCE_raw": mce_raw,
        "Brier_raw": brier_raw,
        "NLL_raw": nll_raw,
        "ROC_AUC_raw": roc_raw,
        "PR_AUC_raw": pr_raw,
        "Cohens_d": d_raw,
        "PointBiserial": rpb_raw,
        "Margin_mean": margin_mean,
        "Entropy_mean": entropy_mean,
        "MI_mean": mi_mean,
        "AURC_raw": aurc_raw,
        "Coverage_at_tau_raw": cov_tau_raw,
        "Risk_at_tau_raw": risk_tau_raw,
        "ECE_cal": ece_cal,
        "MCE_cal": mce_cal,
        "Brier_cal": brier_cal,
        "NLL_cal": nll_cal,
        "ROC_AUC_cal": roc_cal,
        "PR_AUC_cal": pr_cal,
        "AURC_cal": aurc_cal,
        "Coverage_at_tau_cal": cov_tau_cal,
        "Risk_at_tau_cal": risk_tau_cal,
        "Calibrator": args.calibrator,
        "Temperature_T": cal.T_ if cal.T_ is not None else np.nan,
        "Tau_target": tau,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "results_summary.csv"), index=False)
    bins_raw.to_csv(os.path.join(args.outdir, "calibration_bins_raw.csv"), index=False)
    bins_cal.to_csv(os.path.join(args.outdir, "calibration_bins_calibrated.csv"), index=False)
    df_rc_raw.to_csv(os.path.join(args.outdir, "risk_coverage_raw.csv"), index=False)
    df_rc_cal.to_csv(os.path.join(args.outdir, "risk_coverage_calibrated.csv"), index=False)
    df_per_label.to_csv(os.path.join(args.outdir, "per_label_metrics.csv"), index=False)

    # Plots (autosaved with axis names and titles)
    kde_dist_plot(c_te, a_te, os.path.join(args.outdir, "confidence_distributions_kde.png"))
    violin_dist_plot(c_te, a_te, os.path.join(args.outdir, "confidence_distributions_violin.png"))
    reliability_plot(bins_raw, f"Reliability (raw) — ECE={ece_raw:.2f}, MCE={mce_raw:.2f}",
                     os.path.join(args.outdir, "reliability_raw.png"))
    reliability_plot(bins_cal, f"Reliability (calibrated: {args.calibrator}) — ECE={ece_cal:.2f}, MCE={mce_cal:.2f}",
                     os.path.join(args.outdir, "reliability_calibrated.png"))
    pr_plot(c_te, a_te, os.path.join(args.outdir, "pr_curve.png"))
    roc_plot(c_te, a_te, os.path.join(args.outdir, "roc_curve.png"))
    risk_coverage_plot(df_rc_raw, aurc_raw, os.path.join(args.outdir, "risk_coverage_raw.png"), "Raw")
    risk_coverage_plot(df_rc_cal, aurc_cal, os.path.join(args.outdir, "risk_coverage_calibrated.png"),
                       f"Calibrated ({args.calibrator})")
    per_label_heatmap(C, A, args.n_bins, os.path.join(args.outdir, "per_label_calibration_heatmap.png"))

    # Console: settings and metrics
    print("\n=== Dataset ===")
    print(f"Samples={args.n_samples}, Labels={args.n_labels}, Avg_labels/email={args.avg_labels}, "
          f"Mismatch_rate={args.mismatch_rate:.2f}, Pairwise_accuracy={overall_accuracy:.3f}")
    print("\n=== Raw metrics ===")
    print(f"ECE={ece_raw:.3f}, MCE={mce_raw:.3f}, Brier={brier_raw:.3f}, NLL={nll_raw:.3f}")
    print(f"ROC-AUC={roc_raw:.3f}, PR-AUC={pr_raw:.3f}, AURC={aurc_raw:.3f}")
    print(f"Cohen's d={d_raw:.2f}, Point-biserial r={rpb_raw:.2f}")
    print(f"Margin_mean={margin_mean:.3f}, Entropy_mean={entropy_mean:.3f}, MI_mean={mi_mean:.3f}")
    print(f"Coverage@tau={tau:.2f}: coverage={cov_tau_raw:.3f}, risk={risk_tau_raw:.3f}")
    print("\n=== Calibrated metrics ===")
    # Option A (simple and clear)
    T_str = f"(T={cal.T_:.2f})" if cal.T_ is not None else ""
    print(f"Calibrator={args.calibrator}{T_str}")

    print(f"Calibrator={args.calibrator}{T_str}")
    print(f"ECE={ece_cal:.3f}, MCE={mce_cal:.3f}, Brier={brier_cal:.3f}, NLL={nll_cal:.3f}")
    print(f"ROC-AUC={roc_cal:.3f}, PR-AUC={pr_cal:.3f}, AURC={aurc_cal:.3f}")
    print(f"Coverage@tau={tau:.2f}: coverage={cov_tau_cal:.3f}, risk={risk_tau_cal:.3f}")
    print("\n=== Per-label metrics (ECE, ROC-AUC) ===")
    print(df_per_label.to_string(index=False))

    # ----- 8.2 Raw Metric Results (exactly formatted) -----
    def fmt2(x):
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)

    def pct(x):
        try:
            return f"{100.0*float(x):.0f}%"
        except Exception:
            return str(x)

    # Human label names for K=3 (A/B/C) else generic names
    if args.n_labels == 3:
        label_names = ["A", "B", "C"]
    else:
        label_names = [f"Label {k}" for k in range(args.n_labels)]

    per_label_rows = []
    for _, row in df_per_label.iterrows():
        name = label_names[int(row["label"])] if int(row["label"]) < len(label_names) else f"Label {int(row['label'])}"
        per_label_rows.append(f"{name} (ECE={fmt2(row['ECE'])}, AUC={fmt2(row['ROC_AUC'])})")
    per_label_str = "; ".join(per_label_rows)

    cov_risk_str = f"{pct(cov_tau_raw)} (risk={fmt2(risk_tau_raw)})"

    lines = []
    lines.append("8.2 Raw Metric Results")
    lines.append("")
    lines.append(f"ECE: {fmt2(ece_raw)}")
    lines.append(f"MCE: {fmt2(mce_raw)}")
    lines.append(f"Brier: {fmt2(brier_raw)}")
    lines.append(f"NLL: {fmt2(nll_raw)}")
    lines.append(f"ROC-AUC: {fmt2(roc_raw)}")
    lines.append(f"PR-AUC: {fmt2(pr_raw)}")
    lines.append(f"Cohen’s d: {fmt2(d_raw)}")
    lines.append(f"Point-Biserial: {fmt2(rpb_raw)}")
    lines.append(f"Margin mean: {fmt2(margin_mean)}")
    lines.append(f"Entropy mean: {fmt2(entropy_mean)}")
    lines.append(f"MI mean: {fmt2(mi_mean)}")
    lines.append(f"Coverage@{tau:.1f}: {cov_risk_str}")
    lines.append(f"AURC: {fmt2(aurc_raw)}")
    lines.append(f"Per-label: {per_label_str}")

    print("\n" + "\n".join(lines))
    raw_block_path = os.path.join(args.outdir, "results_raw_metrics.txt")
    with open(raw_block_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {os.path.abspath(raw_block_path)}")

    print(f"\nAll CSVs and labeled plots saved under: {args.outdir}")

if __name__ == "__main__":
    main()
