"""Feature selection for the 56-feature model set (stage 1: univariate ranking).

Stage-1 methodology (consistent with the SHARED METHODOLOGY of the handoff):
- per walk-forward fold: mu/sigma estimated on the fold's TRAIN NORMALS (Y==0)
  -> the univariate |z| score is effectively "a single-feature MVG";
- direction (for one-sided features like VIX): sign chosen by maximizing AP on
  the fold TRAIN (never on the val -> no leakage), then evaluated on the val;
- metric: average precision (AUC-PR) on the fold val, the honest metric under
  class imbalance;
- cross-fold aggregation: mean weighted by the val's n_pos (folds 1-2 dominate,
  as per methodology);
- equity_bond_corr_13w joined from routing_triggers (the "57th column" of the
  methodology).

Selection operates on the MODEL feature set only: ``vrp`` / ``jpy_strength``
also exist as routing triggers (handoff §5.4) — routing is out of scope here,
so any decision taken on the model copies leaves the routing engine untouched.

Output: outputs/tables/feature_ranking_univariate.csv (run from the repo root:
``python -m src.feature_selection``) + correlation-cluster counts on the
development set (sizes stage 2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "data" / "processed" / "walkforward"
TRIGGERS = ROOT / "data" / "processed" / "routing_triggers.parquet"
OUT = ROOT / "outputs" / "tables" / "feature_ranking_univariate.csv"
FOLDS = (1, 2, 3, 4, 5)
TARGET = "Y"


def _load(name: str, corr13: pd.DataFrame) -> pd.DataFrame:
    return pd.read_parquet(WF / name).join(corr13, how="left")


def univariate_ranking() -> pd.DataFrame:
    corr13 = pd.read_parquet(TRIGGERS)[["equity_bond_corr_13w"]]
    rows = []
    for f in FOLDS:
        tr = _load(f"fold_{f}_train.parquet", corr13)
        tn = _load(f"fold_{f}_train_normals.parquet", corr13)
        va = _load(f"fold_{f}_val.parquet", corr13)
        y_tr, y_va = tr[TARGET].to_numpy(), va[TARGET].to_numpy()
        for c in [c for c in va.columns if c != TARGET]:
            mu, sd = tn[c].mean(), tn[c].std()
            if not np.isfinite(sd) or sd == 0:
                continue
            z_tr, z_va = ((tr[c] - mu) / sd), ((va[c] - mu) / sd)
            m_tr, m_va = z_tr.notna().to_numpy(), z_va.notna().to_numpy()
            if y_tr[m_tr].sum() == 0 or y_va[m_va].sum() == 0:
                continue
            ap_abs = average_precision_score(y_va[m_va], np.abs(z_va[m_va]))
            sign = 1 if (average_precision_score(y_tr[m_tr], z_tr[m_tr])
                         >= average_precision_score(y_tr[m_tr], -z_tr[m_tr])) else -1
            ap_dir = average_precision_score(y_va[m_va], sign * z_va[m_va])
            rows.append({"fold": f, "feature": c, "n_pos": int(y_va[m_va].sum()),
                         "prevalence": float(y_va[m_va].mean()),
                         "ap_abs": ap_abs, "ap_dir": ap_dir, "sign": sign})
    df = pd.DataFrame(rows)

    out = []
    for feat, g in df.groupby("feature"):
        w = g["n_pos"].to_numpy(dtype=float)
        out.append({
            "feature": feat,
            "AP_abs_w": float(np.average(g["ap_abs"], weights=w)),
            "AP_dir_w": float(np.average(g["ap_dir"], weights=w)),
            "sign": int(np.sign(g["sign"].mul(g["n_pos"]).sum())) or 1,
            "n_folds": int(len(g)),
        })
    agg = pd.DataFrame(out)
    agg["AP_best_w"] = agg[["AP_abs_w", "AP_dir_w"]].max(axis=1)
    agg = agg.sort_values("AP_best_w", ascending=False).reset_index(drop=True)

    base = float(np.average(df.drop_duplicates("fold")["prevalence"],
                            weights=df.drop_duplicates("fold")["n_pos"]))
    agg["lift_vs_random"] = agg["AP_best_w"] / base
    agg.attrs["baseline_AP_random"] = base
    return agg


def correlation_cluster_count(thresholds=(0.3, 0.4, 0.5)) -> dict[float, int]:
    """How many clusters the features form on the development set (average
    linkage, d=1-|rho|). Sizes stage 2 (clustered MDA): t=0.4 ~ |rho|>0.6
    intra-cluster."""
    corr13 = pd.read_parquet(TRIGGERS)[["equity_bond_corr_13w"]]
    dev = _load("development.parquet", corr13).drop(columns=[TARGET]).dropna()
    d = (1.0 - dev.corr().abs()).to_numpy().copy()
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    return {t: int(fcluster(Z, t, criterion="distance").max()) for t in thresholds}


if __name__ == "__main__":
    rank = univariate_ranking()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rank.to_csv(OUT, index=False)
    base = rank.attrs["baseline_AP_random"]
    print(f"baseline AP (random scorer, weighted) = {base:.3f}\n")
    print("=== TOP 15 ===")
    print(rank.head(15).to_string(index=False,
          formatters={c: "{:.3f}".format for c in ("AP_abs_w", "AP_dir_w", "AP_best_w", "lift_vs_random")}))
    print("\n=== BOTTOM 10 ===")
    print(rank.tail(10).to_string(index=False,
          formatters={c: "{:.3f}".format for c in ("AP_abs_w", "AP_dir_w", "AP_best_w", "lift_vs_random")}))
    probe = ["MXRU", "EONIA", "ECSURPUS", "BDIY", "libor_3m_spread_chg4w"]
    pos = {p: (rank.index[rank.feature == p][0] + 1) if (rank.feature == p).any() else None for p in probe}
    print("\nranking position of the 'problematic-for-data-extension' features:", pos)
    print("\ncorrelation clusters on the development set:", correlation_cluster_count())
