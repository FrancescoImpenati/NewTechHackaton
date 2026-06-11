"""Stadio 1 della feature selection: ranking univariato del potere anomaly-detection.

Metodologia (coerente con la SHARED METHODOLOGY del PROJECT_HANDOFF):
- per ogni fold walk-forward: mu/sigma stimati sui TRAIN NORMALS (Y==0) del fold
  -> lo score univariato |z| e' di fatto "un MVG a una sola feature";
- direzione (per le feature one-sided tipo VIX): segno scelto massimizzando l'AP
  sul TRAIN del fold (mai sul val -> niente leakage), poi valutato sul val;
- metrica: average precision (AUC-PR) sul fold val, la metrica onesta sotto sbilanciamento;
- aggregazione cross-fold: media pesata per n_pos del val (fold 1-2 dominano, come da methodology);
- equity_bond_corr_13w joinata da routing_triggers (la "57esima colonna" della methodology).

Output: outputs/tables/feature_ranking_univariate.csv (eseguito dalla root del repo)
        + conteggio dei cluster di correlazione sul development set (dimensiona lo stadio 2).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
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
    """Quanti cluster formano le feature sul development set (average linkage, d=1-|rho|).
    Serve a dimensionare lo stadio 2 (clustered MDA): t=0.4 ~ |rho|>0.6 intra-cluster."""
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
    print("\nposizione in classifica delle feature 'problematiche' per l'estensione dati:", pos)
    print("\ncluster di correlazione sul development:", correlation_cluster_count())
