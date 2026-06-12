"""Feature selection for the 56-feature model set (stages 1-2).

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
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"
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


# --------------------------------------------------------------------------
# Stage 2 — correlation clusters + clustered permutation importance (WP2)
# --------------------------------------------------------------------------
def _development_matrix() -> pd.DataFrame:
    """Enriched development matrix (56 features, no Y, warm-up NaNs dropped)."""
    corr13 = pd.read_parquet(TRIGGERS)[["equity_bond_corr_13w"]]
    return _load("development.parquet", corr13).drop(columns=[TARGET]).dropna()


def correlation_clusters(threshold: float = 0.4, save: bool = True) -> pd.DataFrame:
    """Hierarchical clustering of the 56 MODEL features on the development set.

    Distance = 1 - |rho| (Pearson), average linkage, tree cut at ``threshold``
    (t=0.4 ~ |rho|>0.6 intra-cluster). Returns a DataFrame
    ``[feature, cluster_id]`` and persists it to feature_clusters.csv.

    Note: selection operates on the MODEL feature set only. ``vrp`` and
    ``jpy_strength`` are duplicated as routing triggers (handoff §5.4):
    dropping their model copies leaves the routing engine untouched, so
    routing is out of scope here.
    """
    dev = _development_matrix()
    d = (1.0 - dev.corr().abs()).to_numpy().copy()
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(Z, threshold, criterion="distance")
    df = pd.DataFrame({"feature": dev.columns, "cluster_id": labels.astype(int)})
    df = df.sort_values(["cluster_id", "feature"]).reset_index(drop=True)
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / "feature_clusters.csv", index=False)
    return df


def clustered_permutation_importance(clusters: pd.DataFrame | None = None,
                                     n_repeats: int = 20, seed: int = 42,
                                     save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Cluster importance = drop in val AUC-PR of the soft-median ensemble when
    all columns of the cluster are jointly row-shuffled in the val matrix.

    Per fold the 4 detectors are fit ONCE (leakage-free: clone_unfit of the
    production models + fresh per-fold scalers) and KEPT; permutation is
    re-SCORING only, never re-fitting, and the reference (train-normal)
    percentile distributions stay untouched. The same row permutation is
    applied to every column of the cluster (joint shuffle preserves the
    intra-cluster dependence structure). Drops are averaged over ``n_repeats``
    seeded permutations per fold, then aggregated across folds with
    n_pos weights (methodology contract).

    Returns one row per cluster: members, mean_drop (n_pos-weighted),
    std_drop (n_pos-weighted mean of the per-fold std over repeats) and the
    per-fold mean drops. Baseline per-fold AUC-PRs are in ``df.attrs``.
    """
    from src.ensemble import MODEL_ORDER, clone_unfit, score_to_percentile
    from src.models import weighted_mean
    from src.preprocessing import FoldScaler
    from src.sensitivity import load_model_folds, load_production_models
    from sklearn.metrics import average_precision_score

    clusters = clusters if clusters is not None else correlation_clusters()
    cluster_members = {int(cid): list(g["feature"])
                       for cid, g in clusters.groupby("cluster_id")}

    data = load_model_folds()
    folds = data["folds"]
    production = load_production_models()
    scaler = FoldScaler()
    scaler.fit_per_fold(folds)
    cols = scaler.feature_cols_
    col_idx = {c: i for i, c in enumerate(cols)}

    rng = np.random.default_rng(seed)
    per_fold_mean: dict[int, dict[int, float]] = {}
    per_fold_std: dict[int, dict[int, float]] = {}
    base_ap: dict[int, float] = {}
    n_pos: dict[int, int] = {}

    for fold in folds:
        fid, train, val = fold["fold_id"], fold["train"], fold["val"]
        tn = train[train[TARGET] == 0]
        Xtn = scaler.transform(tn, fid)
        Xval = scaler.transform(val, fid)
        y_val = val[TARGET].to_numpy()
        n_val = len(val)

        fitted, ref = {}, {}
        for name in MODEL_ORDER:
            fitted[name] = clone_unfit(production[name]).fit(Xtn)
            ref[name] = fitted[name].score_samples(Xtn)

        def ensemble_ap(X, n_blocks=1):
            """soft-median ensemble AUC-PR; X may stack n_blocks copies of val."""
            pct = np.array([score_to_percentile(ref[n], fitted[n].score_samples(X))
                            for n in MODEL_ORDER])
            ens = np.median(pct, axis=0)
            if n_blocks == 1:
                return [average_precision_score(y_val, ens)]
            return [average_precision_score(y_val, b)
                    for b in ens.reshape(n_blocks, n_val)]

        base_ap[fid] = ensemble_ap(Xval)[0]
        n_pos[fid] = int(y_val.sum())
        per_fold_mean[fid], per_fold_std[fid] = {}, {}

        for cid in sorted(cluster_members):
            idx = [col_idx[c] for c in cluster_members[cid]]
            blocks = []
            for _ in range(n_repeats):
                perm = rng.permutation(n_val)
                Xp = Xval.copy()
                Xp[:, idx] = Xval[perm][:, idx]
                blocks.append(Xp)
            aps = np.array(ensemble_ap(np.vstack(blocks), n_blocks=n_repeats))
            drops = base_ap[fid] - aps
            per_fold_mean[fid][cid] = float(drops.mean())
            per_fold_std[fid][cid] = float(drops.std())
        if verbose:
            print(f"fold {fid}: baseline AUC-PR={base_ap[fid]:.3f}, "
                  f"{len(cluster_members)} clusters x {n_repeats} repeats done")

    fids = sorted(base_ap)
    w = np.array([n_pos[f] for f in fids], dtype=float)
    rows = []
    for cid in sorted(cluster_members):
        m = np.array([per_fold_mean[f][cid] for f in fids])
        s = np.array([per_fold_std[f][cid] for f in fids])
        rows.append({"cluster_id": cid,
                     "members": "|".join(cluster_members[cid]),
                     "n_members": len(cluster_members[cid]),
                     "mean_drop": weighted_mean(m, w),
                     "std_drop": weighted_mean(s, w),
                     **{f"drop_fold_{f}": m[i] for i, f in enumerate(fids)}})
    df = pd.DataFrame(rows).sort_values("mean_drop", ascending=False).reset_index(drop=True)
    df.attrs["baseline_auc_pr_per_fold"] = base_ap
    df.attrs["n_pos_per_fold"] = n_pos
    if save:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / "cluster_importance.csv", index=False)
    return df


def plot_cluster_importance(imp: pd.DataFrame, out_path: Path | None = None):
    """Sorted horizontal bar chart of the n_pos-weighted mean AUC-PR drops."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = imp.sort_values("mean_drop", ascending=True)
    labels = [f"C{int(r.cluster_id)}: " + (r.members if len(r.members) <= 38
              else r.members[:35] + "...") for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 0.28 * len(d) + 1.5))
    ax.barh(labels, d["mean_drop"], xerr=d["std_drop"], color="#3182bd",
            error_kw={"elinewidth": 0.8, "alpha": 0.6})
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("n_pos-weighted mean drop in val AUC-PR (soft-median ensemble)")
    ax.set_title("Clustered permutation importance (joint shuffle, 20 repeats/fold)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    out_path = out_path or FIGURES_DIR / "cluster_importance.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


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
