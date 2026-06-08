"""Generate dark trading-terminal visuals for EWS_pitch_deck_v2 (dpi=200)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "deck_assets"; OUT.mkdir(parents=True, exist_ok=True)

# ---- palette ----
DARK="#0B1220"; PANEL="#16223B"; PANEL2="#22324f"; INK="#EAF0F8"; MUTE="#93A6C2"
ICE="#5E84B5"; ACCENT="#F2B134"; GREEN="#3FB36B"; RED="#E2503B"; BLUE="#4E8FD0"; GREY="#8A93A6"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,
    "axes.edgecolor":"#3a4a66","xtick.color":MUTE,"ytick.color":MUTE,"axes.labelcolor":INK})

def save(fig, name, bg=DARK):
    fig.savefig(OUT/name, dpi=200, facecolor=bg, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig); print("wrote", name)

# ================= S2 — cost of buy-and-hold (MSCI drawdown, test) =================
eq = pd.read_csv(ROOT/"outputs/tables/equity_curves.csv", index_col=0, parse_dates=True)
m = eq["MSCI_BH"]; dd = (m/m.cummax()-1)*100
fig, ax = plt.subplots(figsize=(6.5,4.5)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
ax.fill_between(dd.index, dd.values, 0, color=RED, alpha=.30)
ax.plot(dd.index, dd.values, color=RED, lw=1.8)
ax.axhline(0, color="#3a4a66", lw=.8)
ax.set_ylabel("MSCI buy-and-hold drawdown (%)", fontsize=10)
ax.annotate("COVID crash\n−27%", xy=(dd.idxmin(), dd.min()), xytext=(dd.idxmin(), dd.min()+10),
    color=INK, fontsize=10, ha="center", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED))
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8); ax.margins(x=0.01)
save(fig,"s2_drawdown.png")

# ================= S3 — risk-off clusters over 2000–2021 =================
from src.data_loader import load_dataset
raw = load_dataset(verbose=False); y = raw["Y"]
fig, ax = plt.subplots(figsize=(11.4,2.0)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
idx = y.index
ax.vlines(idx[y.values==1], 0, 1, color=RED, lw=1.4, alpha=.85)
ax.set_ylim(0,1.6); ax.set_yticks([])
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#3a4a66")
labels={"2001-06-01":"Dot-com","2008-10-01":"GFC","2011-09-01":"Euro","2020-03-15":"COVID"}
for d,t in labels.items():
    ax.text(pd.Timestamp(d),1.28,t,color=ACCENT,fontsize=11,ha="center",fontweight="bold")
gap = pd.Timestamp("2016-01-01")
ax.text(gap,0.5,"2013·2017·2019:\nZERO positives",color=MUTE,fontsize=8.5,ha="center",style="italic")
ax.tick_params(labelsize=9, colors=MUTE)
save(fig,"s3_clusters.png")

# ================= S9 — HERO trigger-map network =================
fig, ax = plt.subplots(figsize=(12.8,5.3)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
def node(x,y,r,fc,ec=None,lw=1.6,z=3):
    ax.add_patch(Circle((x,y),r,facecolor=fc,edgecolor=ec or fc,lw=lw,zorder=z))
def edge(p,q,color,lw=1.3,a=.55,rad=0.0):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=9,color=color,
        lw=lw,alpha=a,zorder=1,connectionstyle=f"arc3,rad={rad}"))
def txt(x,y,s,c=INK,fs=8.5,w="normal",ha="center"): ax.text(x,y,s,color=c,fontsize=fs,fontweight=w,ha=ha,va="center",zorder=5)

xT,xD,xR,xA = 0.045,0.40,0.635,0.90
usd=["libor_3m_chg","vrp","us10y_diff","usa_world_rel"]
gold=["real_yield_chg","jpy_strength","eqbond_corr","gold_oil_chg"]
mbs=["vix_level","term_10y_2y","mxus_dd_52w","hyig_chg4w"]
def ys(n,lo,hi): return np.linspace(hi,lo,n)
uy=ys(4,0.74,0.96); gy=ys(4,0.40,0.62); my=ys(4,0.06,0.28)
dxy_y=0.685  # shared node between USD & Gold
# domain nodes
dom={"USD":(xD,0.85,BLUE),"Gold":(xD,0.51,ACCENT),"MBS":(xD,0.17,GREY)}
# trigger nodes + edges
for n,yy in zip(usd,uy):
    node(xT,yy,0.012,ICE); txt(xT+0.022,yy,n,MUTE,7.5,ha="left"); edge((xT+0.07,yy),(xD-0.03,0.85),BLUE,1.0,.45)
for n,yy in zip(gold,gy):
    node(xT,yy,0.012,ACCENT); txt(xT+0.022,yy,n,MUTE,7.5,ha="left"); edge((xT+0.075,yy),(xD-0.03,0.51),ACCENT,1.0,.45)
for n,yy in zip(mbs,my):
    node(xT,yy,0.012,GREY); txt(xT+0.022,yy,n,MUTE,7.5,ha="left"); edge((xT+0.07,yy),(xD-0.03,0.17),GREY,1.0,.45)
# shared dxy node
node(xT,dxy_y,0.015,DARK,ACCENT,2.2); txt(xT+0.024,dxy_y,"dxy_chg4w",ACCENT,8,"bold",ha="left")
edge((xT+0.075,dxy_y),(xD-0.03,0.85),BLUE,1.8,.9,rad=-0.12); txt(0.235,0.80,"+",BLUE,13,"bold")
edge((xT+0.075,dxy_y),(xD-0.03,0.51),ACCENT,1.8,.9,rad=0.12); txt(0.235,0.585,"−",ACCENT,15,"bold")
# domain nodes draw
for k,(x,yy,c) in dom.items():
    node(x,yy,0.052,PANEL,c,2.4,z=4); txt(x,yy+0.012,k,c,12,"bold"); txt(x,yy-0.022,"sub-score",MUTE,7)
    edge((x+0.055,yy),(xR-0.075,0.55),c,1.6,.6)
# ensemble node (separate input to router)
node(xD,0.985-0.0,0.0,DARK)  # noop spacer
ens_x,ens_y=0.40,0.985
ax.add_patch(FancyBboxPatch((ens_x-0.11,ens_y-0.035),0.22,0.055,boxstyle="round,pad=0.006",
    facecolor=PANEL,edgecolor=RED,lw=2,zorder=4))
txt(ens_x,ens_y-0.008,"ENSEMBLE risk signal",RED,9,"bold")
edge((ens_x+0.11,ens_y-0.01),(xR-0.075,0.62),RED,1.8,.7)
# router
ax.add_patch(FancyBboxPatch((xR-0.075,0.40),0.15,0.30,boxstyle="round,pad=0.01",
    facecolor=PANEL2,edgecolor=ACCENT,lw=2.6,zorder=4))
txt(xR,0.60,"ROUTING",INK,12.5,"bold"); txt(xR,0.555,"ENGINE",INK,12.5,"bold")
txt(xR,0.49,"τusd=1.5",MUTE,8); txt(xR,0.455,"τoro=0.5",MUTE,8)
# regimes
regs=[("LEVERED_EQUITY",GREEN,0.86),("CASH_USD",BLUE,0.62),("GOLD",ACCENT,0.38),("MBS",GREY,0.14)]
for name,c,yy in regs:
    ax.add_patch(FancyBboxPatch((xA-0.085,yy-0.045),0.18,0.09,boxstyle="round,pad=0.008",
        facecolor=c,edgecolor=c,lw=1,zorder=4))
    txt(xA+0.005,yy,name,DARK,9.5,"bold")
    edge((xR+0.075,0.55),(xA-0.09,yy),"#56668a",1.4,.5,rad=(yy-0.55)*0.5)
# column captions
for x,s in [(xT+0.03,"14 trigger signals"),(xD,"3 macro domains"),(xR,""),(xA+0.005,"4 allocation regimes")]:
    txt(x,0.015,s,MUTE,9,"bold")
save(fig,"s9_network.png")

# ================= S11 — model AUC bars =================
mc = pd.read_csv(ROOT/"outputs/tables/models_comparison.csv")
models=mc["model"].tolist(); roc=mc["AUC-ROC_test"]; pr=mc["AUC-PR_test"]
fig, ax = plt.subplots(figsize=(6.4,3.7)); fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
x=np.arange(len(models)); w=0.38
ax.bar(x-w/2, roc, w, color=ICE, label="AUC-ROC")
ax.bar(x+w/2, pr, w, color=ACCENT, label="AUC-PR")
for i,(a,b) in enumerate(zip(roc,pr)):
    ax.text(i-w/2,a+.01,f"{a:.2f}",ha="center",color=INK,fontsize=8.5)
    ax.text(i+w/2,b+.01,f"{b:.2f}",ha="center",color=INK,fontsize=8.5)
ax.set_xticks(x); ax.set_xticklabels(models, color=INK, fontsize=10, fontweight="bold")
ax.set_ylim(0,1.0); ax.set_yticks([0,.5,1.0])
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.legend(facecolor=PANEL2, edgecolor="none", labelcolor=INK, fontsize=9, loc="lower center", ncol=2)
ax.set_title("Test-holdout discrimination", color=INK, fontsize=11, fontweight="bold", loc="left")
save(fig,"s11_models.png", bg=PANEL)

# ================= S12 — error-correlation heatmap =================
# recompute fold1+2 binary-error correlation (matches notebook 05 = 0.507 mean)
from src.splits import walkforward_split
from src.preprocessing import FoldScaler
from src.models import MVGAnomalyDetector, OneClassSVMDetector, AutoencoderDetector, IsolationForestDetector
from src.ensemble import clone_unfit, score_to_percentile
ebc = pd.read_parquet(ROOT/"data/processed/routing_triggers.parquet")["equity_bond_corr_13w"]
enrich=lambda df: df.join(ebc,how="left").dropna(subset=["equity_bond_corr_13w"])
wf=walkforward_split(save=False,verbose=False); sc=FoldScaler.load(ROOT/"outputs/models/scaler_final.pkl")
M={"MVG":MVGAnomalyDetector.load(ROOT/"outputs/models/mvg_final.pkl"),
   "SVM":OneClassSVMDetector.load(ROOT/"outputs/models/svm_final.pkl"),
   "AE":AutoencoderDetector.load(ROOT/"outputs/models/autoencoder_final.keras"),
   "IF":IsolationForestDetector.load(ROOT/"outputs/models/if_final.pkl")}
errs={k:[] for k in M}
for f in wf["models"]["cv_folds"][:2]:
    tr=enrich(f["train"]); val=enrich(f["val"]); tn=tr[tr.Y==0]; yv=val.Y.values
    for k,mod in M.items():
        fm=clone_unfit(mod).fit(sc.transform(tn,f["fold_id"]))
        ref=fm.score_samples(sc.transform(tn,f["fold_id"]))
        sv=fm.score_samples(sc.transform(val,f["fold_id"]))
        thr=np.percentile(ref,100*(1-yv.mean())) if False else mod.threshold_
        pred=(sv>=mod.threshold_).astype(int)
        errs[k].append((pred!=yv).astype(int))
E=pd.DataFrame({k:np.concatenate(v) for k,v in errs.items()})
C=E.corr()
fig, ax = plt.subplots(figsize=(4.6,3.2)); fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
im=ax.imshow(C.values, cmap="magma", vmin=0, vmax=1)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels(C.columns, color=INK, fontsize=10); ax.set_yticklabels(C.index, color=INK, fontsize=10)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f"{C.values[i,j]:.2f}",ha="center",va="center",
                color="white" if C.values[i,j]<.6 else "black", fontsize=10, fontweight="bold")
off=C.values[~np.eye(4,dtype=bool)].mean()
ax.set_title("Binary error correlation · folds 1–2", color=INK, fontsize=11, fontweight="bold")
save(fig,"s12_errcorr.png", bg=PANEL)
print("mean off-diag err corr (this run, AE nondeterministic):", round(off,3), "— slide cites notebook 0.507")

# ================= S13 — growth of $1 (hero results chart) =================
order=["EWS","MSCI_BH","60_40","Butterfly","Permanent","RiskParity"]
col={"EWS":ACCENT,"MSCI_BH":"#cfd8e6","60_40":BLUE,"Butterfly":GREEN,"Permanent":"#9b7bd4","RiskParity":"#e08a3c"}
lab={"EWS":"EWS (ours)","MSCI_BH":"MSCI buy-&-hold","60_40":"60/40","Butterfly":"Butterfly","Permanent":"Permanent","RiskParity":"Risk Parity"}
fig, ax = plt.subplots(figsize=(8.4,4.5)); fig.patch.set_facecolor(DARK); ax.set_facecolor(DARK)
for c in order:
    ax.plot(eq.index, eq[c], color=col[c], lw=3.4 if c=="EWS" else 1.5,
            alpha=1 if c=="EWS" else .8, label=lab[c], zorder=5 if c=="EWS" else 2)
ax.set_yscale("log"); ax.set_ylabel("growth of $1  (log)", fontsize=10)
ax.axhline(1, color="#3a4a66", lw=.8)
covid=pd.Timestamp("2020-03-23")
ax.axvspan(pd.Timestamp("2020-02-20"),pd.Timestamp("2020-04-15"),color=RED,alpha=.10)
ax.annotate("COVID crash →\nrotate into GOLD", xy=(covid, eq.loc[:covid,"EWS"].iloc[-1]),
    xytext=(pd.Timestamp("2019-04-01"),1.55), color=ACCENT, fontsize=9.5, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ACCENT))
ax.text(eq.index[-1], eq["EWS"].iloc[-1]*1.02, "1.87×", color=ACCENT, fontsize=12, fontweight="bold", va="center")
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
ax.legend(facecolor=PANEL, edgecolor="none", labelcolor=INK, fontsize=8.5, loc="upper left", ncol=2)
import matplotlib.ticker as mt
ax.yaxis.set_major_formatter(mt.FuncFormatter(lambda v,_: f"{v:.1f}×"))
save(fig,"s13_equity.png")

# ================= S14 — max drawdown bars (nuance) =================
bm = pd.read_csv(ROOT/"outputs/tables/backtest_metrics_all.csv", index_col=0)
dd = (bm.loc["Max_Drawdown"]*100)[order]
fig, ax = plt.subplots(figsize=(6.6,2.5)); fig.patch.set_facecolor(PANEL); ax.set_facecolor(PANEL)
colors=[ACCENT if c=="EWS" else ("#cfd8e6" if c=="MSCI_BH" else GREY) for c in order]
bars=ax.barh([lab[c] for c in order], dd.values, color=colors)
for b,v,c in zip(bars,dd.values,colors): ax.text(v+0.6, b.get_y()+b.get_height()/2, f"{v:.0f}%", va="center", ha="left", color="#0B1220" if c==ACCENT else INK, fontsize=9, fontweight="bold")
ax.invert_yaxis(); ax.set_xlabel("max drawdown (%)", fontsize=9)
for s in ["top","right"]: ax.spines[s].set_visible(False)
ax.tick_params(labelsize=9, colors=INK)
ax.set_title("Max drawdown — the honest nuance", color=INK, fontsize=10.5, fontweight="bold", loc="left")
save(fig,"s14_dd.png", bg=PANEL)
print("ALL ASSETS DONE")
