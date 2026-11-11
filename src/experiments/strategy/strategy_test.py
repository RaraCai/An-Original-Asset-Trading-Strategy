from __future__ import annotations
import argparse, json
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PPY={"daily":252.0,"weekly":50.4}


def load_test(folder):
    p=folder/"test_predictions.csv"
    if not p.is_file(): raise FileNotFoundError(p)
    d=pd.read_csv(p,parse_dates=["decision_date"]).sort_values("decision_date").reset_index(drop=True)
    req=["decision_date","actual_return","predicted_return","historical_mean_prediction"]
    miss=sorted(set(req)-set(d.columns))
    if miss: raise ValueError(f"{p} missing columns: {miss}")
    if d.empty or d.decision_date.duplicated().any(): raise ValueError(f"{p}: empty/duplicate dates")
    if not np.isfinite(d[req[1:]].to_numpy(float)).all(): raise ValueError(f"{p}: nonfinite values")
    return d



def load_validation_history(folder,pipeline):
    name="alpha_refinement_predictions.csv" if pipeline=="daily" else "weekly_alpha_predictions.csv"
    d=pd.read_csv(folder/name,parse_dates=["decision_date"])
    alpha=0.000225 if pipeline=="daily" else 0.0003
    a=pd.to_numeric(d["alpha"],errors="coerce")
    d=d.loc[np.isclose(a,alpha,rtol=0,atol=1e-12)].sort_values("decision_date")
    if d.empty or d.decision_date.duplicated().any():
        raise ValueError(f"{pipeline}: validation history missing or duplicate after alpha filter")
    return d["actual_return"].to_numpy(float)

def make_weights(frame,pipeline,method,source,p,prior_actual=None):
    x=frame["predicted_return" if source=="model" else "historical_mean_prediction"].to_numpy(float)
    y=frame.actual_return.to_numpy(float)
    if method=="original_mapping": return np.clip(np.maximum(x-p["delta"],0)/p["k"],0,1)
    if method=="strategy_a": return np.clip(p["base_exposure"]+x/p["k"],0,1)
    if method=="strategy_c":
        mult=np.where(x>=0,1.0,p["downside_multiplier"])
        return np.clip(p["base_exposure"]+x/p["k"]*mult,0,1)
    if method=="strategy_b":
        prior=np.asarray([] if prior_actual is None else prior_actual,dtype=float)
        all_y=np.r_[prior,y]
        nprior=len(prior)
        w=np.empty(len(x))
        for i,pred in enumerate(x):
            # Include only labels matured before the current signal (two-period lag).
            end=max(0,nprior+i-1)
            h=all_y[max(0,end-int(p["lookback"])):end]
            rv=np.std(h,ddof=1)*np.sqrt(PPY[pipeline]) if len(h)>=2 else p["target_vol"]
            if not np.isfinite(rv) or rv<=1e-12: rv=p["target_vol"]
            vscale=min(1.0,p["target_vol"]/rv)
            sscale=np.clip(1.0+pred/p["k"],0,1.5)
            w[i]=np.clip(vscale*sscale,0,1)
        return w
    raise ValueError(method)


def returns_metrics(frame,w,pipeline,cost_bps):
    w=np.asarray(w,float)
    if np.any((w<0)|(w>1)) or not np.isfinite(w).all(): raise ValueError("weights outside [0,1]")
    turn=np.abs(np.diff(np.r_[0.0,w]))
    if len(turn): turn[-1]+=abs(w[-1])
    cost=turn*cost_bps/10000
    gross=w*frame.actual_return.to_numpy(float)
    net=gross-cost
    eq=np.cumprod(1+net); dd=eq/np.maximum.accumulate(eq)-1
    sd=np.std(net,ddof=1) if len(net)>1 else 0
    n=len(net)
    m={"n_periods":n,"total_return":eq[-1]-1,
       "annualized_return":eq[-1]**(PPY[pipeline]/n)-1 if eq[-1]>0 else np.nan,
       "annualized_volatility":sd*np.sqrt(PPY[pipeline]),
       "sharpe_ratio":np.mean(net)/sd*np.sqrt(PPY[pipeline]) if sd>0 else np.nan,
       "max_drawdown":dd.min(),"annualized_turnover":turn.sum()/(n/PPY[pipeline]),
       "average_position":w.mean(),"total_transaction_cost":cost.sum()}
    path=pd.DataFrame({"decision_date":frame.decision_date,"weight":w,"turnover":turn,
        "gross_return":gross,"transaction_cost":cost,"net_return":net,"equity":eq,"drawdown":dd})
    return m,path


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":12,
        "axes.labelsize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
        "axes.grid":True,"grid.alpha":.22,"grid.linewidth":.6,"savefig.dpi":220,
        "axes.spines.top":False,"axes.spines.right":False})



def set_academic_style():
    """Consistent typography and restrained colours inspired by the correlation heatmap."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.6,
        "figure.dpi": 120,
        "savefig.dpi": 240,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


STRATEGY_COLOURS = {
  "Buy-and-Hold": "#5A5A5A",
  "Historical Mean": "#2878B5",
  "Original Mapping (Lasso)": "#9AC9DB",
  "Strategy A (Lasso)": "#F8ACBC",
  "Strategy B (Lasso)": "#C82423",
  "Strategy C (Lasso)": "#FF8884",
}

DAILY_COLOUR = "#2878B5"
WEEKLY_COLOUR = "#C82423"

STRATEGY_ORDER = [
    "Buy-and-Hold",
    "Historical Mean",
    "Original Mapping (Lasso)",
    "Strategy A (Lasso)",
    "Strategy B (Lasso)",
    "Strategy C (Lasso)",
]


def _clean_strategy_label(name):
    """Remove Weekly cost suffix for readable, consistent labels."""
    return re.sub(r"\s+\([\d.]+\s+bps\)$", "", str(name))


def plot_pipeline(paths, pipeline, out):
    """Create one clean equity chart and one drawdown chart per pipeline.

    Weekly charts use the 5 bps base-cost scenario only. Other cost scenarios
    remain in the CSV sensitivity output.
    """
    set_academic_style()
    plot_data = paths.copy()

    if pipeline == "weekly" and "weekly_cost_bps" in plot_data.columns:
        plot_data = plot_data.loc[
            np.isclose(pd.to_numeric(plot_data["weekly_cost_bps"]), 5.0)
        ].copy()

    if plot_data.empty:
        raise ValueError(f"No {pipeline} observations available for plotting.")

    plot_data["plot_label"] = plot_data["strategy"].map(_clean_strategy_label)

    for column, title, ylabel, filename in [
        ("equity", "Equity Curve", "Growth of $1", f"{pipeline}_test_equity.pdf"),
        ("drawdown", "Drawdown", "Drawdown", f"{pipeline}_test_drawdown.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(17.0, 5.8))

        for label in STRATEGY_ORDER:
            group = plot_data.loc[plot_data["plot_label"].eq(label)].sort_values(
                "decision_date"
            )
            if group.empty:
                continue
            ax.plot(
                group["decision_date"],
                group[column],
                label=label,
                color=STRATEGY_COLOURS[label],
                linewidth=2.0 if label == "Buy-and-Hold" else 1.65,
                alpha=0.96,
                zorder=4 if label == "Buy-and-Hold" else 3,
            )

        ax.set_title(f"{pipeline.title()} Individual Test: {title}", pad=12)
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="both")
        ax.set_axisbelow(True)

        # Single vertical legend outside the plotting area on the right.
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.805, 0.50),
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            labelspacing=1.15,
            handlelength=2.8,
        )
        fig.subplots_adjust(left=0.075, right=0.785, top=0.90, bottom=0.13)
        fig.savefig(out / filename, bbox_inches="tight")
        plt.close(fig)


def _metric_plot_frame(metrics):
    """Return exactly six comparable strategy rows for each pipeline.

    Daily uses its zero-cost assumption. Weekly uses the predeclared 5 bps
    base-cost scenario. Cost sensitivity remains available separately.
    """
    frame = metrics.copy()
    frame["plot_label"] = frame["strategy"].map(_clean_strategy_label)

    daily = frame.loc[frame["pipeline"].eq("daily")].copy()
    weekly = frame.loc[
        frame["pipeline"].eq("weekly")
        & np.isclose(pd.to_numeric(frame["weekly_cost_bps"]), 5.0)
    ].copy()

    daily = daily.drop_duplicates("plot_label")
    weekly = weekly.drop_duplicates("plot_label")

    expected = set(STRATEGY_ORDER)
    for name, part in [("Daily", daily), ("Weekly", weekly)]:
        found = set(part["plot_label"])
        missing = expected - found
        if missing:
            raise ValueError(
                f"{name} metric chart is missing strategies: {sorted(missing)}"
            )

    daily["pipeline_label"] = "Daily"
    weekly["pipeline_label"] = "Weekly"
    return pd.concat([daily, weekly], ignore_index=True)


def plot_cross_pipeline_metrics(metrics, out):
    """Four grouped bar charts: six strategies x Daily/Weekly = twelve bars."""
    set_academic_style()
    frame = _metric_plot_frame(metrics)

    metric_specs = [
        ("annualized_return", "Annualized Return", "Annualized return", "comparison_annualized_return.png"),
        ("annualized_volatility", "Annualized Volatility", "Annualized volatility", "comparison_annualized_volatility.png"),
        ("max_drawdown", "Maximum Drawdown", "Maximum drawdown", "comparison_max_drawdown.png"),
        ("sharpe_ratio", "Sharpe Ratio", "Sharpe ratio", "comparison_sharpe_ratio.png"),
    ]

    x = np.arange(len(STRATEGY_ORDER))*0.5
    width = 0.2

    for metric, title, ylabel, filename in metric_specs:
        daily_values = []
        weekly_values = []
        for label in STRATEGY_ORDER:
            daily_values.append(
                float(frame.loc[
                    frame["pipeline_label"].eq("Daily")
                    & frame["plot_label"].eq(label), metric
                ].iloc[0])
            )
            weekly_values.append(
                float(frame.loc[
                    frame["pipeline_label"].eq("Weekly")
                    & frame["plot_label"].eq(label), metric
                ].iloc[0])
            )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(
            x - width / 2,
            daily_values,
            width,
            label="Daily",
            color=DAILY_COLOUR,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.bar(
            x + width / 2,
            weekly_values,
            width,
            label="Weekly",
            color=WEEKLY_COLOUR,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        ax.set_title(f"Daily vs Weekly: {title}", pad=12)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([
            "Buy-and-Hold",
            "Historical\nMean",
            "Original\nMapping",
            "Strategy A",
            "Strategy B",
            "Strategy C",
        ])
        ax.axhline(0, color="#555555", linewidth=0.8, zorder=2)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            ncol=2,
            frameon=False,
            handlelength=2.0,
        )
        fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.22)
        fig.savefig(out / filename, bbox_inches="tight")
        plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--daily-test-dir",type=Path,required=True)
    ap.add_argument("--weekly-test-dir",type=Path,required=True)
    ap.add_argument("--config-dir",type=Path,default=Path("configs"))
    ap.add_argument("--output-dir",type=Path,default=Path("outputs"))
    ap.add_argument("--weekly-cost-scenarios",default="0,1,5,10")
    ap.add_argument("--daily-validation-dir",type=Path,required=True)
    ap.add_argument("--weekly-validation-dir",type=Path,required=True)
    a=ap.parse_args()
    cd=a.config_dir if a.config_dir.is_absolute() else Path.cwd()/a.config_dir
    od=a.output_dir if a.output_dir.is_absolute() else Path.cwd()/a.output_dir
    out=od/f"strategy_individual_test_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True,exist_ok=False)
    cost_scenarios=[float(x) for x in a.weekly_cost_scenarios.split(",")]
    if any(x<0 for x in cost_scenarios): raise ValueError("Weekly cost scenarios must be non-negative")
    validation_dirs={"daily":a.daily_validation_dir.resolve(),"weekly":a.weekly_validation_dir.resolve()}
    all_metrics=[]; all_paths=[]; all_annual=[]
    for pipeline,test_dir in [("daily",a.daily_test_dir.resolve()),("weekly",a.weekly_test_dir.resolve())]:
        cfgp=cd/f"{pipeline}_strategy.yaml"
        if not cfgp.is_file(): raise FileNotFoundError(f"{cfgp}; run strategy_search.py first")
        cfg=yaml.safe_load(cfgp.read_text(encoding="utf-8"))
        frame=load_test(test_dir)
        prior_actual=load_validation_history(validation_dirs[pipeline],pipeline)
        specs=[
            ("Buy-and-Hold",None,None,0.0,"benchmark"),
            ("Historical Mean", "original_mapping", cfg["strategies"]["historical_mean_benchmark"]["params"],None,"benchmark"),
        ]
        for method,label in [("original_mapping","Original Mapping"),("strategy_a","Strategy A"),
                             ("strategy_b","Strategy B"),("strategy_c","Strategy C")]:
            specs.append((f"{label} (Lasso)",method,cfg["strategies"][method]["params"],None,"model"))
        ppaths=[]; mrows=[]
        for name,method,params,cost_override,category in specs:
            if method is None:
                w=np.ones(len(frame))
                cost=0.0 if pipeline=="daily" else 5.0
            else:
                source="historical_mean" if name=="Historical Mean" else "model"
                w=make_weights(frame,pipeline,method,source,params,
                               prior_actual=prior_actual if method=="strategy_b" else None)
                cost=float(params["cost_bps"])
            costs_to_run = cost_scenarios if pipeline=="weekly" and method is not None else [cost]
            for scenario_cost in costs_to_run:
                m,path=returns_metrics(frame,w,pipeline,scenario_cost)
                path["pipeline"]=pipeline
                path["strategy"]=name if pipeline=="daily" or method is None else f"{name} ({scenario_cost:g} bps)"
                path["category"]=category
                path["weekly_cost_bps"]=scenario_cost if pipeline=="weekly" else 0.0
                ppaths.append(path)
                mrows.append({"pipeline":pipeline,"strategy":path["strategy"].iloc[0],"category":category,
                              "weekly_cost_bps":scenario_cost if pipeline=="weekly" else 0.0,**m})
        paths=pd.concat(ppaths,ignore_index=True); metrics=pd.DataFrame(mrows)
        annual=[]
        for (name,year),g in paths.groupby(["strategy",paths.decision_date.dt.year],sort=True):
            r=g.net_return.to_numpy(float); n=len(r); eq=np.cumprod(1+r)
            sd=np.std(r,ddof=1) if n>1 else 0.0
            annual.append({"pipeline":pipeline,"strategy":name,"year":int(year),
                "n_periods":n,"total_return":eq[-1]-1,
                "annualized_return":eq[-1]**(PPY[pipeline]/n)-1 if eq[-1]>0 else np.nan,
                "annualized_volatility":sd*np.sqrt(PPY[pipeline]),
                "sharpe_ratio":np.mean(r)/sd*np.sqrt(PPY[pipeline]) if sd>0 else np.nan,
                "max_drawdown":(eq/np.maximum.accumulate(eq)-1).min(),
                "average_position":g.weight.mean(),"total_transaction_cost":g.transaction_cost.sum()})
        all_metrics.append(metrics); all_paths.append(paths); all_annual.append(pd.DataFrame(annual))
        plot_pipeline(paths,pipeline,out)
        print(f"{pipeline.upper()} test complete: {len(frame)} observations",flush=True)
    M=pd.concat(all_metrics,ignore_index=True); P=pd.concat(all_paths,ignore_index=True); Y=pd.concat(all_annual,ignore_index=True)
    M.to_csv(out/"individual_test_metrics.csv",index=False)
    P.to_csv(out/"individual_test_returns.csv",index=False,date_format="%Y-%m-%d")
    Y.to_csv(out/"individual_test_annual_metrics.csv",index=False)
    plot_cross_pipeline_metrics(M, out)
    M.loc[M["pipeline"].eq("weekly")].to_csv(
        out/"weekly_transaction_cost_sensitivity.csv",index=False
    )
    comparison=[]
    for pipeline in ["daily","weekly"]:
        sub=M[M.pipeline==pipeline]; bh=sub[sub.strategy=="Buy-and-Hold"].iloc[0]
        for _,r in sub.iterrows():
            comparison.append({"pipeline":pipeline,"strategy":r.strategy,
                "annualized_return_diff_vs_bh":r.annualized_return-bh.annualized_return,
                "sharpe_diff_vs_bh":r.sharpe_ratio-bh.sharpe_ratio,
                "max_drawdown_improvement_vs_bh":r.max_drawdown-bh.max_drawdown})
    pd.DataFrame(comparison).to_csv(out/"benchmark_comparison.csv",index=False)
    (out/"test_setup.yaml").write_text(yaml.safe_dump({
        "configs_used":[str(cd/"daily_strategy.yaml"),str(cd/"weekly_strategy.yaml")],
        "selection":"Parameters selected on 2018-2021 validation and frozen before test.",
        "test_period":[2022,2025],
        "weekly_base_case_cost_bps":5.0,
        "weekly_cost_sensitivity_bps":[0.0,1.0,5.0,10.0],
        "buy_and_hold_cost":"Weekly Buy-and-Hold pays the 5 bps base-case cost on initial entry and terminal liquidation; Daily cost is zero by assumption.",
        "warning":"This test period has already been explored; retrospective reproducibility run, not an untouched holdout."
    },sort_keys=False),encoding="utf-8")
    print("="*72); print("INDIVIDUAL STRATEGY TEST COMPLETE")
    print(M[["pipeline","strategy","annualized_return","annualized_volatility","sharpe_ratio","max_drawdown","annualized_turnover"]].to_string(index=False))
    print(f"Results: {out}",flush=True)

if __name__=="__main__": main()
