from __future__ import annotations
import argparse, itertools, json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ALPHA = {"daily": 0.000225, "weekly": 0.0003}
PPY = {"daily": 252.0, "weekly": 50.4}
K = [0.0025, 0.005, 0.01]
DELTA = [0.0, 0.0005, 0.001]
BASE = [0.25, 0.50, 0.75]
TARGET_VOL = [0.10, 0.12, 0.15]
DOWNSIDE = [1.0, 2.0, 4.0]
LOOKBACK = {"daily": [20, 60], "weekly": [4, 8, 12]}
WEEKLY_BASE_COST_BPS = 5.0


def read_frame(path, required, allow_duplicates=False):
    if not path.is_file():
        raise FileNotFoundError(path)
    d = pd.read_csv(path, parse_dates=["decision_date"])
    missing = sorted(set(required) - set(d.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    d = d.sort_values("decision_date").reset_index(drop=True)
    if d.empty or (not allow_duplicates and d.decision_date.duplicated().any()):
        raise ValueError(f"{path} is empty or has duplicate decision dates.")
    return d


def load_validation(folder, pipeline):
    pred_name = "alpha_refinement_predictions.csv" if pipeline == "daily" else "weekly_alpha_predictions.csv"
    pred = read_frame(folder / pred_name,
                      ["decision_date", "actual_return", "predicted_return", "alpha"], True)
    a = pd.to_numeric(pred.alpha, errors="coerce")
    pred = pred.loc[np.isclose(a, ALPHA[pipeline], rtol=0, atol=1e-12)].copy()
    pred = pred.sort_values("decision_date").reset_index(drop=True)
    if pred.empty or pred.decision_date.duplicated().any():
        raise ValueError(f"{pipeline}: no unique rows for frozen alpha {ALPHA[pipeline]}")
    hm = read_frame(folder / "monthly_historical_mean_predictions.csv",
                    ["decision_date", "predicted_return"])
    hm = hm[["decision_date", "predicted_return"]].rename(
        columns={"predicted_return": "historical_mean_prediction"})
    out = pred[["decision_date", "actual_return", "predicted_return"]].merge(
        hm, on="decision_date", how="left", validate="one_to_one")
    if out.historical_mean_prediction.isna().any():
        raise ValueError(f"{pipeline}: historical mean dates do not align.")
    return out


def pos_original(x, p):
    return np.clip(np.maximum(x - p["delta"], 0) / p["k"], 0, 1)


def pos_a(x, p):
    return np.clip(p["base_exposure"] + x / p["k"], 0, 1)


def pos_c(x, p):
    mult = np.where(x >= 0, 1.0, p["downside_multiplier"])
    return np.clip(p["base_exposure"] + x / p["k"] * mult, 0, 1)


def pos_b(pred, actual, p, pipeline, prior=None):
    prior = np.asarray([] if prior is None else prior, dtype=float)
    all_r = np.r_[prior, np.asarray(actual, dtype=float)]
    nprior = len(prior)
    w = np.empty(len(actual))
    for i, x in enumerate(pred):
        end = max(0, nprior + i - 1)  # latest included label is two decisions old
        h = all_r[max(0, end - int(p["lookback"])):end]
        rv = np.std(h, ddof=1) * np.sqrt(PPY[pipeline]) if len(h) >= 2 else p["target_vol"]
        if not np.isfinite(rv) or rv <= 1e-12:
            rv = p["target_vol"]
        vol_scale = min(1.0, p["target_vol"] / rv)
        signal_scale = np.clip(1.0 + x / p["k"], 0.0, 1.5)
        w[i] = np.clip(vol_scale * signal_scale, 0.0, 1.0)
    return w


def weights(frame, pipeline, method, source, p):
    x = frame["predicted_return" if source == "model" else "historical_mean_prediction"].to_numpy(float)
    y = frame.actual_return.to_numpy(float)
    if method == "original_mapping": return pos_original(x, p)
    if method == "strategy_a": return pos_a(x, p)
    if method == "strategy_b": return pos_b(x, y, p, pipeline)
    if method == "strategy_c": return pos_c(x, p)
    raise ValueError(method)


def returns_and_metrics(frame, w, pipeline, cost_bps):
    w = np.asarray(w, float)
    if len(w) != len(frame) or not np.isfinite(w).all() or np.any((w < 0) | (w > 1)):
        raise ValueError("Invalid weights; expected finite [0,1] positions.")
    turnover = np.abs(np.diff(np.r_[0.0, w]))
    if len(turnover): turnover[-1] += abs(w[-1])  # terminal liquidation
    cost = turnover * cost_bps / 10000.0
    gross = w * frame.actual_return.to_numpy(float)
    net = gross - cost
    equity = np.cumprod(1 + net)
    dd = equity / np.maximum.accumulate(equity) - 1
    sd = np.std(net, ddof=1) if len(net) > 1 else 0.0
    n = len(net)
    m = {
        "n_periods": n, "total_return": equity[-1] - 1,
        "annualized_return": equity[-1] ** (PPY[pipeline] / n) - 1 if equity[-1] > 0 else np.nan,
        "annualized_volatility": sd * np.sqrt(PPY[pipeline]),
        "sharpe_ratio": np.mean(net) / sd * np.sqrt(PPY[pipeline]) if sd > 0 else np.nan,
        "max_drawdown": dd.min(),
        "annualized_turnover": turnover.sum() / (n / PPY[pipeline]),
        "average_position": np.mean(w), "total_transaction_cost": cost.sum()
    }
    path = pd.DataFrame({
        "decision_date": frame.decision_date, "weight": w, "turnover": turnover,
        "gross_return": gross, "transaction_cost": cost, "net_return": net,
        "equity": equity, "drawdown": dd
    })
    return m, path


def candidates(pipeline, method):
    # Transaction cost is an exogenous assumption, not a parameter to optimize.
    costs = [0.0] if pipeline == "daily" else [WEEKLY_BASE_COST_BPS]
    if method == "original_mapping":
        return [dict(k=k, delta=d, cost_bps=c) for k,d,c in itertools.product(K, DELTA, costs)]
    if method == "strategy_a":
        return [dict(base_exposure=b, k=k, cost_bps=c) for b,k,c in itertools.product(BASE,K,costs)]
    if method == "strategy_b":
        return [dict(target_vol=v, lookback=l, k=k, cost_bps=c)
                for v,l,k,c in itertools.product(TARGET_VOL,LOOKBACK[pipeline],K,costs)]
    if method == "strategy_c":
        return [dict(base_exposure=b,k=k,downside_multiplier=d,cost_bps=c)
                for b,k,d,c in itertools.product(BASE,K,DOWNSIDE,costs)]
    raise ValueError(method)


def search_method(frame, pipeline, method, source):
    rows = []
    for i,p in enumerate(candidates(pipeline, method),1):
        w = weights(frame,pipeline,method,source,p)
        m,_ = returns_and_metrics(frame,w,pipeline,p["cost_bps"])
        rows.append({"pipeline":pipeline,"signal_source":source,"strategy":method,
                     "candidate_id":i,"params":json.dumps(p,sort_keys=True),**m})
    result = pd.DataFrame(rows).sort_values(
        ["sharpe_ratio","annualized_return","max_drawdown"],
        ascending=[False,False,False],kind="stable").reset_index(drop=True)
    return result, result.iloc[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--daily-validation-dir",type=Path,required=True)
    ap.add_argument("--weekly-validation-dir",type=Path,required=True)
    ap.add_argument("--config-dir",type=Path,default=Path("configs"))
    ap.add_argument("--output-dir",type=Path,default=Path("outputs"))
    a=ap.parse_args()
    cd=a.config_dir if a.config_dir.is_absolute() else Path.cwd()/a.config_dir
    od=a.output_dir if a.output_dir.is_absolute() else Path.cwd()/a.output_dir
    cd.mkdir(parents=True,exist_ok=True)
    out=od/f"strategy_search_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True,exist_ok=False)
    all_results=[]
    for pipeline,folder in [("daily",a.daily_validation_dir.resolve()),("weekly",a.weekly_validation_dir.resolve())]:
        frame=load_validation(folder,pipeline)
        selected={}
        print(f"{pipeline.upper()} validation rows={len(frame)} | alpha={ALPHA[pipeline]}",flush=True)
        for method in ["original_mapping","strategy_a","strategy_b","strategy_c"]:
            result,best=search_method(frame,pipeline,method,"model")
            all_results.append(result)
            selected[method]={
                "params":json.loads(best["params"]),
                "validation_metrics":{k:float(best[k]) for k in ["annualized_return","annualized_volatility","sharpe_ratio","max_drawdown"]},
                "candidate_count":len(result),
                "selection_rule":"maximum validation Sharpe; tie-break annualized return then max drawdown"
            }
            print(f"  {method}: {len(result)} candidates | Sharpe={best['sharpe_ratio']:.4f} | {best['params']}",flush=True)
        hm_result,hm_best=search_method(frame,pipeline,"original_mapping","historical_mean")
        all_results.append(hm_result)
        selected["historical_mean_benchmark"]={
            "mapping":"original_mapping","params":json.loads(hm_best["params"]),
            "validation_metrics":{k:float(hm_best[k]) for k in ["annualized_return","annualized_volatility","sharpe_ratio","max_drawdown"]},
            "candidate_count":len(hm_result)
        }
        cfg={
            "pipeline":pipeline,"model":"lasso","frozen_model_alpha":ALPHA[pipeline],
            "validation_period":[2018,2021],"training_window":"rolling_5_years",
            "retraining":"monthly","selection_metric":"validation_sharpe_ratio",
            "selection_rule":"maximum validation Sharpe; tie-break annualized return then max drawdown",
            "strategies":selected,
            "position_constraints":{"min_weight":0.0,"max_weight":1.0,"long_only":True,"leverage":False,"cash_return":0.0},
            "transaction_cost_assumption":{"daily_bps":0.0,"weekly_base_case_bps":WEEKLY_BASE_COST_BPS}
        }
        path=cd/f"{pipeline}_strategy.yaml"
        path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding="utf-8")
        print(f"  config saved: {path}",flush=True)
    pd.concat(all_results,ignore_index=True).to_csv(out/"validation_strategy_search.csv",index=False)
    (out/"search_setup.yaml").write_text(yaml.safe_dump({
        "selection_period":[2018,2021],"selection_metric":"Sharpe",
        "grids":{"k":K,"delta":DELTA,"base_exposure":BASE,"target_vol":TARGET_VOL,
                 "downside_multiplier":DOWNSIDE,"daily_lookback":LOOKBACK["daily"],
                 "weekly_lookback":LOOKBACK["weekly"],"weekly_base_cost_bps":WEEKLY_BASE_COST_BPS},
        "frozen_alpha":ALPHA,"warning":"2022-2025 was previously inspected; subsequent test is retrospective."
    },sort_keys=False),encoding="utf-8")
    print(f"Strategy search complete: {out}",flush=True)

if __name__=="__main__": main()
