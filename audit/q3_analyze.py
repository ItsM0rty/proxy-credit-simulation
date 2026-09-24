import pandas as pd

import harness as H

R = H.Path(__file__).parent / "results"
st = pd.read_csv(R / "q3_static.csv")
dy = pd.read_csv(R / "q3_dynamic.csv")
for df in (st, dy):
    df["cap"] = df["cap"].astype(str)
CAP_ORDER = ["0.15", "0.25", "0.35", "0.5", "none"]
pd.set_option("display.width", 250, "display.max_colwidth", 30)


def grid(df, col, fmt="{:.3f}", rng=True):
    f = (lambda x: H.mr(x, fmt)) if rng else (lambda x: fmt.format(x.median()))
    return df.groupby(["cap", "weight"])[col].apply(f).unstack("weight").reindex(CAP_ORDER)


def show(title, table):
    print(f"\n== {title} (rows: cap, cols: tempering weight; 20 seeds) ==")
    print(table.to_string())


st["proxy_share_pct"] = st.proxy_share_of_promises * 100
st["bundle_gap_pp"] = st["bundle_promised_%"] - st["bundle_true_%"]
st["causal_gap_pp"] = st.causal_promised_sum - st.causal_true_sum
show("proxy share of summed per-action promises, % [project: 24.5 at 0.25/0.4]", grid(st, "proxy_share_pct", "{:.1f}"))
show("bundle promised %", grid(st, "bundle_promised_%", "{:.1f}"))
show("bundle promised - true, pp [project: 31.9 - 31.5 = +0.4]", grid(st, "bundle_gap_pp", "{:.1f}"))
show("causal actions: summed promised - summed true, pp (negative = under-credited)", grid(st, "causal_gap_pp", "{:.1f}"))

b = dy[dy.setting == "baseline"]
for p in (0.25, 0.5, 1.0):
    show(f"promise kept t=10, p={p}", grid(b[b.p == p], "kept_t10", "{:.2f}"))
k = b.pivot_table(index=["cap", "weight", "seed"], columns="p", values="kept_t10")
k["partial_worst"] = (k[0.5] < k[1.0]) & (k[0.5] < k[0.25])
k["recovers_at_1"] = k[1.0] - k[0.5]
show("share of seeds where p=0.5 has the lowest kept (vs 0.25 and 1.0)",
     k.groupby(["cap", "weight"]).partial_worst.mean().unstack("weight").reindex(CAP_ORDER).round(2))
show("kept(p=1) - kept(p=0.5), t=10", grid(k.reset_index(), "recovers_at_1", "{:.2f}"))
h = b[b.p == 0.5]
show("proxy-heavy gap t=1, pp, p=0.5 [project 9.8]", grid(h, "gap_proxy_heavy_t1", "{:.1f}"))
show("proxy-heavy gap t=10, pp, p=0.5 [project 8.5]", grid(h, "gap_proxy_heavy_t10", "{:.1f}"))
show("causal-only gap t=1, pp, p=0.5 [project -7.2]", grid(h, "gap_causal_only_t1", "{:.1f}"))
show("noncausal share of carrier credit t=10, p=0.5", grid(h, "noncausal_t10", "{:.2f}"))
show("camera per-unit GLM coef t=10, p=0.5", grid(h, "camera_coef_t10", "{:.2f}"))
show("share of adopters at credit cap, t=1, p=0.5", grid(h, "at_cap_t1", "{:.2f}", rng=False))
for p in (0.25, 1.0):
    show(f"peak loss ratio, p={p}", grid(b[b.p == p], "peak_lr", "{:.2f}"))

sw = dy[dy.p == 0.5].pivot_table(index=["cap", "weight", "seed"], columns="setting", values="noncausal_t10")
print("\n== proxy true-effect sweep: noncausal share t=10 at camera effect 0 / 0.1 / 0.3 (median), p=0.5 ==")
print(sw.groupby(["cap", "weight"]).median().round(2).apply(
    lambda r: f"{r['baseline']:.2f} / {r['camera=0.1']:.2f} / {r['camera=0.3']:.2f}", axis=1)
    .unstack("weight").reindex(CAP_ORDER).to_string())
mono = (sw["baseline"] > sw["camera=0.1"]) & (sw["camera=0.1"] > sw["camera=0.3"])
show("share of seeds where noncausal share falls monotonically with camera effect",
     mono.groupby(["cap", "weight"]).mean().unstack("weight").reindex(CAP_ORDER).round(2))
