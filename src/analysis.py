"""Does fixture congestion or recent workload raise in-match injury risk? Regression + descriptive charts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
P, CH = ROOT / "data" / "processed", ROOT / "charts"
CH.mkdir(exist_ok=True)

def rate_ci(k, n):
    p = k / n; se = np.sqrt(p * (1 - p) / n); return p, p - 1.96 * se, p + 1.96 * se

def main():
    pm = pd.read_parquet(P / "player_matches.parquet")
    tm = pd.read_csv(P / "team_matches.csv", parse_dates=["date"])
    out = {}

    # ---- 1. Validation: are "injury" substitutions real injuries? ----
    st = pm[(pm.started == 1) & pm.missed_next.notna()]
    miss_inj = st[st.injury_off == 1].missed_next.mean(); miss_other = st[st.injury_off == 0].missed_next.mean()
    out["missed_next_after_injury_sub"] = float(miss_inj); out["missed_next_otherwise"] = float(miss_other)

    # ---- 2. Team level: rest days vs injury substitutions ----
    t = tm.dropna(subset=["rest_days"]).copy()
    t = t[t.rest_days <= 21]
    t["rest"] = pd.cut(t.rest_days, [0, 3, 5, 7, 21], labels=["2-3 days", "4-5 days", "6-7 days", "8+ days"])
    tb = t.groupby("rest", observed=True).agg(matches=("injury_subs", "size"), injury_subs=("injury_subs", "sum"))
    tb["per_100_matches"] = tb.injury_subs / tb.matches * 100
    tb.to_csv(P / "rest_days_table.csv")
    t["rest"] = t.rest.astype(str)
    pois = smf.glm("injury_subs ~ C(rest, Treatment('6-7 days')) + C(league)", data=t, family=sm.families.Poisson()).fit()
    irr = pd.DataFrame({"irr": np.exp(pois.params), "lo": np.exp(pois.conf_int()[0]), "hi": np.exp(pois.conf_int()[1]), "p": pois.pvalues})
    irr.to_csv(P / "team_poisson_irr.csv")
    out["team_rest_table"] = tb.reset_index().astype({"rest": str}).round(3).to_dict("records")
    out["team_irr"] = irr.round(3).reset_index().rename(columns={"index": "term"}).to_dict("records")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ci = [rate_ci(r.injury_subs, r.matches * 1.0) for r in tb.itertuples()]
    ax.bar(tb.index.astype(str), tb.per_100_matches, color="#457b9d",
           yerr=[[(c[0] - c[1]) * 100 for c in ci], [(c[2] - c[0]) * 100 for c in ci]], capsize=5)
    for i, r in enumerate(tb.itertuples()): ax.text(i, 1, f"n={r.matches}", ha="center", color="white", fontsize=8)
    ax.set_ylabel("Injury substitutions per 100 team-matches"); ax.set_xlabel("Days since the team's previous league match")
    ax.set_title("Rest days vs in-match injuries (4 leagues, 2015-16)"); fig.tight_layout(); fig.savefig(CH / "01_rest_days.png", dpi=150); plt.close()

    # ---- 3. Player level: recent minutes vs injury in this match (starters, after 3+ apps) ----
    s = pm[(pm.started == 1) & (pm.apps_so_far >= 3)].copy()
    s["load14"] = pd.cut(s.mins_prev_14d, [-1, 90, 180, 270, 1000], labels=["0-90", "91-180", "181-270", "271+"]).astype(str)
    s["short_rest"] = (s.days_since_last <= 3).astype(int)
    s["pos"] = s.position.fillna("").map(lambda p: "GK" if "Goalkeeper" in p else "DEF" if "Back" in p else "MID" if "Midfield" in p else "FWD")
    s = s[s.pos != "GK"]
    s["intensity_per_min_28d"] = s.intensity_prev_28d / s.mins_prev_28d.replace(0, np.nan)
    lt = s.groupby("load14").agg(starts=("injury_off", "size"), injuries=("injury_off", "sum"))
    lt["per_1000_starts"] = lt.injuries / lt.starts * 1000
    lt = lt.loc[["0-90", "91-180", "181-270", "271+"]]
    lt.to_csv(P / "load_table.csv")
    logit = smf.logit("injury_off ~ C(load14, Treatment('91-180')) + short_rest + C(pos) + C(league)", data=s).fit(disp=0)
    orr = pd.DataFrame({"odds_ratio": np.exp(logit.params), "lo": np.exp(logit.conf_int()[0]), "hi": np.exp(logit.conf_int()[1]), "p": logit.pvalues})
    orr.to_csv(P / "player_logit_or.csv")
    out["player_load_table"] = lt.reset_index().round(3).to_dict("records")
    out["player_or"] = orr.round(3).reset_index().rename(columns={"index": "term"}).to_dict("records")
    out["n_starts"] = int(len(s)); out["n_injuries_model"] = int(s.injury_off.sum())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ci = [rate_ci(r.injuries, r.starts) for r in lt.itertuples()]
    ax.bar(lt.index, lt.per_1000_starts, color="#e76f51", yerr=[[(c[0] - c[1]) * 1000 for c in ci], [(c[2] - c[0]) * 1000 for c in ci]], capsize=5)
    ax.set_xlabel("League minutes played in the previous 14 days"); ax.set_ylabel("Injury substitutions per 1,000 starts")
    ax.set_title("Recent workload vs in-match injury (outfield starters)"); fig.tight_layout(); fig.savefig(CH / "02_workload.png", dpi=150); plt.close()

    # Forest plot of the logit
    f = orr[orr.index.str.contains("load14|short_rest|pos")]
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(f))
    ax.errorbar(f.odds_ratio, y, xerr=[f.odds_ratio - f.lo, f.hi - f.odds_ratio], fmt="o", color="#1d3557")
    ax.axvline(1, color="k", ls="--", lw=1); ax.set_yticks(y, [i.replace("C(load14, Treatment('91-180'))", "14-day mins").replace("C(pos)", "position") for i in f.index])
    ax.set_xlabel("Odds ratio for injury substitution (95% CI)"); ax.set_title("Logistic regression (league fixed effects)")
    fig.tight_layout(); fig.savefig(CH / "03_odds_ratios.png", dpi=150); plt.close()

    # ---- 4. When injuries happen ----
    inj = pm[pm.injury_off == 1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].hist(inj.injury_minute, bins=range(0, 100, 5), color="#2a9d8f"); ax[0].set_xlabel("Match minute of injury substitution"); ax[0].set_title("Injury substitutions by minute")
    mon = pm.assign(month=pm.date.dt.to_period("M")).groupby("month").agg(inj=("injury_off", "sum"), starts=("started", "sum"))
    (mon.inj / mon.starts * 1000).plot.bar(ax=ax[1], color="#264653", rot=45); ax[1].set_ylabel("per 1,000 starts"); ax[1].set_title("Injury substitutions by month")
    fig.tight_layout(); fig.savefig(CH / "04_timing.png", dpi=150); plt.close()
    out["first_half_share"] = float((inj.injury_minute < 45).mean())
    out["by_month"] = (mon.inj / mon.starts * 1000).round(2).astype(float).rename(index=str).to_dict()

    # ---- 5. Team congestion profile ----
    cong = tm.groupby(["league", "team"]).agg(matches=("match_id", "size"), short_turnarounds=("rest_days", lambda s: (s <= 3).sum()),
                                              injury_subs=("injury_subs", "sum")).reset_index()
    cong.to_csv(P / "team_congestion.csv", index=False)
    r = float(np.corrcoef(cong.short_turnarounds, cong.injury_subs)[0, 1]); out["team_corr_short_turnarounds_injuries"] = r
    fig, ax = plt.subplots(figsize=(8, 6))
    for lg, g in cong.groupby("league"): ax.scatter(g.short_turnarounds, g.injury_subs, label=lg, alpha=.7)
    ax.set_xlabel("League matches played on 3 or fewer days' rest"); ax.set_ylabel("Injury substitutions in season")
    ax.set_title(f"Team congestion vs injuries (r = {r:.2f})"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(CH / "05_team_congestion.png", dpi=150); plt.close()

    (P / "key_findings.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))

if __name__ == "__main__":
    main()
