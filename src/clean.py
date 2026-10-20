"""Build player-match and team-match tables with workload and injury-substitution outcomes."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
LEAGUE = {"premier_league_2015_16": "Premier League", "la_liga_2015_16": "La Liga",
          "serie_a_2015_16": "Serie A", "ligue_1_2015_16": "Ligue 1"}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, team_rows = [], []
    for lab, league in LEAGUE.items():
        ev = pd.read_parquet(RAW / f"{lab}.parquet")
        mt = pd.read_csv(RAW / f"{lab}_matches.csv", parse_dates=["match_date"]).set_index("match_id")
        intensity = ev[ev.type.isin(["Pressure", "Duel", "Dribble"])].groupby(["match_id", "player_id"]).size()
        for mid, g in ev.groupby("match_id"):
            date = mt.loc[mid, "match_date"]; end = g.minute.max() + 1
            on = {}
            for r in g[g.type == "Starting XI"].itertuples():
                for pid, name, pos in json.loads(r.extra):
                    on[pid] = dict(player=name, team=r.team, start=0, end=end, started=1, position=pos, injury_off=0, injury_minute=np.nan)
            for r in g[g.type == "Substitution"].itertuples():
                rid, rname = json.loads(r.extra)
                if r.player_id in on:
                    on[r.player_id]["end"] = r.minute
                    if r.outcome == "Injury":
                        on[r.player_id]["injury_off"] = 1; on[r.player_id]["injury_minute"] = r.minute
                on[rid] = dict(player=rname, team=r.team, start=r.minute, end=end, started=0, position=r.position, injury_off=0, injury_minute=np.nan)
            for pid, d in on.items():
                rows.append({"league": league, "match_id": mid, "date": date, "player_id": pid, **d,
                             "minutes": max(d["end"] - d["start"], 0), "intensity_actions": int(intensity.get((mid, pid), 0))})
            for team in g.team.unique():
                team_rows.append({"league": league, "match_id": mid, "date": date, "team": team,
                                  "injury_subs": int(sum(1 for d in on.values() if d["team"] == team and d["injury_off"]))})
    pm = pd.DataFrame(rows).sort_values(["player_id", "date"])
    tm = pd.DataFrame(team_rows).sort_values(["team", "date"])
    tm["rest_days"] = tm.groupby("team").date.diff().dt.days

    # Rolling workload from the player's previous league matches (strictly before this match)
    def prior(g):
        g = g.set_index("date")
        out = {}
        for w in (7, 14, 28):
            out[f"mins_prev_{w}d"] = g.minutes.rolling(f"{w}D", closed="left").sum().fillna(0).values
        out["intensity_prev_28d"] = g.intensity_actions.rolling("28D", closed="left").sum().fillna(0).values
        out["days_since_last"] = g.index.to_series().diff().dt.days.values
        out["apps_so_far"] = np.arange(len(g))
        return pd.DataFrame(out, index=g.index)
    pr = pm.groupby("player_id", group_keys=False)[["date", "minutes", "intensity_actions"]].apply(prior)
    pm = pm.reset_index(drop=True)
    pr = pr.reset_index(drop=True)
    pm = pd.concat([pm, pr], axis=1)
    # Availability proxy: a player who started this match - does he appear in his team's NEXT league match?
    nxt = tm.assign(next_match=tm.groupby("team").match_id.shift(-1))[["match_id", "team", "next_match"]]
    pm = pm.merge(nxt, on=["match_id", "team"], how="left")
    played = set(zip(pm.match_id, pm.player_id))
    pm["missed_next"] = [np.nan if pd.isna(n) else float((n, p) not in played) for n, p in zip(pm.next_match, pm.player_id)]
    pm = pm.merge(tm[["match_id", "team", "rest_days"]], on=["match_id", "team"], how="left")
    pm.to_parquet(OUT / "player_matches.parquet", index=False)
    tm.to_csv(OUT / "team_matches.csv", index=False)
    print(len(pm), "player-matches;", int(pm.injury_off.sum()), "injury substitutions;", len(tm), "team-matches")

if __name__ == "__main__":
    main()
