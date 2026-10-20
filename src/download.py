"""Download StatsBomb open data events for chosen competitions and keep a compact event table.

Full event JSONs are ~3-4 MB per match, so each file is parsed and reduced on the fly to the columns
this project needs, then saved as one parquet file per competition-season in data/raw/.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
import requests

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
# (competition_id, season_id, label)
COMPETITIONS = [(2, 27, "premier_league_2015_16"), (11, 27, "la_liga_2015_16"), (12, 27, "serie_a_2015_16"),
                (7, 27, "ligue_1_2015_16")]
KEEP_TYPES = {"Pass", "Carry", "Shot", "Dribble", "Ball Receipt*", "Interception", "Ball Recovery",
              "Duel", "Clearance", "Block", "Pressure", "Starting XI", "Substitution", "Foul Won", "Dispossessed", "Miscontrol"}

def get(url):
    for _ in range(4):
        try:
            r = requests.get(url, timeout=90); r.raise_for_status(); return r.json()
        except Exception as e:
            err = e
    raise err

def reduce_match(match):
    mid = match["match_id"]
    ev = get(f"{BASE}/events/{mid}.json")
    rows = []
    for e in ev:
        t = e["type"]["name"]
        if t not in KEEP_TYPES:
            continue
        sub = e.get(t.lower().replace(" ", "_").replace("*", ""), {}) or {}
        row = {"match_id": mid, "index": e["index"], "period": e["period"], "minute": e["minute"], "second": e["second"],
               "type": t, "team": e["team"]["name"], "possession_team": e["possession_team"]["name"], "possession": e["possession"],
               "player_id": e.get("player", {}).get("id"), "player": e.get("player", {}).get("name"),
               "position": e.get("position", {}).get("name"), "under_pressure": bool(e.get("under_pressure", False)),
               "x": (e.get("location") or [None, None])[0], "y": (e.get("location") or [None, None])[1],
               "end_x": None, "end_y": None, "outcome": (sub.get("outcome") or {}).get("name"),
               "subtype": (sub.get("type") or {}).get("name"), "xg": sub.get("statsbomb_xg"),
               "body_part": (sub.get("body_part") or {}).get("name"), "cross": bool(sub.get("cross", False)),
               "pass_height": (sub.get("height") or {}).get("name"), "duration": e.get("duration")}
        if "end_location" in sub:
            row["end_x"], row["end_y"] = sub["end_location"][0], sub["end_location"][1]
        if t == "Starting XI":
            row["extra"] = json.dumps([[p["player"]["id"], p["player"]["name"], p["position"]["name"]] for p in e["tactics"]["lineup"]])
        elif t == "Substitution":
            row["extra"] = json.dumps([sub["replacement"]["id"], sub["replacement"]["name"]])
        rows.append(row)
    return rows

def main(comps=COMPETITIONS):
    RAW.mkdir(parents=True, exist_ok=True)
    for cid, sid, label in comps:
        out = RAW / f"{label}.parquet"
        if out.exists():
            print("exists", out.name); continue
        matches = get(f"{BASE}/matches/{cid}/{sid}.json")
        pd.json_normalize(matches).to_csv(RAW / f"{label}_matches.csv", index=False)
        with ThreadPoolExecutor(12) as ex:
            res = list(ex.map(reduce_match, matches))
        df = pd.DataFrame([r for m in res for r in m])
        df.to_parquet(out, index=False)
        print(label, len(matches), "matches", len(df), "events")

if __name__ == "__main__":
    main()
