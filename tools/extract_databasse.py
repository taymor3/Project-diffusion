# === Build ONE DB (Parquet) per family, each with N scenarios (mini split) ===
# - Mines windows directly from nuPlan .db (works on mini).
# - For each family in FAMILY_MAP, samples PER_FAMILY[family] anchors across all DBs.
# - Writes: cache/family_dbs/<family>.parquet  (many scenarios inside)
#
# Columns (subset): family, scenario_name, scenario_tag, iteration, time_s, rel_time_s,
#                   agent_id, agent_type, x, y, yaw, length, width, speed

import os, sqlite3, glob, math, random, gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------- Paths / IO ----------------
DB_ROOT  = "/home/taimor/data1/nuplan-v1.1/splits/mini"   # folder with *.db
OUT_DIR  = Path("./cache/family_dbs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Family map (yours) ----------------
FAMILY_MAP = {
    "turn_left": ["starting_left_turn"],
    "turn_right": ["starting_right_turn"],
    "straight_through_intersection": [
        "traversing_intersection",
        "traversing_traffic_light_intersection",
        "on_intersection",
        "on_all_way_stop_intersection",
        "on_traffic_light_intersection",
        "starting_straight_stop_sign_intersection_traversal",
        "starting_straight_traffic_light_intersection_traversal",
        "on_stopline_stop_sign",
    ],
    "lane_change_left": ["changing_lane_to_left"],
    "lane_change_right": ["changing_lane_to_right"],
    "stop_for_red_light": [
        "on_stopline_traffic_light",
        "stationary_at_traffic_light_with_lead",
        "stationary_at_traffic_light_without_lead",
        "stopping_at_traffic_light_with_lead",
        "stopping_at_traffic_light_without_lead",
    ],
    "go_on_green": [
        "accelerating_at_traffic_light",
        "accelerating_at_traffic_light_with_lead",
        "accelerating_at_traffic_light_without_lead",
        "starting_straight_traffic_light_intersection_traversal",
    ],
    "yield_to_pedestrian": [
        "near_pedestrian_on_crosswalk",
        "near_pedestrian_on_crosswalk_with_ego",
        "stationary_at_crosswalk",
        "stopping_at_crosswalk",
        "traversing_crosswalk",
        "waiting_for_pedestrian_to_cross",
        "on_stopline_crosswalk",
        "behind_pedestrian_on_driveable",
        "near_pedestrian_at_pickup_dropoff",
    ],
    "cut_in": [
        "near_long_vehicle",
        "near_high_speed_vehicle",
        "near_multiple_vehicles",
    ],
    "car_following": [
        "stationary_in_traffic",
        "stationary",
        "following_lane_with_lead",
        "following_lane_with_slow_lead",
        "following_lane_without_lead",
        "stopping_with_lead",
        "behind_bike",
    ],
    "pickup_dropoff": [
        "on_pickup_dropoff",
        "traversing_pickup_dropoff",
        "behind_pedestrian_on_pickup_dropoff",
        "near_pedestrian_at_pickup_dropoff",
        "on_carpark",
    ],
    "construction_zone": [
        "near_construction_zone_sign",
        "near_trafficcone_on_driveable",
        "near_barrier_on_driveable",
        "traversing_narrow_lane",
    ],
    "speed_events": [
        "high_magnitude_speed",
        "medium_magnitude_speed",
        "low_magnitude_speed",
        "high_magnitude_jerk",
        "high_lateral_acceleration",
    ],
}
number_of_scenarios_per_family = 3
# ---------------- How many scenarios per family? ----------------
# Set an integer for each family you want. Omit a family (or set 0) to skip it.
PER_FAMILY: Dict[str, int] = {
    "turn_left": number_of_scenarios_per_family,
    "turn_right": number_of_scenarios_per_family,
    "pickup_dropoff": number_of_scenarios_per_family,
    "car_following": number_of_scenarios_per_family,
    "speed_events": number_of_scenarios_per_family,
    "construction_zone": number_of_scenarios_per_family,
    "car_following": number_of_scenarios_per_family,
    "cut_in": number_of_scenarios_per_family,
    "yield_to_pedestrian": number_of_scenarios_per_family,
    "go_on_green": number_of_scenarios_per_family,
    "stop_for_red_light": number_of_scenarios_per_family,
    "lane_change_left": number_of_scenarios_per_family,
    "lane_change_right": number_of_scenarios_per_family,
    "straight_through_intersection": number_of_scenarios_per_family,
}
SHUFFLE_SEED = 123  # set None for non-deterministic

# ---------------- Window / sampling controls ----------------
HISTORY_S = 2.0      # seconds before anchor
FUTURE_S  = 6.0      # seconds after anchor
SUBSAMPLE_EVERY = 1  # 1=keep all frames (~20Hz), 2=~10Hz, 4=~5Hz

# ================== Helper funcs ==================
def list_db_files(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**/*.db"), recursive=True))

def to_secs(ts):
    return (float(ts)/1e6) if ts is not None and float(ts) > 1e10 else (float(ts) if ts is not None else None)

def yaw_from_quat(qw,qx,qy,qz):
    return math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

def hex_or_none(b): return b.hex() if isinstance(b, (bytes, bytearray)) else None

def get_columns(con, table):
    try: return [r[1] for r in con.execute(f"PRAGMA table_info({table});")]
    except: return []

def anchor_table(con):
    tabs = [t[0] for t in con.execute("SELECT name FROM sqlite_master WHERE type='table';")]
    if "scenario_tag" in tabs: return "scenario_tag"
    if "scenario" in tabs:     return "scenario"
    raise RuntimeError("No scenario_tag/scenario table")

def load_anchors(con):
    t = anchor_table(con)
    cols = get_columns(con, t)
    anchor_col = next((c for c in cols if "lidar" in c.lower() and "token" in c.lower()), None) \
                 or next((c for c in ["pc_token","anchor_token"] if c in cols), None)
    label_col  = next((c for c in ["name","type","scenario_type","category"] if c in cols), None)
    if not anchor_col: raise RuntimeError(f"No anchor token column in {t}")
    q = f"SELECT {anchor_col} AS anchor_pc{(', ' + label_col) if label_col else ''} FROM {t};"
    df = pd.read_sql_query(q, con)
    df.rename(columns={label_col: "label"}, inplace=True, errors="ignore")
    return df  # columns: anchor_pc (bytes), label (optional)

def ego_time_for_pc(con, pc_token):
    r = con.execute("SELECT ego_pose_token FROM lidar_pc WHERE token = ?;", (pc_token,)).fetchone()
    if not r: return None
    r2 = con.execute("SELECT timestamp FROM ego_pose WHERE token = ?;", (r[0],)).fetchone()
    return to_secs(r2[0]) if r2 else None

def step_prev(con, pc_tok):
    r = con.execute("SELECT prev_token FROM lidar_pc WHERE token = ?;", (pc_tok,)).fetchone()
    return r[0] if r and r[0] is not None else None

def step_next(con, pc_tok):
    r = con.execute("SELECT next_token FROM lidar_pc WHERE token = ?;", (pc_tok,)).fetchone()
    return r[0] if r and r[0] is not None else None

def collect_window(con, anchor_pc, hist_s, fut_s):
    t_anchor = ego_time_for_pc(con, anchor_pc)
    if t_anchor is None: return []
    tmin, tmax = t_anchor - hist_s, t_anchor + fut_s
    seq = []

    # back
    tok = anchor_pc
    while True:
        prev = step_prev(con, tok)
        if not prev: break
        tt = ego_time_for_pc(con, prev)
        if tt is None or tt < tmin: break
        seq.append((prev, tt))
        tok = prev
    seq.sort(key=lambda x: x[1])

    # anchor
    seq.append((anchor_pc, t_anchor))

    # forward
    tok = anchor_pc
    while True:
        nxt = step_next(con, tok)
        if not nxt: break
        tt = ego_time_for_pc(con, nxt)
        if tt is None or tt > tmax: break
        seq.append((nxt, tt))
        tok = nxt

    seq.sort(key=lambda x: x[1])

    # optional subsample
    if SUBSAMPLE_EVERY > 1:
        seq = seq[::SUBSAMPLE_EVERY]
    return seq

def load_boxes_for_pc(con, pc_token):
    cols_lb = get_columns(con, "lidar_box")
    if not cols_lb: return pd.DataFrame()
    cols_trk = get_columns(con, "track")
    cols_cat = get_columns(con, "category")

    want = [c for c in ["token","lidar_pc_token","track_token","x","y","z","yaw","qw","qx","qy","qz","length","width","height","vx","vy","vz"] if c in cols_lb]
    sel  = ", ".join("lb."+c for c in want) if want else "lb.*"
    join_trk = "LEFT JOIN track t ON t.token = lb.track_token" if ("track_token" in cols_lb and "token" in cols_trk) else ""
    join_cat = "LEFT JOIN category c ON c.token = t.category_token" if ("category_token" in cols_trk and "token" in cols_cat) else ""
    sel_extra = ", c.name as category_name" if ("name" in cols_cat) else ""

    q = f"SELECT {sel}{sel_extra} FROM lidar_box lb {join_trk} {join_cat} WHERE lb.lidar_pc_token = ?"
    df = pd.read_sql_query(q, con, params=(pc_token,))

    # normalize
    if "yaw" not in df.columns and set(["qw","qx","qy","qz"]).issubset(df.columns):
        df["yaw"] = [yaw_from_quat(*row[["qw","qx","qy","qz"]]) for _,row in df.iterrows()]
    for k,alts in {"length":["length","l","size_x"],"width":["width","w","size_y"]}.items():
        if k not in df.columns:
            for a in alts:
                if a in df.columns:
                    df.rename(columns={a:k}, inplace=True)
                    break
    if "speed" not in df.columns and {"vx","vy"}.issubset(df.columns):
        df["speed"] = np.hypot(df["vx"].astype(float).fillna(0), df["vy"].astype(float).fillna(0))
    if "category_name" not in df.columns:
        df["category_name"] = "unknown"

    if "track_token" in df.columns:
        df["agent_id"] = df["track_token"].apply(hex_or_none)
    elif "token" in df.columns:
        df["agent_id"] = df["token"].apply(hex_or_none)
    else:
        df["agent_id"] = None

    keep = ["agent_id","x","y","yaw","length","width","speed","category_name"]
    for k in keep:
        if k not in df.columns: df[k] = np.nan
    df.rename(columns={"category_name":"agent_type"}, inplace=True)
    return df[["agent_id","agent_type","x","y","yaw","length","width","speed"]]

def export_window_for_anchor(con, db_name: str, anchor_pc, family: str, scenario_tag: Optional[str]):
    seq = collect_window(con, anchor_pc, HISTORY_S, FUTURE_S)
    if not seq: return pd.DataFrame()
    t0 = seq[0][1]
    scen_id = f"{db_name}_{hex_or_none(anchor_pc)}"  # unique across DBs

    rows = []
    for i,(pc,t) in enumerate(seq):
        dfb = load_boxes_for_pc(con, pc)
        if dfb.empty: 
            continue
        step = dfb.copy()
        step["time_s"]        = float(t)
        step["rel_time_s"]    = float(t - t0)
        step["iteration"]     = i
        step["scenario_name"] = scen_id
        step["family"]        = family
        step["scenario_tag"]  = scenario_tag or ""
        rows.append(step)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ================== 1) Build reverse map: tag -> family ==================
TAG_TO_FAMILY: Dict[str, str] = {}
for fam, tags in FAMILY_MAP.items():
    for tag in tags:
        TAG_TO_FAMILY[tag] = fam

# ================== 2) Gather candidate anchors per family ==================
db_files = list_db_files(DB_ROOT)
if not db_files:
    raise FileNotFoundError(f"No .db files under {DB_ROOT}")

candidates: Dict[str, List[Tuple[str, bytes, str]]] = {fam: [] for fam in PER_FAMILY.keys()}  # fam -> [(db_path, anchor_pc, tag), ...]

for db in db_files:
    con = sqlite3.connect(db)
    try:
        anchors = load_anchors(con)  # columns: anchor_pc (bytes), label (maybe None)
        for _, row in anchors.iterrows():
            tag = str(row.get("label")) if "label" in anchors.columns else None
            fam = TAG_TO_FAMILY.get(tag)
            if fam in candidates:
                candidates[fam].append((db, row["anchor_pc"], tag))
    finally:
        con.close()

# ================== 3) Sample N per family ==================
if SHUFFLE_SEED is not None:
    random.seed(SHUFFLE_SEED)

selected: Dict[str, List[Tuple[str, bytes, str]]] = {}
for fam, want in PER_FAMILY.items():
    pool = candidates.get(fam, [])
    if not pool:
        print(f"[warn] no anchors found for family '{fam}' in mini split.")
        continue
    random.shuffle(pool)
    chosen = pool[:min(want, len(pool))]
    selected[fam] = chosen
    print(f"[select] {fam}: picked {len(chosen)} / {len(pool)} anchors")

# ================== 4) Export and write ONE Parquet per family ==================
for fam, anchors in selected.items():
    print(f"\n[export] family='{fam}' scenarios={len(anchors)}")
    fam_rows = []
    for (db_path, pc, tag) in tqdm(anchors, desc=fam, leave=False):
        con = sqlite3.connect(db_path)
        try:
            df = export_window_for_anchor(con, Path(db_path).stem, pc, fam, tag)
        finally:
            con.close()
        if df.empty:
            continue
        fam_rows.append(df)
    if not fam_rows:
        print(f"  nothing exported for {fam}")
        continue
    fam_df = pd.concat(fam_rows, ignore_index=True)
    out_fp = OUT_DIR / f"{fam}.parquet"
    fam_df.to_parquet(out_fp, index=False)
    print(f"  saved {len(fam_df):,} rows across {fam_df['scenario_name'].nunique()} scenarios -> {out_fp}")

    # quick sanity
    by_scen = fam_df.groupby("scenario_name")["iteration"].nunique().describe()
    print("  per-scenario frame counts (summary):")
    print(by_scen)



def export_families_database(number_of_scenarios_per_family=3):
    check_files = list((OUT_DIR).glob("*.parquet"))
    if check_files:
        fp = check_files[0]
        df = pd.read_parquet(fp)
        scen = df["scenario_name"].iloc[0]
        g = df[df["scenario_name"] == scen]
        print(f"\n[sanity] {fp.name}: scenarios={df['scenario_name'].nunique()}  rows={len(df)}")
        print(" frames in first scenario:", g["iteration"].nunique(), " unique time_s:", g["time_s"].nunique())
        print(" first 10 rel_time_s:", sorted(g['rel_time_s'].unique())[:10])
    else:
        print("\nNo family parquet files written — check PER_FAMILY and mappings.")
