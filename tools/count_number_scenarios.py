# ===== Count scenarios per family (with per-tag breakdown) =====
from collections import Counter, OrderedDict
import os

# 0) Use your FAMILY_MAP (preferred) or build it from CANONICAL_14
if "FAMILY_MAP" in globals():
    _FAMILY_MAP = FAMILY_MAP
elif "CANONICAL_14" in globals():
    _FAMILY_MAP = {k: sorted(v) for k, v in CANONICAL_14.items()}
else:
    raise RuntimeError("Define FAMILY_MAP or CANONICAL_14 before running this cell.")

# 1) Reuse existing builder/worker if present; otherwise create minimal ones
try:
    builder
except NameError:
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    builder = NuPlanScenarioBuilder(
        data_root=os.getenv("NUPLAN_DATA_ROOT", "/data/nuplan-v1.1/trainval"),
        map_root=os.getenv("NUPLAN_MAPS_ROOT", "/data/nuplan-v1.1/maps"),
        sensor_root=os.getenv("NUPLAN_SENSOR_ROOT"),
        db_files=os.getenv("NUPLAN_DB_FILES"),
        map_version=os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
    )

try:
    worker
except NameError:
    from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
    worker = SingleMachineParallelExecutor(use_process_pool=False)  # notebook-safe

# If you loaded log_names earlier (nuplan_train.json), this will reuse it
log_names = log_names if "log_names" in globals() else None

# 2) ScenarioFilter helper (matches your working positional signature)
def _mk_filter(scenario_types, limit_total=None, shuffle=False):
    map_names=None; timestamp_threshold_s=None; ego_displacement_minimum_m=None
    expand_scenarios=False; remove_invalid_goals=False
    scenario_tokens=None
    ego_start_speed_threshold=None; ego_stop_speed_threshold=None; speed_noise_tolerance=None
    return (
        scenario_types, scenario_tokens, log_names, map_names,
        None,  # num_scenarios_per_type
        limit_total, timestamp_threshold_s, ego_displacement_minimum_m,
        expand_scenarios, remove_invalid_goals, shuffle,
        ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance
    )

def _scenario_id(s):
    for k in ("scenario_token", "token", "scenario_name", "log_name"):
        if hasattr(s, k):
            return str(getattr(s, k))
    return str(id(s))  # fallback

# 3) Count per family
def count_scenarios_by_family(family_map: dict, limit_total=None):
    results = OrderedDict()
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    for fam, tags in family_map.items():
        tags = [t for t in tags if t]  # clean
        if not tags:
            results[fam] = {"total_unique_scenarios": 0, "per_tag": {}}
            continue

        filt = ScenarioFilter(*_mk_filter(tags, limit_total=limit_total, shuffle=False))
        scenarios = builder.get_scenarios(filt, worker)

        # Unique scenarios (by token/name), and per-tag breakdown
        uniq = set()
        per_tag = Counter()
        for s in scenarios:
            uniq.add(_scenario_id(s))
            per_tag[getattr(s, "scenario_type", "UNKNOWN")] += 1

        results[fam] = {
            "total_unique_scenarios": len(uniq),
            "per_tag": dict(sorted(per_tag.items(), key=lambda x: (-x[1], x[0]))),
        }
    return results

# 4) Run and print nicely
LIMIT_TOTAL = None  # set to an int (e.g., 200) if you want to cap for speed
counts = count_scenarios_by_family(_FAMILY_MAP, limit_total=LIMIT_TOTAL)

print("=== Scenario counts by family ===")
for fam, info in counts.items():
    total = info["total_unique_scenarios"]
    print(f"\n{fam}: {total} scenarios")
    for tag, n in info["per_tag"].items():
        print(f"  • {tag}: {n}")


# === Summarize scenario counts per family (and per tag) + save CSV/JSON ===
from collections import Counter, OrderedDict
import pandas as pd, json, os
from pathlib import Path
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

OUT_DIR = Path("artifacts/family_stats"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# use your FAMILY_MAP (or CANONICAL_14) already defined
FAM = FAMILY_MAP if "FAMILY_MAP" in globals() else {k: sorted(v) for k, v in CANONICAL_14.items()}

# reuse builder/worker from earlier cells
assert "builder" in globals() and "worker" in globals()

def _mk_filter(scenario_types=None, limit_total=None, shuffle=False, scenario_tokens=None, log_names=None):
    map_names=None; timestamp_threshold_s=None; ego_displacement_minimum_m=None
    expand_scenarios=False; remove_invalid_goals=False
    ego_start_speed_threshold=None; ego_stop_speed_threshold=None; speed_noise_tolerance=None
    return (
        scenario_types, scenario_tokens, log_names, map_names,
        None,  # num_scenarios_per_type
        limit_total, timestamp_threshold_s, ego_displacement_minimum_m,
        expand_scenarios, remove_invalid_goals, shuffle,
        ego_start_speed_threshold, ego_stop_speed_threshold, speed_noise_tolerance
    )

def _sid(s):
    for k in ("scenario_token", "token", "scenario_name", "log_name"):
        if hasattr(s, k): return str(getattr(s, k))
    return str(id(s))

def summarize_counts(family_map, limit_total=None):
    rows = []
    for fam, tags in family_map.items():
        fam_ids = set()
        per_tag = Counter()
        for tag in tags:
            filt = ScenarioFilter(*_mk_filter(scenario_types=[tag], limit_total=limit_total, shuffle=False,
                                              log_names=globals().get("log_names")))
            scens = builder.get_scenarios(filt, worker)
            ids = { _sid(s) for s in scens }
            per_tag[tag] = len(ids)
            fam_ids |= ids

        rows.append({"family": fam, "tag": "__TOTAL__", "count": len(fam_ids)})
        for tag, n in per_tag.items():
            rows.append({"family": fam, "tag": tag, "count": n})
    df = pd.DataFrame(rows).sort_values(["family","tag"])
    return df

df = summarize_counts(FAM, limit_total=None)
print("=== Scenario counts by family ===")

for fam in df["family"].unique():
    total = int(df[(df.family==fam) & (df.tag=="__TOTAL__")]["count"].iloc[0])
    print(f"\n{fam}: {total} scenarios")
    for _, r in df[(df.family==fam) & (df.tag!="__TOTAL__")].sort_values("count", ascending=False).iterrows():
        print(f"{r.tag}: {(r.count)}")

# save
csv_fp = OUT_DIR / "family_counts.csv"
json_fp = OUT_DIR / "family_counts.json"
df.to_csv(csv_fp, index=False)
with open(json_fp, "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, indent=2)
print(f"\nSaved summary to:\n  - {csv_fp}\n  - {json_fp}")
