"""
Minimal canonical family viewer for nuPlan tutorials.

- Uses ONLY the public tutorial helpers (no edits to tutorial_utils).
- Shows a dropdown of the 14 canonical families (optionally only those that exist).
- On click, picks a random/cycled example from that family and renders it.
- Old Bokeh output is cleared between runs by owning a dedicated ipywidgets.Output area.

Place this file at: tools/canonical_family_viewer.py
Then, in a notebook cell, do:

    from tools.canonical_family_viewer import launch_canonical_viewer
    launch_canonical_viewer(
        data_root=os.getenv("NUPLAN_DATA_ROOT"),
        map_root=os.getenv("NUPLAN_MAPS_ROOT"),
        map_version=os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0"),
        show_only_available=True,
        fixed_port=8899,
    )

"""
from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import ipywidgets as widgets
from IPython.display import display, clear_output

# bokeh + tutorial hooks
from bokeh.io import output_notebook
from tutorials.utils.tutorial_utils import (
    setup_notebook,
    discover_log_dbs,
    get_scenario_type_token_map,
    get_default_scenario_from_token,
    visualize_scenario,
)
# To be modified
# Canonical 14 → scenario-type tags observed in your screenshots
CANONICAL_14: dict[str, set[str]] = {
    "turn_left": {
        "starting_left_turn",
    },
    "turn_right": {
        "starting_right_turn",
    },
    "straight_through_intersection": {
        "traversing_intersection",
        "traversing_traffic_light_intersection",
        "on_intersection",
        "on_all_way_stop_intersection",
        "on_traffic_light_intersection",
        "starting_straight_stop_sign_intersection_traversal",
        "starting_straight_traffic_light_intersection_traversal",
        "on_stopline_stop_sign",
        "accelerating_at_stop_sign",
        "accelerating_at_stop_sign_no_crosswalk",
        "stopping_at_stop_sign_with_lead",
        "stopping_at_stop_sign_without_lead",
        "stopping_at_stop_sign_no_crosswalk",
    },
    "lane_change_left": {
        "changing_lane_to_left",
    },
    "lane_change_right": {
        "changing_lane_to_right",
    },
    "stop_for_red_light": {
        "on_stopline_traffic_light",
        "stationary_at_traffic_light_with_lead",
        "stationary_at_traffic_light_without_lead",
        "stopping_at_traffic_light_with_lead",
        "stopping_at_traffic_light_without_lead",
    },
    "go_on_green": {
        "accelerating_at_traffic_light",
        "accelerating_at_traffic_light_with_lead",
        "accelerating_at_traffic_light_without_lead",
        "starting_straight_traffic_light_intersection_traversal",
    },
    "yield_to_pedestrian": {
        "near_pedestrian_on_crosswalk",
        "near_pedestrian_on_crosswalk_with_ego",
        "stationary_at_crosswalk",
        "stopping_at_crosswalk",
        "traversing_crosswalk",
        "waiting_for_pedestrian_to_cross",
        "on_stopline_crosswalk",
        "accelerating_at_crosswalk",
        "behind_pedestrian_on_driveable",
        "near_pedestrian_at_pickup_dropoff",
        "near_multiple_pedestrians",
    },
    "cut_in": {
        "near_long_vehicle",
        "near_high_speed_vehicle",
        "near_multiple_vehicles",
        "behind_long_vehicle",
    },
    "car_following": {
        "stationary_in_traffic",
        "stationary",
        "following_lane_with_lead",
        "following_lane_with_slow_lead",
        "following_lane_without_lead",
        "stopping_with_lead",
        "behind_bike",
    },
    "pickup_dropoff": {
        "on_pickup_dropoff",
        "traversing_pickup_dropoff",
        "behind_pedestrian_on_pickup_dropoff",
        "near_pedestrian_at_pickup_dropoff",
        "on_carpark",
    },
    "construction_zone": {
        "near_construction_zone_sign",
        "near_trafficcone_on_driveable",
        "near_barrier_on_driveable",
        "traversing_narrow_lane",
    },
    "speed_events": {
        "high_magnitude_speed",
        "medium_magnitude_speed",
        "low_magnitude_speed",
        "high_magnitude_jerk",
        "high_lateral_acceleration",
    },
}




def _index_family_examples(data_root: str) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return {family: [(db_file, lidar_pc_token, raw_type), ...]} using tutorial helpers."""
    log_db_files = discover_log_dbs(data_root)
    type_to_pairs = get_scenario_type_token_map(log_db_files)  # {raw_type: [(db_file, token), ...]}

    family_to_examples: Dict[str, List[Tuple[str, str, str]]] = {fam: [] for fam in CANONICAL_14}
    for raw_type, pairs in type_to_pairs.items():
        for fam, aliases in CANONICAL_14.items():
            if raw_type in aliases:
                family_to_examples[fam].extend((db, tok, raw_type) for (db, tok) in pairs)
    return family_to_examples


def launch_canonical_viewer(
    data_root: str,
    map_root: str,
    map_version: str = "nuplan-maps-v1.0",
    show_only_available: bool = True,
    fixed_port: int = 8898,
) -> None:
    """Create a small UI to browse canonical families and render one episode per click.

    Args:
        data_root: Directory or list of .db files accepted by discover_log_dbs.
        map_root: Maps root.
        map_version: Map version string.
        show_only_available: If True, show only families that exist in these DBs; else show all 14.
        fixed_port: Reuse this port; tutorial_utils shuts down the previous server on the same port.
    """
    # Sanity
    if not data_root or not os.path.isdir(data_root):
        raise AssertionError(f"Invalid data_root: {data_root}")
    if not map_root or not os.path.isdir(map_root):
        raise AssertionError(f"Invalid map_root: {map_root}")

    # Bind bokeh to the notebook once
    setup_notebook()
    output_notebook()

    # Own two dedicated output areas: text + bokeh
    info_out = widgets.Output()
    viz_out = widgets.Output()
    display(info_out, viz_out)

    # Build the example index
    family_to_examples = _index_family_examples(data_root)

    all_families = list(CANONICAL_14.keys())
    family_options = [f for f in all_families if family_to_examples[f]] if show_only_available else all_families

    family_dd = widgets.Dropdown(options=family_options, description="Scenario:")
    mode_dd = widgets.Dropdown(options=["random", "cycle"], value="random", description="Pick:")
    play_btn = widgets.Button(description="Play")

    # cycle cursors per family
    cursors = {f: 0 for f in all_families}

    def on_play(_):
        fam = family_dd.value
        examples = family_to_examples.get(fam, [])

        with info_out:
            clear_output(wait=True)
            if not examples:
                print(f"No examples for '{fam}' in these DBs.")
                return

        # pick an example
        if mode_dd.value == "random":
            db_file, token, raw_type = random.choice(examples)
        else:
            i = cursors[fam] % len(examples)
            db_file, token, raw_type = examples[i]
            cursors[fam] = i + 1

        # Clear visualization area so the previous app disappears
        viz_out.clear_output(wait=True)

        # Ensure bokeh is ready (re-bind after any clears)
        setup_notebook()
        output_notebook()

        # Build scenario and render inside our viz_out
        with viz_out:
            print(f"{fam} → raw='{raw_type}'  |  db='{Path(db_file).name}'")
            scenario = get_default_scenario_from_token(data_root, db_file, token, map_root, map_version)
            visualize_scenario(scenario, bokeh_port=fixed_port)

    play_btn.on_click(on_play)

    ui = widgets.VBox([widgets.HBox([family_dd, mode_dd, play_btn]), info_out])
    display(ui)
