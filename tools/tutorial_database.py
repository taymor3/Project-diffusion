from bokeh.io import output_notebook
from tutorials.utils.tutorial_utils import visualize_nuplan_scenarios, setup_notebook
import os, glob
def setup_notebook_env():
    output_notebook()
    setup_notebook()

def visualize_database(bokeh_port = 8899 , data_root = None, map_root = None, map_version = None):
    setup_notebook_env()
    # Paths
    NUPLAN_DATA_ROOT = data_root   # folder that contains many .db files
    NUPLAN_MAPS_ROOT = map_root                     # folder that contains nuplan-maps-v1.0.json
    NUPLAN_MAP_VERSION = map_version

    assert os.path.isdir(NUPLAN_DATA_ROOT), f"DATA dir missing: {NUPLAN_DATA_ROOT}"
    assert os.path.isfile(os.path.join(NUPLAN_MAPS_ROOT, "nuplan-maps-v1.0.json")), "maps json not found"

    # Build db_files properly: list of .db files (preferred)
    db_files = sorted(glob.glob(os.path.join(NUPLAN_DATA_ROOT, "*.db")))
    if not db_files:
        # fallback: pass the directory itself (must exist)
        db_files = [NUPLAN_DATA_ROOT]

    print(f"Found {len(db_files)} .db files")
    print("Sample:", db_files[:3])

    visualize_nuplan_scenarios(
        data_root=NUPLAN_DATA_ROOT,
        db_files=db_files,                 # <-- list of files or a valid dir
        map_root=NUPLAN_MAPS_ROOT,
        map_version=NUPLAN_MAP_VERSION,
        bokeh_port=bokeh_port
    )

