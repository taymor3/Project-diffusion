import math
import torch
import warnings
import numpy as np

from typing import Deque, Dict, List, Type
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config


# nuPlan map spam
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
)

# torch attention mask dtype warning
warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool*",
    category=UserWarning,
)


class DiffusionPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
        ):

        if device not in ("cpu", "cuda"):
            raise ValueError(f"device {device} not supported")
        if device == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError("cuda is not available")

            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._ckpt_path = ckpt_path

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

        # --- selection stats (ticks) ---
        self._sel_total = 0
        self._sel_parent0 = 0
        self._sel_non_parent0 = 0

        # keep global across scenarios (planner instance persists)
        self._sel_total_all = 0
        self._sel_parent0_all = 0
        self._sel_non_parent0_all = 0

        self._sel_print_every = 0   # 0 disables periodic stats printing
        self._debug_verbose = False

        self._margin = 4.0          # parent0 override margin (lower score is better)
        self._model_loaded = False  # avoid re-loading checkpoint every scenario

        # Lightweight batch sanity checks (only once per batch size)
        self._checked_batch_sanity = set()
        self._batch_sanity_keys = ("ego_agent_past", "neighbor_agents_past")

        # ---------------- GPU memory logging (OFF by default) ----------------
        self._mem_log = False          # set True to enable prints
        self._mem_every = 49           # print every N ticks within a scenario
        self._scenario_tick = 0        # reset in initialize()

        # scenario-level baseline + peaks (MB)
        self._mem_start_alloc_mb = 0.0
        self._mem_start_reserved_mb = 0.0
        self._mem_peak_alloc_mb = 0.0
        self._mem_peak_reserved_mb = 0.0

        # Roll injection switch:
        # False -> use pred t=1 (original behavior)
        # True  -> use pred t=0 (the “one-tick” alternative you tested)
        self._roll_use_pred_t0 = False





    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._sel_reset()  # reset per-scenario counters (global *_all counters remain)

        # ---------------- GPU memory logging: per-scenario reset/summary ----------------
        if self._device == "cuda" and getattr(self, "_mem_log", False):
            # Print summary for the previous scenario (initialize is called at next scenario start)
            if getattr(self, "_scenario_tick", 0) > 0:
                print(
                    f"[GPU_MEM_SCENARIO] ticks={self._scenario_tick} "
                    f"start_alloc={self._mem_start_alloc_mb:.1f}MB start_res={self._mem_start_reserved_mb:.1f}MB "
                    f"peak_alloc={self._mem_peak_alloc_mb:.1f}MB peak_res={self._mem_peak_reserved_mb:.1f}MB"
                )

            # Reset counters + CUDA peak stats for the new scenario
            self._scenario_tick = 0
            torch.cuda.reset_peak_memory_stats()
            self._mem_start_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            self._mem_start_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            self._mem_peak_alloc_mb = self._mem_start_alloc_mb
            self._mem_peak_reserved_mb = self._mem_start_reserved_mb


        # Move model once to device (cheap if already there)
        self._planner = self._planner.to(self._device)

        # Load checkpoint only once per planner instance
        if (not self._model_loaded) and (self._ckpt_path is not None):
            state: Dict = torch.load(self._ckpt_path, map_location=self._device)

            # unwrap common checkpoint formats
            if self._ema_enabled and isinstance(state, dict) and ("ema_state_dict" in state):
                state = state["ema_state_dict"]
            elif isinstance(state, dict) and ("model" in state):
                state = state["model"]

            # strip DDP "module." prefix if present; otherwise keep as-is
            if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
                state = {k[len("module."):]: v for k, v in state.items() if k.startswith("module.")}

            self._planner.load_state_dict(state)
            self._model_loaded = True

        elif (not self._model_loaded) and (self._ckpt_path is None):
            if self._debug_verbose:
                print("[DEBUG] ckpt_path is None: using random-initialized weights")
            self._model_loaded = True

        self._planner.eval()


    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)

        model_inputs = self.data_processor.observation_adapter(
            history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device
        )

        return model_inputs


    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)
        return states


    class MidOutput:
        """Holds one parent trajectory and its branch candidates for this tick."""
        def __init__(self, parent_trajectory: AbstractTrajectory, parent_outputs: Dict[str, torch.Tensor]):

            # Parent trajectory and outputs
            self.parent_trajectory = parent_trajectory
            self.parent_outputs = parent_outputs

            # List to hold branch trajectories and their outputs
            self.branch_trajectories: List[AbstractTrajectory] = []
            self.branch_outputs: List[Dict[str, torch.Tensor]] = []


        # Method to add a branch trajectory and its outputs
        def add_branch(self, branch_trajectory: AbstractTrajectory, branch_output: Dict[str, torch.Tensor]):
            self.branch_trajectories.append(branch_trajectory)
            self.branch_outputs.append(branch_output)


    def _sel_reset(self) -> None:
        self._sel_total = 0
        self._sel_parent0 = 0
        self._sel_non_parent0 = 0


    def _sel_log(self, tag: str) -> None:
        print(
            f"[STATS][{tag}] ticks={self._sel_total} parent0={self._sel_parent0} "
            f"non_parent0={self._sel_non_parent0} | "
            f"ALL ticks={self._sel_total_all} parent0={self._sel_parent0_all} "
            f"non_parent0={self._sel_non_parent0_all}"
        )

            
    def score_branch(self, branch_traj: AbstractTrajectory, branch_outputs: Dict[str, torch.Tensor]) -> float:
        """
        Heuristic score for a branch trajectory.
        Lower score is better.

        Components:
        - progress: how far ego moves along the path (more is better)
        - smoothness: sum of absolute heading changes (less is better)
        - collision_penalty: large penalty if predicted ego-neighbor distance gets too small
        """

        # InterpolatedTrajectory already stores the discrete list in _trajectory
        states = getattr(branch_traj, "_trajectory", None)
        if states is None:
            try:
                states = branch_traj.get_sampled_trajectory()
            except AttributeError:
                states = []

        if not states or len(states) < 2:
            if self._debug_verbose:
                print("[DEBUG][score_branch] too few states in branch trajectory")
            return 1e9

        # --- Progress component ---
        start = states[0].rear_axle
        end = states[-1].rear_axle

        dx = float(end.x - start.x)
        dy = float(end.y - start.y)
        progress = math.sqrt(dx * dx + dy * dy)

        # --- Smoothness component (heading changes) ---
        headings = [float(s.rear_axle.heading) for s in states]
        heading_changes = []
        for h0, h1 in zip(headings[:-1], headings[1:]):
            # Wrap difference to [-pi, pi] to avoid jumps
            dh = (h1 - h0 + math.pi) % (2.0 * math.pi) - math.pi
            heading_changes.append(abs(dh))
        smoothness_penalty = sum(heading_changes)

        # --- Collision component (predicted ego vs neighbors) ---
        # Fast path: compute on CPU (xy only). Avoids per-branch GPU kernels + sync.
        collision_penalty = 0.0
        pred = branch_outputs.get("prediction", None)
        try:
            if pred is not None:
                # Accept [B, P, T, 4] (expected) or [P, T, 4]
                pred_xy = pred[0, :, :, :2] if pred.ndim == 4 else (pred[:, :, :2] if pred.ndim == 3 else None)
                if pred_xy is not None and pred_xy.shape[0] > 1:
                    pred_xy_np = pred_xy.detach().cpu().numpy()     # [P, T, 2]
                    ego_xy = pred_xy_np[0]                          # [T, 2]
                    neigh_xy = pred_xy_np[1:]                       # [N, T, 2]

                    diff = neigh_xy - ego_xy[None, :, :]            # [N, T, 2]
                    dist2 = (diff * diff).sum(axis=-1)              # [N, T]
                    min_dist2 = float(dist2.min())

                    if min_dist2 < 1.0:                             # < (1m)^2
                        collision_penalty = 100.0
                    elif min_dist2 < 4.0:                           # < (2m)^2
                        min_dist = math.sqrt(min_dist2)
                        collision_penalty = (2.0 - min_dist) * 20.0
        except Exception as e:
            if getattr(self, "_debug_verbose", False):
                print(f"[DEBUG][score_branch] collision term error: {e}")
                
        # We want:
        # - more progress -> better (lower score)
        # - less heading change -> better (lower score)
        # - fewer close contacts -> better (lower score)
        #
        # Simple linear combination:
        w_progress  = -1.0     # negative: more progress reduces score
        w_smooth    =  0.5     # positive: more heading change increases score
        w_collision =  1.0     # collision_penalty already scaled large

        score = (
            w_progress  * progress +
            w_smooth    * smoothness_penalty +
            w_collision * collision_penalty
        )

        if self._debug_verbose:
            print(
                f"[DEBUG][score_branch] progress={progress:.2f}, "
                f"smooth_pen={smoothness_penalty:.2f}, "
                f"coll_pen={collision_penalty:.2f}, score={score:.2f}"
            )

        return score


    def get_best_trajectory(self, trajectories: List["DiffusionPlanner.MidOutput"]) -> AbstractTrajectory:
        """
        v1.2 selection:
        - Score all branches.
        - Bias toward parent 0: override only if another parent beats parent0 by > margin.
        """
        global_best_score = None
        global_best_parent_traj = None

        parent0_best_score = None

        for parent_idx, mo in enumerate(trajectories):
            for branch_traj, branch_outputs in zip(mo.branch_trajectories, mo.branch_outputs):
                score = self.score_branch(branch_traj, branch_outputs)

                # Track best score for parent 0 separately
                if parent_idx == 0:
                    if parent0_best_score is None or score < parent0_best_score:
                        parent0_best_score = score

                # Track global best over all parents
                if global_best_score is None or score < global_best_score:
                    global_best_score = score
                    global_best_parent_traj = mo.parent_trajectory

        # Fallback: if anything went wrong, default to parent 0
        if global_best_parent_traj is None or parent0_best_score is None:
            return trajectories[0].parent_trajectory

        # Margin in score units (lower is better).
        selected_non_parent0 = (global_best_score + self._margin < parent0_best_score)


        # Update counters (one tick = one planner call)
        self._sel_total += 1
        self._sel_total_all += 1
        if selected_non_parent0:
            self._sel_non_parent0 += 1
            self._sel_non_parent0_all += 1
        else:
            self._sel_parent0 += 1
            self._sel_parent0_all += 1
            
        # periodic logging
        if self._sel_print_every and (self._sel_total % self._sel_print_every == 0):
            self._sel_log("running")

        if selected_non_parent0:
            if self._debug_verbose:
                print("[DEBUG][get_best_trajectory] selecting non-parent0 trajectory")
            return global_best_parent_traj
        else:
            if self._debug_verbose:
                print("[DEBUG][get_best_trajectory] keeping parent0 trajectory")
            return trajectories[0].parent_trajectory

    
    # function to add noise to inputs
    def _add_noise_to_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        std: float = 0.05,
        keys=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Adds small Gaussian noise to selected continuous input tensors.
        Assumes inputs are already normalized (~N(0,1)).
        """
        if keys is None:
            # by default, only perturb dynamic tensors
            keys = ["ego_agent_past", "neighbor_agents_past"]

        if std <= 0.0:
            return inputs  # fast path: no allocations

        noisy = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and (k in keys) and ("mask" not in k) and ("type" not in k):
                noisy[k] = v + torch.randn_like(v) * std
            else:
                noisy[k] = v  # reuse tensor/reference (no clone)
        return noisy

    

    def roll_inputs_with_predictions_multi_agent(
        self,
        original_inputs: Dict[str, torch.Tensor],
        parent_outputs: Dict[str, torch.Tensor],
        num_new_ticks: int = 1,
        k_neighbors: int = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a new model inputs dict where:
        - neighbor_agents_past is rolled forward by 1 tick,
        - the last history step of the first K neighbors is replaced
            by the model's predicted state at t=1 (or t=0 if _roll_use_pred_t0=True),
        - ego_current_state and other keys are copied unchanged.

        This does NOT change the coordinate frame; it just injects the
        predicted next-step states into the history tensor.

        Args:
            original_inputs: dict as passed to the planner
            parent_outputs: dict containing key 'prediction' with shape [1, P, T, 4]
            num_new_ticks: currently only 1 is supported (one-step roll)
            k_neighbors: how many predicted neighbors to use (<= P-1 and <= N).
                        If None, we use all P-1 predicted neighbors (clipped by N).

        Returns:
            new_inputs: dict with rolled 'neighbor_agents_past'.
            (Call observation_normalizer(new_inputs) before reusing.)
        """
        if num_new_ticks != 1:
            raise ValueError(f"Only num_new_ticks=1 is supported, got {num_new_ticks}")

        pred = parent_outputs["prediction"]

        Bp, P, T, _ = pred.shape
        if Bp != 1:
            raise ValueError(f"Only batch size 1 is supported for prediction, got B={Bp}")

        neigh = original_inputs["neighbor_agents_past"]

        Bn, N, H, Dn = neigh.shape
        if Bn != 1:
            raise ValueError(f"Only batch size 1 is supported for neighbor_agents_past, got B={Bn}")

        # How many neighbors do we update from prediction?
        # We assume P = 1 (ego) + K neighbors.
        max_pred_neighbors = max(P - 1, 0)
        if max_pred_neighbors == 0:
            raise ValueError("prediction must have at least 2 participants (ego + neighbor).")

        if k_neighbors is None:
            K = max_pred_neighbors
        else:
            K = min(k_neighbors, max_pred_neighbors)
        K = min(K, N)  # cannot exceed number of neighbor slots

        if K <= 0:
            raise ValueError(f"k_neighbors must resolve to >0, got K={K}")

        # New neighbor history tensor
        new_neigh = neigh.clone()

        # Roll ALL neighbors by 1 tick (vectorized): new[..., 0:H-1] = old[..., 1:H]
        new_neigh[:, :, :-1, :] = neigh[:, :, 1:, :]

        # Default last row: repeat last known state (same as your else-branch)
        new_neigh[:, :, -1, :] = neigh[:, :, -1, :]

        # Overwrite last row dims 0..3 for first K neighbors using configurable pred time index
        # pred: [1, P, T, 4] where participants 1..K correspond to neighbors 0..K-1
        t_idx = 0 if getattr(self, "_roll_use_pred_t0", False) else 1
        if t_idx >= T:
            t_idx = 0  # safety fallback

        pred_t = pred[0, 1:1 + K, t_idx, :4]
        if pred_t.dtype != new_neigh.dtype or pred_t.device != new_neigh.device:
            pred_t = pred_t.to(dtype=new_neigh.dtype, device=new_neigh.device)
        new_neigh[0, :K, -1, :4] = pred_t



        new_inputs = dict(original_inputs)  # shallow copy (no tensor clones)
        new_inputs["neighbor_agents_past"] = new_neigh
        return new_inputs
    

    def _stack_model_inputs(self, inputs_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Concat a list of per-sample input dicts (each with batch=1 tensors) into a single batched dict."""
        if not inputs_list:
            raise ValueError("inputs_list must be non-empty")
        
        batched: Dict[str, torch.Tensor] = {}

        keys = inputs_list[0].keys()
        for k in keys:
            v0 = inputs_list[0][k]
            if isinstance(v0, torch.Tensor):
                batched[k] = torch.cat([d[k] for d in inputs_list], dim=0)  # [B, ...]
            else:
                # Non-tensors (rare in model inputs) must be identical across batch; keep first.
                batched[k] = v0
        return batched


    def _split_model_outputs(self, outputs: Dict[str, torch.Tensor], batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """Split a batched output dict into a list of per-sample output dicts (each with batch=1 tensors)."""
        out_list: List[Dict[str, torch.Tensor]] = []
        for i in range(batch_size):
            oi: Dict[str, torch.Tensor] = {}
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    oi[k] = v[i:i + 1]  # keep batch dimension = 1
                else:
                    oi[k] = v
            out_list.append(oi)
        return out_list


    def _planner_forward_batched(self, inputs_list: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        Run self._planner on a list of per-sample input dicts in one batched forward pass.
        Returns a list of per-sample output dicts (each with batch=1 tensors).
        """
        bs = len(inputs_list)
        if bs <= 0:
            raise ValueError("inputs_list must be non-empty")

        batched_inputs = self._stack_model_inputs(inputs_list)

        # One-time, cheap sanity checks per batch size
        if bs not in self._checked_batch_sanity:
            for k in self._batch_sanity_keys:
                if k in batched_inputs:
                    v = batched_inputs[k]
                    if not isinstance(v, torch.Tensor):
                        raise TypeError(f"{k} must be a Tensor, got {type(v)}")
                    if not (v.ndim >= 1 and v.shape[0] == bs):
                        raise ValueError(f"{k}: expected batch={bs}, got shape={tuple(v.shape)}")
            self._checked_batch_sanity.add(bs)

        with torch.no_grad():
            _, outputs = self._planner(batched_inputs)

        return self._split_model_outputs(outputs, batch_size=bs)




    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        v1.2:
        - Parents are generated in one batched model call.
        - All branches (all parents × branches) are generated in one batched model call.
        - Score leaves and return the selected parent trajectory (with margin gate).
        """
        # RAW inputs from DataProcessor
        inputs_raw = self.planner_input_to_model_inputs(current_input)

        # Normalize for the model (cloning everything here is expensive).
        # Assumption: observation_normalizer does NOT mutate inputs_raw in-place.
        norm_inputs = self.observation_normalizer(inputs_raw)

        # ---------------- GPU memory logging: periodic within-scenario ----------------
        if self._device == "cuda" and getattr(self, "_mem_log", False):
            self._scenario_tick += 1

            if self._mem_every and (self._scenario_tick % self._mem_every == 0):
                cur_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                cur_res = torch.cuda.memory_reserved() / (1024 ** 2)
                peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_res = torch.cuda.max_memory_reserved() / (1024 ** 2)

                if peak_alloc > self._mem_peak_alloc_mb:
                    self._mem_peak_alloc_mb = peak_alloc
                if peak_res > self._mem_peak_reserved_mb:
                    self._mem_peak_reserved_mb = peak_res

                print(
                    f"[GPU_MEM] tick={self._scenario_tick} "
                    f"alloc={cur_alloc:.1f}MB res={cur_res:.1f}MB "
                    f"peak_alloc={peak_alloc:.1f}MB peak_res={peak_res:.1f}MB"
                )


        # Tree shape
        num_of_parents = 4
        num_of_branches = 2

        # Noise levels
        parent_noise_std = 0.02  # parent 0 will be baseline-like (std=0)
        branch_noise_std = 0.02  # same scale for branches

        all_traj: List[DiffusionPlanner.MidOutput] = []

        parent_inputs_list: List[Dict[str, torch.Tensor]] = []
        for parent_idx in range(num_of_parents):
            cur_parent_noise = 0.0 if parent_idx == 0 else parent_noise_std
            parent_inputs_list.append(self._add_noise_to_inputs(norm_inputs, std=cur_parent_noise))

        # One model call for all parents (batch size = num_of_parents)
        parent_outputs_list = self._planner_forward_batched(parent_inputs_list)

        # Build ALL branches for ALL parents first, then run ONE batched branch call (bs = num_of_parents * num_of_branches)
        branch_inputs_all: List[Dict[str, torch.Tensor]] = []
        branch_owner: List[int] = []  # maps each branch sample to its parent_idx (keeps parent-branch pairing correct)

        for parent_idx, parent_outputs in enumerate(parent_outputs_list):
            if self._debug_verbose:
                print("[DEBUG] PARENT requires_grad:",
                      parent_outputs["prediction"].requires_grad,
                      parent_outputs["prediction"].grad_fn)

            parent_traj = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(parent_outputs, current_input.history.ego_states)
            )
            mo = self.MidOutput(parent_traj, parent_outputs)
            all_traj.append(mo)

            rolled_inputs_raw = self.roll_inputs_with_predictions_multi_agent(
                original_inputs=inputs_raw,
                parent_outputs=parent_outputs,
                num_new_ticks=1,
                k_neighbors=5,
            )
            rolled_inputs = self.observation_normalizer(rolled_inputs_raw)

            for _ in range(num_of_branches):
                branch_inputs_all.append(self._add_noise_to_inputs(rolled_inputs, std=branch_noise_std))
                branch_owner.append(parent_idx)

        expected = num_of_parents * num_of_branches
        if len(branch_inputs_all) != expected or len(branch_owner) != expected:
            raise RuntimeError(f"Branch batching mismatch: got {len(branch_inputs_all)} inputs, expected {expected}")

        # Second model call: ALL branches at once (batch size = expected, e.g. 8)
        branch_outputs_all = self._planner_forward_batched(branch_inputs_all)

        # Assign each branch output back to the correct parent
        for branch_outputs, parent_idx in zip(branch_outputs_all, branch_owner):
            branch_traj = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(branch_outputs, current_input.history.ego_states)
            )
            all_traj[parent_idx].add_branch(branch_traj, branch_outputs)

            if self._debug_verbose:
                print("[DEBUG] BRANCH requires_grad:",
                      branch_outputs["prediction"].requires_grad,
                      branch_outputs["prediction"].grad_fn)

        # Choose parent based on best leaf (branch)
        return self.get_best_trajectory(all_traj)

