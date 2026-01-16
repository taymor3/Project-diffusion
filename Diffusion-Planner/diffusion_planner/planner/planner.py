
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type
import math

warnings.filterwarnings("ignore")

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

def identity(ego_state, predictions):
    return predictions


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

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

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

        if self._ckpt_path is not None:
            state_dict:Dict = torch.load(self._ckpt_path, map_location=self._device)
            
            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

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


    # ###################### change here ######################
    # Define a class to hold mid-level outputs for tree search
    class MidOutput:
        def __init__(self, parent_trajectory : AbstractTrajectory, parent_outputs: Dict[str, torch.Tensor]):
            # Parent trajectory and outputs
            self.parent_trajectory = parent_trajectory
            self.parent_outputs = parent_outputs

            # List to hold branch trajectories and their outputs
            self.branch_trajectories = []
            self.branch_outputs = []

        # Method to add a branch trajectory and its outputs
        def add_branch(self, branch_trajectory: AbstractTrajectory, branch_output: Dict[str, torch.Tensor]):
            self.branch_trajectories.append(branch_trajectory)
            self.branch_outputs.append(branch_output)


    def debug_print_model_inputs(self,model_inputs: dict, ego_state_history) -> None:
        print("\n========== [PlannerInput Debug] ==========")

        # ----- Ego history from nuPlan buffer -----
        history_list = list(ego_state_history)
        print(f"# ego_states in history buffer: {len(history_list)}")
        if history_list:
            last = history_list[-1]
            print(
                "last ego state (world): "
                f"x={last.rear_axle.x:.3f}, "
                f"y={last.rear_axle.y:.3f}, "
                f"yaw={float(last.rear_axle.heading):.3f}, "
                f"speed={float(last.dynamic_car_state.speed):.3f}"
            )

        print("\n=== [Model Inputs Dict] ===")
        for name, v in model_inputs.items():
            if not isinstance(v, torch.Tensor):
                print(f"{name}: non-tensor (type={type(v)})")
                continue

            shape = tuple(v.shape)
            print(f"{name}: shape={shape}, dtype={v.dtype}, device={v.device}")

            # Generic stats for float tensors
            if v.dtype.is_floating_point:
                flat = v.detach().cpu().view(-1)
                mean = flat.mean().item()
                std = flat.std().item()
                vmin = flat.min().item()
                vmax = flat.max().item()
                print(f"  stats: mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}")

            # Special cases we care about

            if name == "ego_current_state":
                # ego_current_state is [1, D] or [B, D]
                ecs = v[0].detach().cpu()
                print("  ego_current_state[0]:")
                for i, val in enumerate(ecs):
                    print(f"    dim {i}: {val:.4f}")
                # heuristic: show norm of [cos, sin] if dims 2,3 exist
                if ecs.shape[0] >= 4:
                    cs_norm = (ecs[2]**2 + ecs[3]**2).sqrt().item()
                    print(f"    -> orientation norm (dim2,3): {cs_norm:.4f}")

            if name == "neighbor_agents_past" and v.ndim == 4:
                B, N, H, D = v.shape
                print(f"  neighbor_agents_past: B={B}, N={N}, H={H}, D={D}")
                # Show one example agent over time
                sample = v[0, 0]  # [H, D]
                print("  sample neighbor 0 history (first 3 timesteps, first 8 dims):")
                for t in range(min(3, H)):
                    row = sample[t, :8].detach().cpu().tolist()
                    print(f"    t={t}: {['%.4f' % x for x in row]}")

                # Per-feature stats like you already printed
                print("  --- per-feature stats over all (B,N,H) ---")
                for d in range(D):
                    vals = v[..., d].detach().cpu().view(-1)
                    mean = vals.mean().item()
                    std = vals.std().item()
                    vmin = vals.min().item()
                    vmax = vals.max().item()
                    print(f"    dim {d}: mean={mean:.4f}, std={std:.4f}, "
                        f"min={vmin:.4f}, max={vmax:.4f}")
            
    def score_branch(self, branch_traj: AbstractTrajectory, branch_outputs: Dict[str, torch.Tensor]) -> float:
        """
        Heuristic score for a branch trajectory.
        Lower score is better.

        Components:
        - progress: how far ego moves along the path (more is better)
        - smoothness: sum of absolute heading changes (less is better)
        - collision_penalty: large penalty if predicted ego-neighbor distance gets too small
        """

        # Get list of states along the trajectory
        try:
            states = branch_traj.get_sampled_trajectory()
        except AttributeError:
            states = getattr(branch_traj, "_trajectory", [])

        if not states or len(states) < 2:
            # Very bad score if we don't have a meaningful trajectory
            print("[DEBUG][score_branch] too few states in branch trajectory")  # TODO-s: remove
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
        collision_penalty = 0.0
        pred = branch_outputs.get("prediction", None)
        try:
            if pred is not None:
                # pred: [B, A, T, 4], we assume B=1 here
                pred_np = pred.detach().cpu().numpy()
                if pred_np.shape[0] > 0 and pred_np.shape[1] > 1:
                    # ego positions and neighbor positions in the model's frame
                    ego_xy = pred_np[0, 0, :, :2]              # [T, 2]
                    neighbors_xy = pred_np[0, 1:, :, :2]       # [N, T, 2]

                    if neighbors_xy.size > 0:
                        # Compute distances: [N, T]
                        diff = neighbors_xy - ego_xy[None, :, :]   # broadcast over neighbors
                        dist = np.linalg.norm(diff, axis=-1)
                        min_dist = float(dist.min())

                        # Simple piecewise penalty
                        if min_dist < 1.0:
                            collision_penalty = 100.0
                        elif min_dist < 2.0:
                            collision_penalty = (2.0 - min_dist) * 20.0
                        else:
                            collision_penalty = 0.0
        except Exception as e:
            print(f"[DEBUG][score_branch] collision term error: {e}")  # TODO-s: remove

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

        # Debug print
        print(
            f"[DEBUG][score_branch] progress={progress:.2f}, "
            f"smooth_pen={smoothness_penalty:.2f}, "
            f"coll_pen={collision_penalty:.2f}, score={score:.2f}"
        )  # TODO-s: remove

        return score


    def get_best_trajectory(self, trajectories: List["DiffusionPlanner.MidOutput"]) -> AbstractTrajectory:
        """
        v0.7 selection:
        - Score all branches (as in v0.6).
        - Strong bias toward parent 0 (the v0-like parent):
          only switch to another parent if its best branch score
          is better than parent 0's best branch by more than a margin.
        """
        global_best_score = None
        global_best_parent_traj = None

        parent0_best_score = None

        for parent_idx, mo in enumerate(trajectories):
            for branch_idx, (branch_traj, branch_outputs) in enumerate(
                zip(mo.branch_trajectories, mo.branch_outputs)
            ):
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
            print("[DEBUG][get_best_trajectory] fallback to parent 0")
            return trajectories[0].parent_trajectory

        # Margin in score units (lower is better). With scores typically in [-55, -82],
        # a margin of 4 means we only override parent 0 when another parent is
        # clearly better (by > 4 points).
        MARGIN = 4.0

        print(
            f"[DEBUG][get_best_trajectory] parent0_best={parent0_best_score:.2f}, "
            f"global_best={global_best_score:.2f}, margin={MARGIN:.2f}"
        )

        # Only switch away from parent 0 if the global best is significantly better.
        # Remember: lower score is better.
        if global_best_score + MARGIN < parent0_best_score:
            print("[DEBUG][get_best_trajectory] selecting non-parent0 trajectory")
            return global_best_parent_traj
        else:
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

        noisy = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and k in keys:
                # don't touch masks / one-hots
                if "mask" in k or "type" in k:
                    noisy[k] = v.clone()
                else:
                    noisy[k] = v + torch.randn_like(v) * std
            else:
                noisy[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return noisy


    def debug_print_planner_outputs(self, parent_outputs: Dict[str, torch.Tensor], ego_state_history=None) -> None:
        print("\n========== [Planner Outputs Debug] ==========")

        # ----- raw keys -----
        print("Keys in parent_outputs:", list(parent_outputs.keys()))

        for name, v in parent_outputs.items():
            print(f"\n--- [{name}] ---")
            if isinstance(v, torch.Tensor):
                shape = tuple(v.shape)
                print(f"  tensor: shape={shape}, dtype={v.dtype}, device={v.device}")

                if v.dtype.is_floating_point:
                    flat = v.detach().cpu().view(-1)
                    mean = flat.mean().item()
                    std = flat.std().item()
                    vmin = flat.min().item()
                    vmax = flat.max().item()
                    print(f"  stats: mean={mean:.4f}, std={std:.4f}, min={vmin:.4f}, max={vmax:.4f}")
            else:
                print(f"  non-tensor, type={type(v)}")

        # ----- special handling for 'prediction' -----
        if "prediction" in parent_outputs and isinstance(parent_outputs["prediction"], torch.Tensor):
            pred = parent_outputs["prediction"]
            print("\n=== [prediction tensor details] ===")

            if pred.ndim == 4:
                B, P, T, D = pred.shape
                print(f"prediction shape: B={B}, P={P}, T={T}, D={D}")

                ego_traj = pred[0, 0].detach().cpu()  # [T, D] (assuming ego is index 0)
                max_time = min(5, T)
                max_dims_to_show = min(8, D)

                print(f"ego trajectory (first {max_time} steps, first {max_dims_to_show} dims):")
                for t in range(max_time):
                    row = ego_traj[t, :max_dims_to_show].tolist()
                    pieces = [f"d{j}={row[j]:.3f}" for j in range(max_dims_to_show)]
                    print(f"  t={t}: " + ", ".join(pieces))

                # rough step distance only if we have at least x,y
                if T >= 2 and D >= 2:
                    dx = float(ego_traj[1, 0] - ego_traj[0, 0])
                    dy = float(ego_traj[1, 1] - ego_traj[0, 1])
                    step_dist = math.sqrt(dx * dx + dy * dy)
                    print(f"approx step distance between t=0 and t=1: {step_dist:.3f}")
            else:
                print(f"prediction has unexpected ndim={pred.ndim}, shape={tuple(pred.shape)}")

        # ----- optionally: decode to EgoState list and compare to history -----
        if ego_state_history is not None and "prediction" in parent_outputs:
            try:
                ego_future_states = self.outputs_to_trajectory(parent_outputs, ego_state_history)
                print("\n=== [decoded ego_future_states from outputs_to_trajectory] ===")
                print(f"# future states: {len(ego_future_states)}")

                history_list = list(ego_state_history)
                if history_list:
                    last = history_list[-1]
                    print(
                        "last real ego state (world): "
                        f"x={last.rear_axle.x:.3f}, "
                        f"y={last.rear_axle.y:.3f}, "
                        f"yaw={float(last.rear_axle.heading):.3f}, "
                        f"speed={float(last.dynamic_car_state.speed):.3f}"
                    )

                print("first min(5, len) decoded future ego states:")
                for i, s in enumerate(ego_future_states[:5]):
                    print(
                        f"  t={i}: "
                        f"x={s.rear_axle.x:.3f}, "
                        f"y={s.rear_axle.y:.3f}, "
                        f"yaw={float(s.rear_axle.heading):.3f}, "
                        f"speed={float(s.dynamic_car_state.speed):.3f}"
                    )
            except Exception as e:
                print(f"[debug_print_planner_outputs] outputs_to_trajectory failed: {e}")


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
            by the model's predicted state at t=1,
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

        pred = parent_outputs["prediction"]

        Bp, P, T, Dout = pred.shape

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

        # For each neighbor slot
        for n_idx in range(N):
            hist = neigh[0, n_idx]  # [H, Dn]
            new_hist = hist.clone()

            # Roll history by 1: drop earliest, shift left
            # old: t=0..H-1 -> new: t=0..H-2 are old t=1..H-1
            new_hist[:-1, :] = hist[1:, :]

            if n_idx < K:
                # This neighbor has a predicted trajectory in 'prediction'
                # participant index in prediction: 1 + n_idx (0 is ego)
                pred_traj = pred[0, 1 + n_idx]  # [T, Dout]
                pred_t1 = pred_traj[1]          # state at t=1, [Dout]

                # Start from last old row to preserve extra dims (size, type, etc.)
                row = new_hist[-1].clone()

                # Overwrite the first 4 dims: x, y, cos, sin
                row[0] = pred_t1[0]
                row[1] = pred_t1[1]
                row[2] = pred_t1[2]
                row[3] = pred_t1[3]
                # We could try to infer speed and other dims here if needed,
                # but for now we leave them unchanged.

                new_hist[-1] = row
                if n_idx == 0:
                    # sanity check for first neighbor: last row vs pred_t1
                    diff = (new_hist[-1, :4] - pred_t1).abs()
                    print("[sanity] neighbor 0 last row vs pred_t1 diff:", diff.tolist())
            else:
                # For neighbors without explicit prediction, we simply repeat
                # their last known state in the final slot.
                new_hist[-1, :] = hist[-1, :]

            new_neigh[0, n_idx] = new_hist

        # Build the new inputs dict
        new_inputs: Dict[str, torch.Tensor] = {}
        for k, v in original_inputs.items():
            if k == "neighbor_agents_past":
                new_inputs[k] = new_neigh
            else:
                new_inputs[k] = v.clone() if isinstance(v, torch.Tensor) else v

        return new_inputs
    

    def debug_check_roll_multi_agent(
        self,
        original_inputs: Dict[str, torch.Tensor],
        rolled_inputs: Dict[str, torch.Tensor],
        parent_outputs: Dict[str, torch.Tensor],
        k_neighbors: int,
    ) -> None:
        print("\n========== [Roll Debug] ==========")

        neigh_old = original_inputs["neighbor_agents_past"]    # [1, N, H, Dn]
        neigh_new = rolled_inputs["neighbor_agents_past"]      # [1, N, H, Dn]
        pred = parent_outputs["prediction"]                    # [1, P, T, 4]

        B, N, H, Dn = neigh_old.shape
        _, P, T, Dout = pred.shape
        print(f"old neigh shape: {neigh_old.shape}")
        print(f"new neigh shape: {neigh_new.shape}")
        print(f"prediction shape: {pred.shape}")

        # Check first neighbor (index 0) which should correspond to participant 1 in prediction
        n_idx = 0
        if n_idx >= N:
            print("No neighbor 0 to inspect, N=0?")
            return

        old_hist = neigh_old[0, n_idx]   # [H, Dn]
        new_hist = neigh_new[0, n_idx]   # [H, Dn]

        # predicted neighbor traj (participant 1, because 0 is ego)
        pred_traj = pred[0, 1 + n_idx]   # [T, 4]
        pred_t0 = pred_traj[0]
        pred_t1 = pred_traj[1]

        print("\n--- neighbor 0, first 3 timesteps (dims 0..3) ---")
        print("old:")
        for t in range(min(3, H)):
            print(f"  t={t}: {old_hist[t, :4].tolist()}")
        print("new:")
        for t in range(min(3, H)):
            print(f"  t={t}: {new_hist[t, :4].tolist()}")

        print("\n--- neighbor 0, last 3 timesteps (dims 0..3) ---")
        print("old:")
        for t in range(max(0, H - 3), H):
            print(f"  t={t}: {old_hist[t, :4].tolist()}")
        print("new:")
        for t in range(max(0, H - 3), H):
            print(f"  t={t}: {new_hist[t, :4].tolist()}")

        print("\n--- prediction for neighbor 0 (participant 1), t=0,1 (dims 0..3) ---")
        print("  pred t=0:", pred_t0[:4].tolist())
        print("  pred t=1:", pred_t1[:4].tolist())
        
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        v0.5:
        - Multiple parents (roots): different noisy samples from the same observation.
        - For each parent: build branches by rolling inputs + noise.
        - Score leaves and select the parent of the best leaf.
        """
        # RAW inputs from DataProcessor
        inputs_raw = self.planner_input_to_model_inputs(current_input)

        # Normalized inputs for the parent pass (original behavior)
        norm_inputs = self.observation_normalizer(inputs_raw)

        # Tree shape
        num_of_parents = 3
        num_of_branches = 2

        # Noise levels
        parent_noise_std = 0.02  # small; parent 0 will effectively be near-original
        branch_noise_std = 0.02  # same scale for branches

        all_traj: List[DiffusionPlanner.MidOutput] = []

        for parent_idx in range(num_of_parents):
            # For parent 0 you can optionally reduce noise for a "baseline-like" parent
            if parent_idx == 0:
                cur_parent_noise = 0.0
            else:
                cur_parent_noise = parent_noise_std

            # Parent prediction: same observation, different noise per parent
            noisy_parent_inputs = self._add_noise_to_inputs(norm_inputs, std=cur_parent_noise)
            _, parent_outputs = self._planner(noisy_parent_inputs)

            parent_traj = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(parent_outputs, current_input.history.ego_states)
            )
            mo = self.MidOutput(parent_traj, parent_outputs)
            all_traj.append(mo)

            # Build rolled RAW inputs using this parent's outputs
            rolled_inputs_raw = self.roll_inputs_with_predictions_multi_agent(
                original_inputs=inputs_raw,
                parent_outputs=parent_outputs,
                num_new_ticks=1,
                k_neighbors=5,
            )
            # Normalize rolled inputs before branch passes
            rolled_inputs = self.observation_normalizer(rolled_inputs_raw)

            # Branch predictions for this parent
            for branch_idx in range(num_of_branches):
                noisy_branch_inputs = self._add_noise_to_inputs(rolled_inputs, std=branch_noise_std)
                _, branch_outputs = self._planner(noisy_branch_inputs)

                branch_traj = InterpolatedTrajectory(
                    trajectory=self.outputs_to_trajectory(branch_outputs, current_input.history.ego_states)
                )
                mo.add_branch(branch_traj, branch_outputs)

        # Choose parent based on best leaf (branch)
        return self.get_best_trajectory(all_traj)
