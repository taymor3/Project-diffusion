
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
            self.parent_trajectory = parent_trajectory
            self.parent_outputs = parent_outputs
            self.branch_trajectories = []
            self.branch_outputs = []
        def add_branch(self, branch_trajectory: AbstractTrajectory, branch_output: Dict[str, torch.Tensor]):
            self.branch_trajectories.append(branch_trajectory)
            self.branch_outputs.append(branch_output)


    def roll_ego_history_with_predictions(
        self,
        original_inputs: Dict[str, torch.Tensor],
        parent_outputs: Dict[str, torch.Tensor],
        ego_state_history: Deque[EgoState],
        num_new_ticks: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a new model inputs dict whose ego history is rolled forward by `num_new_ticks`.

        - If `ego_agent_past` does NOT exist in `original_inputs`, we create it once from
          `ego_state_history`, using the same feature dimension as `neighbor_agents_past`.
        - Then we keep the first (H - num_new_ticks) steps (e.g. 15 if H=20)
          and append `num_new_ticks` new ego states coming from `parent_outputs`
          via `outputs_to_trajectory`.

        Returns:
            new_inputs: dict like original_inputs, but with updated 'ego_agent_past'.

        NOTE:
        - This returns UNNORMALIZED inputs. Call `self.observation_normalizer(new_inputs)`
          before passing into `self._planner` again.
        """

        if "prediction" not in parent_outputs:
            raise KeyError("parent_outputs must contain key 'prediction' from the planner.")

        # ------------------------------------------------------------------
        # 1) Ensure we have an ego_agent_past tensor [1, H, D]
        # ------------------------------------------------------------------
        if "ego_agent_past" in original_inputs:
            ego_past = original_inputs["ego_agent_past"]  # [1, H, D]
            if ego_past.ndim != 3:
                raise ValueError(f"ego_agent_past must be [1, H, D], got {ego_past.shape}")
            B, H, D = ego_past.shape
        else:
            # Synthesize ego_agent_past from ego_state_history
            if "neighbor_agents_past" not in original_inputs:
                raise KeyError(
                    "neighbor_agents_past not found in inputs; cannot infer feature dim for ego_agent_past."
                )

            neigh = original_inputs["neighbor_agents_past"]  # [1, N_max, Hn, D]
            if neigh.ndim != 4:
                raise ValueError(f"neighbor_agents_past must be [1, N, Hn, D], got {neigh.shape}")

            _, _, _, D = neigh.shape
            # we want H=20 last ticks if possible, but cap by history length
            history_list = list(ego_state_history)
            H = min(20, len(history_list))
            if H < num_new_ticks:
                raise ValueError(
                    f"Not enough ego history to build H={H} with num_new_ticks={num_new_ticks}"
                )

            history_slice = history_list[-H:]  # last H ego states
            # create on same device/dtype as neighbor_agents_past
            ego_past = neigh.new_zeros((1, H, D))  # [1, H, D]

            # reference = last real ego state (current)
            ref_state = history_slice[-1]
            ref_x = ref_state.rear_axle.x
            ref_y = ref_state.rear_axle.y

            for i, s in enumerate(history_slice):
                dx = s.rear_axle.x - ref_x
                dy = s.rear_axle.y - ref_y
                yaw = float(s.rear_axle.heading)
                speed = float(s.dynamic_car_state.speed)

                row = ego_past.new_zeros((D,))
                # heuristic feature layout: match neighbor_agents_past as much as we can
                if D >= 1:
                    row[0] = dx
                if D >= 2:
                    row[1] = dy
                if D >= 3:
                    row[2] = math.cos(yaw)
                if D >= 4:
                    row[3] = math.sin(yaw)
                if D >= 5:
                    row[4] = speed
                # type one-hot (like neighbor dim 8 == "car") if exists
                if D > 8:
                    row[8] = 1.0

                ego_past[0, i] = row

            B = 1  # by construction

        if B != 1:
            raise ValueError(f"Only batch size 1 is supported here, got B={B}")
        if H <= num_new_ticks:
            raise ValueError(
                f"History length H={H} must be > num_new_ticks={num_new_ticks}"
            )

        # ------------------------------------------------------------------
        # 2) Convert planner outputs to world-frame ego future states
        #    Using your existing outputs_to_trajectory logic
        # ------------------------------------------------------------------
        ego_future_states = self.outputs_to_trajectory(parent_outputs, ego_state_history)
        if len(ego_future_states) <= num_new_ticks:
            raise ValueError(
                f"Not enough future states: got {len(ego_future_states)}, "
                f"need at least {num_new_ticks + 1} (including t=0)."
            )

        # ego_future_states[0] is t=0 (aligned with last history state)
        # we want NEW ticks: t=1..num_new_ticks
        new_ego_segment = ego_future_states[1 : 1 + num_new_ticks]  # list[EgoState], len=num_new_ticks

        # ------------------------------------------------------------------
        # 3) Split old ego history: keep first H - num_new_ticks, roll in new 5
        # ------------------------------------------------------------------
        keep_len = H - num_new_ticks          # e.g. 20 - 5 = 15
        old_segment = ego_past[:, :keep_len]  # [1, keep_len, D]

        # build [num_new_ticks, D] from new_ego_segment
        new_segment = ego_past.new_zeros((num_new_ticks, D))  # [5, D]

        # reference state for relative position = last REAL history state
        ref_state = ego_future_states[0]
        ref_x = ref_state.rear_axle.x
        ref_y = ref_state.rear_axle.y

        # use last row of ego_past as template for extra dims
        last_old_row = ego_past[0, -1, :]  # [D]

        for i, s in enumerate(new_ego_segment):
            dx = s.rear_axle.x - ref_x
            dy = s.rear_axle.y - ref_y
            yaw = float(s.rear_axle.heading)
            speed = float(s.dynamic_car_state.speed)

            row = last_old_row.clone()
            if D >= 1:
                row[0] = dx
            if D >= 2:
                row[1] = dy
            if D >= 3:
                row[2] = math.cos(yaw)
            if D >= 4:
                row[3] = math.sin(yaw)
            if D >= 5:
                row[4] = speed
            new_segment[i] = row

        # [1, keep_len, D] + [1, num_new_ticks, D] → [1, H, D]
        new_ego_past = torch.cat([old_segment, new_segment.unsqueeze(0)], dim=1)
        assert new_ego_past.shape == (1, H, D), f"new_ego_past has wrong shape {new_ego_past.shape}"

        # ------------------------------------------------------------------
        # 4) Build new inputs dict with updated ego_agent_past
        # ------------------------------------------------------------------
        new_inputs: Dict[str, torch.Tensor] = {}
        for k, v in original_inputs.items():
            if k == "ego_agent_past":
                # overwrite if existed
                new_inputs[k] = new_ego_past
            else:
                new_inputs[k] = v.clone() if isinstance(v, torch.Tensor) else v

        # if it didn't exist before, add it now
        if "ego_agent_past" not in new_inputs:
            new_inputs["ego_agent_past"] = new_ego_past

        return new_inputs



    def get_best_trajectory(self, trajectories: List[MidOutput]) -> AbstractTrajectory:
        """
        Select the best trajectory based on a simple heuristic (e.g., minimum distance to a reference path).
        This is a placeholder for more sophisticated selection logic.
        """
        # Placeholder: return the first trajectory

        return trajectories[0].parent_trajectory

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)

        norm_inputs = self.observation_normalizer(inputs)
        num_of_parents = 2
        num_of_branches = 2
        all_traj = []

        for i in range(num_of_parents):
            _, outputs = self._planner(inputs)

            trajectory = InterpolatedTrajectory(
                    trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
                )
            mo = self.MidOutput(trajectory, outputs)
            all_traj.append(mo)
            
            rolled_inputs = self.roll_ego_history_with_predictions(
                        original_inputs=inputs,
                        parent_outputs=outputs,
                        ego_state_history=current_input.history.ego_states,
                        num_new_ticks=5,
                    )
            rolled_inputs = self.observation_normalizer(rolled_inputs)
            for j in range(num_of_branches):
                _, branch_outputs = self._planner(rolled_inputs)

                branch_trajectory = InterpolatedTrajectory(
                        trajectory=self.outputs_to_trajectory(branch_outputs, current_input.history.ego_states)
                    )
                mo.add_branch(branch_trajectory, branch_outputs)


        return self.get_best_trajectory(all_traj)  
