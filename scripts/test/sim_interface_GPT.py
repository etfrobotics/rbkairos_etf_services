#!/usr/bin/env python3
"""sim_interface.py (rewritten)

This module now builds a *reward-consistent* "environment interface" around
`FruitHarvestingRewardEvaluator` from evaluator.py.

Key goals:
- Keep the existing API expectation from the old ServiceCaller:
    - step(action_name, action_params, t)
    - get_obs() -> List[prost_bridge/KeyValue]
    - attributes: terminate, last_reward
- Drive reward computation via evaluator.py (instead of re-implementing reward).
- Provide a deterministic mock transition model using evaluator.simulate_step.

For real-robot integration, this sim is still useful as a mock world model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rospy

from prost_bridge.msg import KeyValue

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None

try:
    from pyRDDLGym.core.env import RDDLEnv
except Exception:  # pragma: no cover
    RDDLEnv = None

from evaluator import FruitHarvestingRewardEvaluator


TERMINATION_STRING = "ROUND_END"


def _get_objects_by_type(env_model: Any, type_name: str) -> List[str]:
    """Best-effort to retrieve a stable, index-consistent ordering of objects."""

    # Preferred: pyRDDLGym keeps an ordered list per type.
    if hasattr(env_model, "_objects") and isinstance(env_model._objects, dict) and type_name in env_model._objects:
        return list(env_model._objects[type_name])

    # Fallback: use object_to_index ordering (then name).
    obj_to_type = getattr(env_model, "_object_to_type", {}) or {}
    obj_to_idx = getattr(env_model, "_object_to_index", {}) or {}
    objs = [o for o, t in obj_to_type.items() if t == type_name]
    objs.sort(key=lambda o: (obj_to_idx.get(o, 10**9), o))
    return objs


def _bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "t", "yes", "y")
    return False


@dataclass
class ActionFluents:
    grasp_fruit: "np.ndarray"
    load_to_bin: "np.ndarray"
    navigate: "np.ndarray"
    unload: bool


class FruitHarvestingSim:
    """Deterministic mock environment that uses evaluator.py for reward.

    Transition dynamics are a *mock* (deterministic) version implemented in
    FruitHarvestingRewardEvaluator.simulate_step.

    Reward is computed by FruitHarvestingRewardEvaluator.evaluate_reward.
    """

    def __init__(self, domain_file: str, instance_file: str):
        if np is None:
            raise RuntimeError("numpy is required")
        if RDDLEnv is None:
            raise RuntimeError("pyRDDLGym is required (RDDLEnv import failed)")

        self.env = RDDLEnv(domain=domain_file, instance=instance_file)
        self.model = self.env.model
        self.evaluator = FruitHarvestingRewardEvaluator(self.model)

        # Stable object order for converting to KeyValue[]
        self.positions: List[str] = _get_objects_by_type(self.model, "aisle_position")
        self.locations: List[str] = _get_objects_by_type(self.model, "location")
        self.pos_to_i = {p: i for i, p in enumerate(self.positions)}
        self.loc_to_i = {l: i for i, l in enumerate(self.locations)}

        self.state = self._state_from_model()

        self.terminate: bool = False
        self.last_reward: float = 0.0

    # ---------------------------------------------------------------------
    # Compatibility with old SimInterface
    # ---------------------------------------------------------------------

    def get_obs(self) -> List[KeyValue]:
        return self._state_to_keyvalues(self.state)

    def step(self, action_name: str, action_params: Sequence[str], t: int) -> float:
        if self.terminate:
            return self.last_reward

        if action_name == TERMINATION_STRING:
            self.terminate = True
            self.last_reward = 0.0
            return self.last_reward

        action_fluents = self._planner_action_to_action_fluents(action_name, action_params)
        action_dict = self.evaluator.create_observation_template()
        action_dict["grasp_fruit"] = action_fluents.grasp_fruit
        action_dict["load_to_bin"] = action_fluents.load_to_bin
        action_dict["navigate"] = action_fluents.navigate
        action_dict["unload"] = bool(action_fluents.unload)

        # next state via mock deterministic dynamics
        next_state = self.evaluator.simulate_step(self.state, action_dict)

        # reward uses *current state + action fluents* and *next state*
        obs_with_action = self._merge_state_and_action(self.state, action_dict)
        reward = float(self.evaluator.evaluate_reward(obs_with_action, next_state))

        self.state = next_state
        self.last_reward = reward

        self.terminate = self._is_terminal(next_state)
        return reward

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------

    def _state_from_model(self) -> Dict[str, Any]:
        obs = self.evaluator.create_observation_template()

        # Load initial state from pyRDDLGym model, following evaluator.py __main__
        obs["fruit_at"] = np.array(self.model._state_fluents["fruit_at"], dtype=bool)
        obs["fruit_collected"] = np.array(self.model._state_fluents["fruit_collected"], dtype=bool)
        obs["fruit_in_bin"] = np.array(self.model._state_fluents["fruit_in_bin"], dtype=bool)
        obs["fruits_unloaded"] = np.array(self.model._state_fluents["fruits_unloaded"], dtype=bool)
        obs["robot_at"] = np.array(self.model._state_fluents["robot_at"], dtype=bool)
        obs["position_visited"] = np.array(self.model._state_fluents["position_visited"], dtype=bool)

        # Ensure action fluents are clear
        obs["grasp_fruit"][:] = False
        obs["load_to_bin"][:] = False
        obs["navigate"][:] = False
        obs["unload"] = False

        return obs

    def _merge_state_and_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        merged = self.evaluator.create_observation_template()
        for k in ("fruit_at", "fruit_collected", "fruit_in_bin", "fruits_unloaded", "robot_at", "position_visited"):
            merged[k] = state[k].copy()
        for k in ("grasp_fruit", "load_to_bin", "navigate"):
            merged[k] = action[k].copy()
        merged["unload"] = bool(action["unload"])
        return merged

    def _planner_action_to_action_fluents(self, action_name: str, action_params: Sequence[str]) -> ActionFluents:
        grasp = np.zeros(self.evaluator.num_locations, dtype=bool)
        load = np.zeros(self.evaluator.num_locations, dtype=bool)
        nav = np.zeros(self.evaluator.num_positions, dtype=bool)
        unload = False

        a = (action_name or "").strip()
        params = list(action_params) if action_params is not None else []

        if a == "navigate" and params:
            tgt = params[0]
            if tgt in self.pos_to_i:
                nav[self.pos_to_i[tgt]] = True
            else:
                rospy.logwarn(f"[FruitHarvestingSim] Unknown navigate target: {tgt}")

        elif a == "grasp_fruit" and params:
            loc = params[0]
            if loc in self.loc_to_i:
                grasp[self.loc_to_i[loc]] = True
            else:
                rospy.logwarn(f"[FruitHarvestingSim] Unknown grasp target: {loc}")

        elif a == "load_to_bin" and params:
            loc = params[0]
            if loc in self.loc_to_i:
                load[self.loc_to_i[loc]] = True
            else:
                rospy.logwarn(f"[FruitHarvestingSim] Unknown load target: {loc}")

        elif a == "unload":
            unload = True

        elif a == "" or a.lower() == "noop":
            pass

        else:
            rospy.logwarn(f"[FruitHarvestingSim] Unknown action: {action_name} params={params}")

        return ActionFluents(grasp, load, nav, unload)

    def _is_terminal(self, state: Dict[str, Any]) -> bool:
        """Terminal = the same condition used for the +200 end bonus in the reward."""
        fruit_at_p = state["fruit_at"].astype(bool)
        fruit_collected_p = state["fruit_collected"].astype(bool)
        fruits_unloaded_p = state["fruits_unloaded"].astype(bool)
        robot_at_p = state["robot_at"].astype(bool)

        no_ripe_fruit_at_next = bool(np.all((~fruit_at_p) | (~self.evaluator.fruit_ripe.astype(bool))))
        collected_ripe_implies_unloaded = bool(np.all((~(fruit_collected_p & self.evaluator.fruit_ripe.astype(bool))) | fruits_unloaded_p))
        at_unload = bool(np.any(robot_at_p & self.evaluator.unload_station.astype(bool)))
        return bool(at_unload and no_ripe_fruit_at_next and collected_ripe_implies_unloaded)

    def _state_to_keyvalues(self, state: Dict[str, Any]) -> List[KeyValue]:
        out: List[KeyValue] = []

        # positions
        for i, p in enumerate(self.positions):
            out.append(KeyValue(key=f"robot_at({p})", value="true" if bool(state["robot_at"][i]) else "false"))
            out.append(KeyValue(key=f"position_visited({p})", value="true" if bool(state["position_visited"][i]) else "false"))

        # locations
        for i, l in enumerate(self.locations):
            out.append(KeyValue(key=f"fruit_at({l})", value="true" if bool(state["fruit_at"][i]) else "false"))
            out.append(KeyValue(key=f"fruit_collected({l})", value="true" if bool(state["fruit_collected"][i]) else "false"))
            out.append(KeyValue(key=f"fruit_in_bin({l})", value="true" if bool(state["fruit_in_bin"][i]) else "false"))
            out.append(KeyValue(key=f"fruits_unloaded({l})", value="true" if bool(state["fruits_unloaded"][i]) else "false"))

        return out


class SimInterface:
    """Factory wrapper (kept for compatibility with the original code)."""

    def __init__(self, domain_file: Optional[str] = None, instance_file: Optional[str] = None):
        self.domain_file = domain_file or rospy.get_param("~domain_file", "")
        self.instance_file = instance_file or rospy.get_param("~instance_file", "")

    def robot_sim(self, robot_id: str):
        if robot_id != "Robotnik":
            rospy.logwarn("[SimInterface] Only 'Robotnik' is supported in this mock.")
        if not self.domain_file or not self.instance_file:
            raise ValueError("domain_file and instance_file must be provided (or set as ROS params)")
        return FruitHarvestingSim(self.domain_file, self.instance_file)
