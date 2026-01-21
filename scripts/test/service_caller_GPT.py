#!/usr/bin/env python3
"""service_caller.py (rewritten for real-robot integration)

This script now:
- Starts a PROST planning session via prost_ros services.
- Computes reward using evaluator.py (FruitHarvestingRewardEvaluator), following
  the same pattern as evaluator.py's demo: reward = evaluate_reward(obs_with_action, next_obs).
- Calls the action orchestrator service `/action_server` (see action_manager.py)
  to execute high-level actions on the robot.
- Uses a world-state provider. For now, we wire a mock implementation
  (real_world_state_mock.py) that advances deterministic mock dynamics.

When running on the real robot, replace the state provider with one that reads
robot pose, bin content state, and fruit perception.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rospy

from prost_ros.srv import StartPlanning, SubmitObservation
from prost_bridge.msg import KeyValue

from evaluator import FruitHarvestingRewardEvaluator

import numpy as np
from std_msgs.msg import Bool
from pyRDDLGym.core.env import RDDLEnv
from rbkairos_etf_services.srv import ActionServer, ActionServerRequest

from real_world_state_mock import RealWorldStateMock


TERMINATION_STRING = "ROUND_END"


def _get_objects_by_type(env_model: Any, type_name: str) -> List[str]:
    if hasattr(env_model, "_objects") and isinstance(env_model._objects, dict) and type_name in env_model._objects:
        return list(env_model._objects[type_name])

    obj_to_type = getattr(env_model, "_object_to_type", {}) or {}
    obj_to_idx = getattr(env_model, "_object_to_index", {}) or {}
    objs = [o for o, t in obj_to_type.items() if t == type_name]
    objs.sort(key=lambda o: (obj_to_idx.get(o, 10**9), o))
    return objs


@dataclass
class ActionMapping:
    """Mapping from symbolic planner params -> robot execution poses.

    Populate these via ROS params or your own map server.
    """

    # base targets: symbol -> (x, y, theta)
    base_goals: Dict[str, Tuple[float, float, float]]

    # grasp targets: location symbol -> [x, y, z, R, P, Y]
    grasp_poses: Dict[str, List[float]]


class ActionExecutor:
    """Executes PROST actions via /action_server.

    action_manager.py expects:
      - action_id: NAVIGATE / GRASP / LOAD_TO_BIN / UNLOAD
      - input: float64[6] = [x,y,z,R,P,Y] (NAVIGATE uses x,y,theta from the same array)
    """

    def __init__(self, mapping: ActionMapping, enable: bool = True):
        self.mapping = mapping
        self.enable = enable
        self._srv = None
        
        if self.enable:
            rospy.wait_for_service("/action_server")
            self._srv = rospy.ServiceProxy("/action_server", ActionServer)

    def execute(self, action_name: str, action_params: Sequence[str], timeout_s: float = 60.0) -> Tuple[bool, str]:
        if not self.enable:
            rospy.loginfo(f"[ActionExecutor] dry-run: {action_name} {list(action_params)}")
            return True, "dry-run"

        a = (action_name or "").strip()
        params = list(action_params) if action_params is not None else []

        if a == "navigate":
            if not params:
                return False, "navigate missing target"
            tgt = params[0]
            if tgt not in self.mapping.base_goals:
                return False, f"navigate target '{tgt}' not in base_goals map"
            x, y, th = self.mapping.base_goals[tgt]
            arr = [float(x), float(y), float(th), 0.0, 0.0, 0.0]
            return self._call("NAVIGATE", arr, timeout_s)

        if a == "grasp_fruit":
            if not params:
                return False, "grasp_fruit missing location"
            loc = params[0]
            if loc not in self.mapping.grasp_poses:
                return False, f"grasp location '{loc}' not in grasp_poses map"
            arr = list(map(float, self.mapping.grasp_poses[loc]))
            if len(arr) != 6:
                return False, f"grasp pose for '{loc}' must have 6 values"
            return self._call("GRASP", arr, timeout_s)

        if a == "load_to_bin":
            # ActionManager uses a constant pose; input is still required.
            return self._call("LOAD_TO_BIN", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], timeout_s)

        if a == "unload":
            return self._call("UNLOAD", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], timeout_s)

        if a == "" or a.lower() == "noop":
            return True, "noop"

        return False, f"unknown action '{action_name}'"

    def _call(self, action_id: str, arr6: List[float], timeout_s: float) -> Tuple[bool, str]:
        req = ActionServerRequest()
        req.action_id = action_id
        req.input = arr6
        req.timeout = float(timeout_s)
        try:
            resp = self._srv(req)
            return bool(resp.success), str(resp.message)
        except rospy.ServiceException as e:
            return False, f"Service call failed: {e}"


class ProstLoop:
    def __init__(
        self,
        domain_file: str,
        instance_file: str,
        prost_timeout_s: int = 60,
        horizon: int = 150,
        execute_real_actions: bool = False,
        use_mock_world: bool = True,
    ):
        rospy.init_node("prost_service_caller", anonymous=False)

        # ------------------------------------------------------------------
        # PROST bridge services
        # ------------------------------------------------------------------
        rospy.wait_for_service("/prost_bridge/start_planning")
        rospy.wait_for_service("/prost_bridge/submit_observation")

        self.start_planning = rospy.ServiceProxy("/prost_bridge/start_planning", StartPlanning)
        self.submit_obs = rospy.ServiceProxy("/prost_bridge/submit_observation", SubmitObservation)

        # ------------------------------------------------------------------
        # Feedback Listener
        # ------------------------------------------------------------------
        self.latest_feedback = None
        self.feedback_sub = rospy.Subscriber("/action_feedback", Bool, self._feedback_cb)
        
        with open(domain_file, "r") as f:
            domain_content = f.read()
        with open(instance_file, "r") as f:
            instance_content = f.read()

        rospy.loginfo("Sending start_planning to /prost_bridge")
        try:
            resp = self.start_planning(domain_content, instance_content, int(prost_timeout_s))
            if not resp.success:
                raise RuntimeError("/prost_bridge/start_planning returned success=false")
        except rospy.ServiceException as e:
            raise RuntimeError(f"Failed to call start_planning: {e}")

        # ------------------------------------------------------------------
        # Evaluator init (integrates evaluator.py __main__ init steps)
        # ------------------------------------------------------------------
        self.env = RDDLEnv(domain=domain_file, instance=instance_file)
        self.model = self.env.model
        self.evaluator = FruitHarvestingRewardEvaluator(self.model)
        self.positions = _get_objects_by_type(self.model, "aisle_position")
        self.locations = _get_objects_by_type(self.model, "location")
        self.pos_to_i = {p: i for i, p in enumerate(self.positions)}
        self.loc_to_i = {l: i for i, l in enumerate(self.locations)}

        # ------------------------------------------------------------------
        # World state provider
        # ------------------------------------------------------------------
        if use_mock_world:
            self.world = RealWorldStateMock(domain_file, instance_file)
        else:
            # Placeholder: wire your real robot sensors here.
            self.world = RealWorldStateMock(domain_file, instance_file)
            rospy.logwarn("use_mock_world=False requested, but real provider not implemented; using mock")

        # ------------------------------------------------------------------
        # Action execution mapping (fill these via ROS params)
        # ------------------------------------------------------------------
        base_goals = rospy.get_param("~base_goals", {})
        grasp_poses = rospy.get_param("~grasp_poses", {})
        mapping = ActionMapping(base_goals=base_goals, grasp_poses=grasp_poses)

        self.executor = ActionExecutor(mapping, enable=bool(execute_real_actions))

        self.horizon = int(horizon)
        self.t = 0
        self.last_reward = 0.0

    def _feedback_cb(self, msg: Bool):
        self.latest_feedback = msg.data

    # ------------------------------------------------------------------
    # Observation conversion
    # ------------------------------------------------------------------

    def _state_to_keyvalues(self, state: Dict[str, Any]) -> List[KeyValue]:
        out: List[KeyValue] = []

        for i, p in enumerate(self.positions):
            out.append(KeyValue(key=f"robot_at({p})", value="true" if bool(state["robot_at"][i]) else "false"))
            out.append(KeyValue(key=f"position_visited({p})", value="true" if bool(state["position_visited"][i]) else "false"))

        for i, l in enumerate(self.locations):
            out.append(KeyValue(key=f"fruit_at({l})", value="true" if bool(state["fruit_at"][i]) else "false"))
            out.append(KeyValue(key=f"fruit_collected({l})", value="true" if bool(state["fruit_collected"][i]) else "false"))
            out.append(KeyValue(key=f"fruit_in_bin({l})", value="true" if bool(state["fruit_in_bin"][i]) else "false"))
            out.append(KeyValue(key=f"fruits_unloaded({l})", value="true" if bool(state["fruits_unloaded"][i]) else "false"))

        return out

    def _planner_action_to_action_dict(self, action_name: str, action_params: Sequence[str]) -> Dict[str, Any]:
        act = self.evaluator.create_observation_template()
        act["grasp_fruit"][:] = False
        act["load_to_bin"][:] = False
        act["navigate"][:] = False
        act["unload"] = False

        a = (action_name or "").strip()
        params = list(action_params) if action_params is not None else []

        if a == "navigate" and params:
            tgt = params[0]
            if tgt in self.pos_to_i:
                act["navigate"][self.pos_to_i[tgt]] = True

        elif a == "grasp_fruit" and params:
            loc = params[0]
            if loc in self.loc_to_i:
                act["grasp_fruit"][self.loc_to_i[loc]] = True

        elif a == "load_to_bin" and params:
            loc = params[0]
            if loc in self.loc_to_i:
                act["load_to_bin"][self.loc_to_i[loc]] = True

        elif a == "unload":
            act["unload"] = True

        return act

    def _merge_state_and_action(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        merged = self.evaluator.create_observation_template()
        for k in ("fruit_at", "fruit_collected", "fruit_in_bin", "fruits_unloaded", "robot_at", "position_visited"):
            merged[k] = state[k].copy()
        for k in ("grasp_fruit", "load_to_bin", "navigate"):
            merged[k] = action[k].copy()
        merged["unload"] = bool(action["unload"])
        return merged

    # ------------------------------------------------------------------
    # Main planning/execution loop
    # ------------------------------------------------------------------

    def run(self):
        # Initial obs (t=0)
        state = self.world.get_state()
        obs_kv = self._state_to_keyvalues(state)
        rospy.loginfo("Submitting initial observation (t=0)")
        act_resp = self.submit_obs(obs_kv, float(self.last_reward))

        rospy.loginfo(f"PROST action @t=0: {act_resp.action_name} {list(act_resp.action_params)}")

        while (
            act_resp.action_name != TERMINATION_STRING
            and self.t < self.horizon
            and not rospy.is_shutdown()
        ):
            action_name = act_resp.action_name
            action_params = list(act_resp.action_params)

            # 1) Execute on robot (or dry-run)
            self.latest_feedback = None  # Clear previous feedback
            ok, msg = self.executor.execute(action_name, action_params)
            
            success = ok
            if ok:
                # Wait for feedback from topic (published by ActionManager)
                # We give it a small grace period if execution was successful locally
                start_wait = rospy.Time.now()
                while self.latest_feedback is None and (rospy.Time.now() - start_wait).to_sec() < 2.0:
                    rospy.sleep(0.05)
                
                if self.latest_feedback is not None:
                    success = self.latest_feedback
                    rospy.loginfo(f"Received feedback: {success}")
                else:
                    rospy.logwarn("No feedback received on /action_feedback, assuming local success status.")
            
            if not success:
               rospy.logwarn(f"Action failed (executor={ok}, feedback={self.latest_feedback}): {msg}")

            # 2) Update world model / read real next state
            #    For mock: advance deterministic dynamics using evaluator.simulate_step,
            #    ONLY IF the action was successful.
            pre_state = self.world.get_state()
            action_dict = self._planner_action_to_action_dict(action_name, action_params)
            
            # Use the correct method name `apply_planner_action` and pass success flag
            next_state = self.world.apply_planner_action(action_name, action_params, success=success)

            # 3) Compute reward from transition
            obs_with_action = self._merge_state_and_action(pre_state, action_dict)
            self.last_reward = float(self.evaluator.evaluate_reward(obs_with_action, next_state))

            # 4) Send next observation + reward to PROST
            obs_kv = self._state_to_keyvalues(next_state)
            act_resp = self.submit_obs(obs_kv, float(self.last_reward))

            self.t += 1
            rospy.loginfo(f"PROST action @t={self.t}: {act_resp.action_name} {list(act_resp.action_params)}")


if __name__ == "__main__":
    # Defaults kept close to the original script, but configurable via ROS params.
    domain_file = rospy.get_param(
        "~domain_file",
        "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_domain.rddl",
    )
    instance_file = rospy.get_param(
        "~instance_file",
        "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_inst.rddl",
    )

    horizon = int(rospy.get_param("~horizon", 150))
    prost_timeout_s = int(rospy.get_param("~prost_timeout_s", 60))

    execute_real_actions = bool(rospy.get_param("~execute_real_actions", False))
    use_mock_world = bool(rospy.get_param("~use_mock_world", True))

    loop = ProstLoop(
        domain_file=domain_file,
        instance_file=instance_file,
        prost_timeout_s=prost_timeout_s,
        horizon=horizon,
        execute_real_actions=execute_real_actions,
        use_mock_world=use_mock_world,
    )
    loop.run()
