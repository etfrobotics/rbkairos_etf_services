#!/usr/bin/env python3

import json
import math
import os
import time

import numpy as np
import rospkg
import rospy

from nav_msgs.msg import Odometry
from std_msgs.msg import String
from prost_ros.msg import KeyValue
from prost_ros.srv import StartPlanning, SubmitObservation
from rbkairos_etf_services.srv import ActionServer

# Package Imports
from rbkairos_etf_services.evaluator import FruitHarvestingRewardEvaluator


def get_problem_data_paths(
    package_name="rbkairos_etf_services",
    domain_name="fruit_collection_domain.rddl",
    instance_name="instance_multi_robot.rddl",
    actions_name="actions.json",
):
    rp = rospkg.RosPack()
    pkg_path = rp.get_path(package_name)
    problem_data = os.path.join(pkg_path, "problem_data")
    return (
        os.path.join(problem_data, domain_name),
        os.path.join(problem_data, instance_name),
        os.path.join(problem_data, actions_name),
    )


class ServiceCaller:
    TERMINAL_PLANNER_ACTIONS = {"ERROR"}
    RESTART_PLANNER_ACTIONS  = {"ROUND_END", "FAILED"}
    MAX_PLANNING_RESTARTS    = 5
    SIM_ROBOT_NS = {
        "robot1": "robot",
        "robot2": "robot_b",
    }

    def __init__(self, domain_file, instance_file, actions_file):
        bridge_ns = rospy.get_param("~prost_bridge_ns", "/prost_bridge")

        rospy.wait_for_service(f"{bridge_ns}/start_planning")
        rospy.wait_for_service(f"{bridge_ns}/submit_observation")

        self.start_planning = rospy.ServiceProxy(f"{bridge_ns}/start_planning", StartPlanning)
        self.submit_obs = rospy.ServiceProxy(f"{bridge_ns}/submit_observation", SubmitObservation)

        with open(domain_file, 'r') as f:
            self.domain = f.read()
        with open(instance_file, 'r') as f:
            self.instance = f.read()

        rospy.loginfo("Sending the Planning request")
        self.resp = self.start_planning(self.domain, self.instance, 60)
        if not self.resp.success:
            rospy.logerr("PROST server failed")

        self.evaluator = FruitHarvestingRewardEvaluator(domain_file, instance_file)

        self.obs = self.evaluator.get_initial_state()
        self.reward = 0.0

        # Load the actions.json action-to-robot motion data
        with open(actions_file, "r", encoding="utf-8") as f:
            self.ACTION_DATA = json.load(f)

        # Object index maps (ordered by RDDL instance definition)
        self.positions = self.evaluator.positions
        self.locations = self.evaluator.locations
        self.robot_names = self.evaluator.robots
        self.robot_to_idx = self.evaluator.robot_to_idx

        self.pos_to_idx = {name: i for i, name in enumerate(self.positions)}
        self.loc_to_idx = {name: i for i, name in enumerate(self.locations)}

        # Global actions.json lookup order, independent of each robot's local RDDL instance order.
        # Example: robot2 may see a21 as local index 0, but in shared actions.json a21 is index 20.
        # These maps convert target names like a21/l55 to the correct actions.json row.
        # Local evaluator indexes are still used separately for observed_action updates.
        self.navigate_action_names = ["a{}".format(index) for index in range(1, 41)] + ["unload1", "unload2"]
        self.location_action_names = ["l{}".format(index) for index in range(1, 129)]
        self.navigate_action_to_idx = { name: index for index, name in enumerate(self.navigate_action_names) }
        self.location_action_to_idx = { name: index for index, name in enumerate(self.location_action_names) }

        self.idle_count = 0
        self._planning_restarts = 0

        # When dry_run=True the node skips all action-server calls and treats
        # every action as instantly successful.  Useful for testing the planner
        # loop without a running action_manager.
        self.dry_run = rospy.get_param("~dry_run", False)
        if self.dry_run:
            rospy.logwarn("DRY RUN mode: action server calls are SKIPPED, all actions assumed successful.")

        # Per-robot action server proxies — created lazily on first use
        self._robot_action_proxies = {}

        metrics_root = rospy.get_param("~metrics_root", "")
        if not metrics_root:
            metrics_root = os.path.join(
                rospkg.RosPack().get_path("rbkairos_etf_services"),
                "mission_metrics",
            )
        self.metrics_root = metrics_root
        self._metrics_written = False
        self._robot_metrics = {
            robot_name: {
                "start_time": None,
                "end_time": None,
                "fruits": [],
                "total_distance_m": 0.0,
                "last_odom_xy": None,
                "distance_tracking_active": False,
                "unload_station_trips": 0,
            }
            for robot_name in self.robot_names
        }
        self._odom_subscribers = {}
        for robot_name in self.robot_names:
            self._subscribe_robot_odom(robot_name)

        self._completed_waypoint_pubs = {}
        for robot_name in self.robot_names:
            sim_robot_ns = self._sim_robot_ns(robot_name)
            topic_name = f"/{sim_robot_ns}/path_nav/completed_waypoints"
            self._completed_waypoint_pubs[robot_name] = rospy.Publisher(
                topic_name,
                String,
                queue_size=1,
                latch=True,
            )
        self.publish_completed_waypoints(self.obs)

    def _sim_robot_ns(self, robot_name):
        return self.SIM_ROBOT_NS.get(robot_name, robot_name)

    def _subscribe_robot_odom(self, robot_name):
        sim_robot_ns = self._sim_robot_ns(robot_name)
        odom_topic = f"/{sim_robot_ns}/robotnik_base_control/odom_gt"
        self._odom_subscribers[robot_name] = rospy.Subscriber(
            odom_topic,
            Odometry,
            self._odom_callback,
            callback_args=robot_name,
            queue_size=10,
        )

    def _odom_callback(self, msg, robot_name):
        metrics = self._robot_metrics.get(robot_name)
        if metrics is None:
            return

        position = msg.pose.pose.position
        xy = (float(position.x), float(position.y))
        last_xy = metrics["last_odom_xy"]

        if metrics["distance_tracking_active"] and last_xy is not None:
            metrics["total_distance_m"] += math.hypot(
                xy[0] - last_xy[0],
                xy[1] - last_xy[1],
            )

        metrics["last_odom_xy"] = xy

    def _get_robot_proxy(self, robot_name: str):
        sim_robot_ns = self._sim_robot_ns(robot_name)
 
        if robot_name not in self._robot_action_proxies:
            service_name = f"/{sim_robot_ns}/sim_action_server"
            rospy.loginfo(f"Waiting for action server: {service_name}")
            rospy.wait_for_service(service_name)
            self._robot_action_proxies[robot_name] = rospy.ServiceProxy(service_name, ActionServer)
        return self._robot_action_proxies[robot_name]

    def _mark_robot_action_start(self, robot_name, start_time):
        metrics = self._robot_metrics[robot_name]
        if metrics["start_time"] is None:
            metrics["start_time"] = start_time
        metrics["distance_tracking_active"] = True

    def _mark_robot_action_end(self, robot_name, end_time):
        metrics = self._robot_metrics[robot_name]
        metrics["end_time"] = end_time
        metrics["distance_tracking_active"] = False

    def completed_waypoints_for_robot(self, obs, robot_name):
        completed_waypoints = []
        fruit_at = np.asarray(obs["fruit_at"], dtype=bool)

        for pos_idx, position_name in enumerate(self.positions):
            if self.evaluator.unload_station[pos_idx]:
                continue

            reachable_ripe = self.evaluator.reachable_from[:, pos_idx] & self.evaluator.fruit_ripe
            if not np.any(reachable_ripe):
                continue

            if np.all(~fruit_at[reachable_ripe]):
                completed_waypoints.append(position_name)

        return completed_waypoints

    def publish_completed_waypoints(self, obs):
        for robot_name, publisher in self._completed_waypoint_pubs.items():
            data = {
                "robot": robot_name,
                "completed_waypoints": self.completed_waypoints_for_robot(obs, robot_name),
            }
            publisher.publish(String(data=json.dumps(data)))

    def decode_orange_name(self, arr):
        try:
            row = int(round(float(arr[0])))
            tree = int(round(float(arr[1])))
            side_code = int(round(float(arr[2])))
            height_code = int(round(float(arr[3])))
        except (TypeError, ValueError, IndexError):
            return None

        if row < 1 or row > 4 or tree < 1 or tree > 8:
            return None
        if side_code not in (0, 1) or height_code not in (0, 1):
            return None

        side = "L" if side_code == 0 else "R"
        height = "B" if height_code == 0 else "T"
        return "orange{}.{}.{}.{}".format(row, tree, side, height)

    def _record_grasp_metric(self, robot_name, real_action, duration, moveit_planning_time_sec, grasp_cycle_time, success):
        orange_name = self.decode_orange_name(real_action)
        if orange_name is None:
            orange_name = "unknown"

        duration = float(duration)
        moveit_planning_time_sec = float(moveit_planning_time_sec)
        grasp_cycle_time = float(grasp_cycle_time)
        self._robot_metrics[robot_name]["fruits"].append({
            "fruit": orange_name,
            "grasp_time_sec": round(duration, 3),
            "moveit_planning_time_sec": round(moveit_planning_time_sec, 3),
            "grasp_cycle_time": round(grasp_cycle_time, 3),
            "success": bool(success),
        })

    def _next_metrics_file(self, robot_name):
        robot_dir = os.path.join(self.metrics_root, robot_name)
        os.makedirs(robot_dir, exist_ok=True)

        used_indexes = set()
        for file_name in os.listdir(robot_dir):
            stem, ext = os.path.splitext(file_name)
            if ext == ".json" and stem.startswith("test") and stem[4:].isdigit():
                used_indexes.add(int(stem[4:]))

        run_index = 1
        while run_index in used_indexes:
            run_index += 1

        run_name = "test{}".format(run_index)
        return run_name, os.path.join(robot_dir, "{}.json".format(run_name))

    def write_metrics_files(self):
        if self._metrics_written:
            return
        self._metrics_written = True

        for robot_name in self.robot_names:
            metrics = self._robot_metrics[robot_name]
            start_time = metrics["start_time"]
            end_time = metrics["end_time"]
            if start_time is None or end_time is None:
                total_makespan = 0.0
            else:
                total_makespan = end_time - start_time

            run_name, metrics_file = self._next_metrics_file(robot_name)
            data = {
                "robot": robot_name,
                "run_name": run_name,
                "total_makespan_sec": round(float(total_makespan), 3),
                "total_distance_traveled_m": round(float(metrics["total_distance_m"]), 3),
                "unload_station_trips": int(metrics["unload_station_trips"]),
                "fruits": metrics["fruits"],
            }

            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.write("\n")

            rospy.loginfo("Wrote mission metrics for %s to %s", robot_name, metrics_file)

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def parse_action(self, action_name, action_params):
        """Return (robot_idx, aisle_index, location_index) for one robot action."""
        robot_idx = 0
        aisle_index = -1
        location_index = -1

        if action_params:
            maybe_robot = action_params[0]
            robot_idx = self.robot_to_idx.get(maybe_robot, robot_idx)

        if action_name == "navigate" and len(action_params) >= 2:
            aisle_index = self.pos_to_idx.get(action_params[1], -1)
        elif action_name in ("grasp_fruit", "load_to_bin") and len(action_params) >= 2:
            location_index = self.loc_to_idx.get(action_params[1], -1)

        return robot_idx, aisle_index, location_index

    # Convert an action target name to its shared actions.json index.
    # Returns -1 when the action type or target has no valid motion entry.  
    def get_action_data_index(self, action_name, target_name):
        if action_name == "navigate":
            return self.navigate_action_to_idx.get(target_name, -1)
        if action_name in ("grasp_fruit", "load_to_bin"):
            return self.location_action_to_idx.get(target_name, -1)
        return -1

    # ------------------------------------------------------------------
    # Observation submission helpers
    # ------------------------------------------------------------------

    def append_scalar_fluent(self, obs_list, name, value):
        obs_list.append(KeyValue(name, "true" if bool(value) else "false"))

    def append_vector_fluent(self, obs_list, name, values, objects):
        for idx, val in enumerate(values):
            obs_list.append(KeyValue(
                f"{name}({objects[idx]})",
                "true" if bool(val) else "false",
            ))

    def append_robot_fluent(self, obs_list, name, values, objects):
        for robot_idx, robot_name in enumerate(self.robot_names):
            for obj_idx, obj_name in enumerate(objects):
                obs_list.append(KeyValue(
                    f"{name}({robot_name},{obj_name})",
                    "true" if bool(values[robot_idx, obj_idx]) else "false",
                ))

    def append_robot_scalar_fluent(self, obs_list, name, values):
        for robot_idx, robot_name in enumerate(self.robot_names):
            obs_list.append(KeyValue(
                f"{name}({robot_name})",
                "true" if bool(values[robot_idx]) else "false",
            ))

    # ------------------------------------------------------------------
    # Single-robot action dispatch
    # ------------------------------------------------------------------

    def _dispatch_robot_action(self, action_name, action_params):
        """
        Resolve motion data and dispatch one robot action to the appropriate
        physical action server.  Returns (robot_idx, action_name, action_index).

        All actions are assumed to succeed (caller guarantee).
        """
        robot_idx, aisle_index, location_index = self.parse_action(action_name, action_params)
        robot_name = self.robot_names[robot_idx]
        
        # data_index selects the real/sim motion from shared actions.json.
        # action_index remains the robot-local RDDL index for evaluator updates.
        target_name = action_params[1] if len(action_params) >= 2 else None
        data_index = self.get_action_data_index(action_name, target_name)
        current_pos_indices = np.where(self.obs["robot_at"][robot_idx])[0]
        starting_pos_name = None
        if current_pos_indices.size > 0:
            starting_pos_name = self.positions[int(current_pos_indices[0])]

        real_action = np.zeros(6, dtype=float)
        action_index = -1
        action_success = True

        if action_name == "navigate":
            if 0 <= data_index < len(self.ACTION_DATA.get(action_name, [])):
                real_action = self.ACTION_DATA[action_name][data_index]
            else:
                rospy.logwarn(
                    f"No motion entry for navigate target {target_name} (action-data index {data_index})."
                )
            action_index = aisle_index

        elif action_name == "grasp_fruit":
            if 0 <= data_index < len(self.ACTION_DATA.get(action_name, [])):
                real_action = self.ACTION_DATA[action_name][data_index]
            else:
                rospy.logwarn(
                    f"No motion entry for grasp target {target_name} (action-data index {data_index})."
                )
            action_index = location_index

        elif action_name == "load_to_bin":
            if 0 <= data_index < len(self.ACTION_DATA.get(action_name, [])):
                real_action = self.ACTION_DATA[action_name][data_index]
            else:
                rospy.logwarn(
                    f"No motion entry for load target {target_name} (action-data index {data_index})."
                )
            action_index = location_index

        # elif action_name == "unload":
        #     # The robot is already at the unload station when this action fires.
        #     # Reuse the navigate pose for that position so the action_manager
        #     # can do the final dock-and-tip sequence, consistent with how it
        #     # handles the unload action (action_navigate).
        #     current_pos = np.where(self.obs["robot_at"][robot_idx])[0]
        #     if current_pos.size > 0:
        #         pos_idx = int(current_pos[0])
        #         nav_data = self.ACTION_DATA.get("navigate", [])
        #         if 0 <= pos_idx < len(nav_data):
        #             real_action = nav_data[pos_idx]
        #         else:
        #             rospy.logwarn(f"No navigate entry for unload station at position index {pos_idx}.")
        
        elif action_name == "unload":
            unload_station_indices = np.where(self.evaluator.unload_station)[0]
            current_pos = np.where(self.obs["robot_at"][robot_idx])[0]
            current_station_idx = -1
            if current_pos.size > 0 and self.evaluator.unload_station[current_pos[0]]:
                matching = np.where(unload_station_indices == current_pos[0])[0]
                if matching.size > 0:
                    current_station_idx = int(matching[0])
            if 0 <= current_station_idx < len(self.ACTION_DATA.get(action_name, [])):
                real_action = self.ACTION_DATA[action_name][current_station_idx]
            else:
                rospy.logwarn("Could not map unload action to an unload-station motion entry.")

        elif action_name in ("wait", "NOOP"):
            pass

        else:
            rospy.logwarn(f"Unknown planner action '{action_name}'.")

        should_time_action = action_name not in ("wait", "NOOP")
        action_start_time = None
        moveit_planning_time_sec = 0.0
        grasp_cycle_time = 0.0
        proxy = None

        if action_name not in ("wait", "NOOP") and not self.dry_run:
            try:
                proxy = self._get_robot_proxy(robot_name)
            except Exception as e:
                rospy.logwarn(f"\033[91mAction server call failed for {robot_name}: {e}\033[0m")
                action_success = False

        if should_time_action:
            action_start_time = time.monotonic()
            self._mark_robot_action_start(robot_name, action_start_time)

        # Execute on the physical robot/sim and keep the returned success flag.
        if proxy is not None:
            try:
                # rospy.loginfo(
                #     "\033[93mWaiting for sim action result: %s %s on %s\033[0m \n \n",
                #     action_name,
                #     list(real_action),
                #     robot_name,
                # )
                response = proxy(action_name, real_action, 1000.0)
                action_success = bool(response.success)
                if action_success:
                    rospy.loginfo(
                        "\033[92mSim action succeeded: %s on %s -> %s\033[0m",
                        action_name,
                        robot_name,
                        response.message,
                    )
                    # rospy.sleep(2.5)
                moveit_planning_time_sec = float(getattr(response, "moveit_planning_time_sec", 0.0))
                grasp_cycle_time = float(getattr(response, "post_moveit_planning_grasp_time_sec", 0.0))
                if not action_success:
                    rospy.logwarn(
                        f"\033[91mAction server reported failure for {robot_name}: "
                        f"{action_name} {list(real_action)} -> {response.message}\033[0m"
                    )
                    # rospy.sleep(2.5)
            except Exception as e:
                rospy.logwarn(f"\033[91mAction server call failed for {robot_name}: {e}\033[0m")
                action_success = False

        if should_time_action:
            action_end_time = time.monotonic()
            action_duration = action_end_time - action_start_time
            self._mark_robot_action_end(robot_name, action_end_time)

            if (
                action_success
                and action_name == "navigate"
                and target_name in ("unload1", "unload2")
                and starting_pos_name != target_name
            ):
                self._robot_metrics[robot_name]["unload_station_trips"] += 1

            if action_name == "grasp_fruit":
                self._record_grasp_metric(
                    robot_name,
                    real_action,
                    action_duration,
                    moveit_planning_time_sec,
                    grasp_cycle_time,
                    action_success,
                )

        return robot_idx, action_name, action_index, action_success

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        while True:
            for robot_idx, robot_name in enumerate(self.robot_names):
                robot_row = list(self.obs["robot_at"][robot_idx])
                true_count = sum(1 for v in robot_row if v)
                if true_count != 1:
                    rospy.logerr(f"{robot_name} robot_at has {true_count} true values — expected 1.")

            rospy.logdebug("Sending observations and reward to planner.")

            obs_to_submit = []
            self.append_scalar_fluent(obs_to_submit, "all_fruits_done", self.obs["all_fruits_done"])
            self.append_robot_fluent(obs_to_submit, "robot_at", self.obs["robot_at"], self.positions)
            self.append_vector_fluent(obs_to_submit, "fruit_at", self.obs["fruit_at"], self.locations)
            self.append_robot_fluent(
                obs_to_submit, "fruit_collected", self.obs["fruit_collected"], self.locations
            )
            self.append_robot_fluent(
                obs_to_submit, "fruit_in_bin", self.obs["fruit_in_bin"], self.locations
            )
            self.append_vector_fluent(
                obs_to_submit, "fruits_unloaded", self.obs["fruits_unloaded"], self.locations
            )
            self.append_vector_fluent(
                obs_to_submit, "position_visited", self.obs["position_visited"], self.positions
            )
            if "prev_position" in self.obs:
                self.append_robot_fluent(
                    obs_to_submit, "prev_position", self.obs["prev_position"], self.positions
                )

            response = self.submit_obs(obs_to_submit, self.reward)
            action_name = response.action_name
            action_data = response.action_params

            # rospy.loginfo(f"\033[96mPlanner response: action_name={action_name}, params={list(action_data)}\033[0m")

            if action_name in self.TERMINAL_PLANNER_ACTIONS:
                rospy.logerr(
                    f"Planner returned terminal status '{action_name}' with params {list(action_data)}. "
                    "Stopping execution loop."
                )
                self.write_metrics_files()
                break

            # Decode joint (multi-robot) or single-robot action
            if action_name == "JOINT":
                # Bridge encoded all robot actions as JSON in action_data[0]
                robot_action_list = json.loads(action_data[0])
                robot_actions = [(entry[0], entry[1:]) for entry in robot_action_list]
            else:
                robot_actions = [(action_name, list(action_data))]

            rospy.logdebug(f"Dispatching {len(robot_actions)} robot action(s).")

            observed_action = self.evaluator.create_action_template()

            for single_action_name, single_action_params in robot_actions:
                rospy.logdebug(f"Dispatching {single_action_name} {single_action_params}")
                robot_idx, _, action_index, action_success = self._dispatch_robot_action(
                    single_action_name, single_action_params
                )

                if single_action_name == "NOOP":
                    continue

                if not action_success:
                    # rospy.logwarn(
                    #     f"Skipping RDDL action update for failed action '{single_action_name}'."
                    # )
                    continue

                if single_action_name == "unload":
                    observed_action["unload"][robot_idx] = True
                elif single_action_name == "wait":
                    observed_action["wait"][robot_idx] = True
                elif action_index >= 0:
                    observed_action[single_action_name][robot_idx][action_index] = True
                else:
                    rospy.logwarn(
                        f"Skipping state update for '{single_action_name}': target index not resolved."
                    )

            next_obs = self.evaluator.step(self.obs, observed_action)
            reward = self.evaluator.evaluate_reward(self.obs, observed_action, next_obs)
            self.publish_completed_waypoints(next_obs)

            if self.idle_count > 200 or next_obs["all_fruits_done"]:
                reward = 0.0
                rospy.loginfo("Sim finished")
                self.write_metrics_files()
                break

            self.obs = next_obs
            self.reward = reward
            self.idle_count += 1


if __name__ == "__main__":
    rospy.init_node("prost_service_caller")

    startup_delay = rospy.get_param("~startup_delay", 0.0)
    if startup_delay > 0:
        rospy.loginfo(f"Startup delay: {startup_delay}s — waiting for other planners to initialize.")
        rospy.sleep(startup_delay)

    domain_file, instance_file, actions_file = get_problem_data_paths(
        domain_name=rospy.get_param("~domain_name", "fruit_collection_domain.rddl"),
        instance_name=rospy.get_param("~instance_name", "instance_multi_robot.rddl"),
    )

    rospy.loginfo(f"Domain:   {domain_file}")
    rospy.loginfo(f"Instance: {instance_file}")

    sc = ServiceCaller(domain_file, instance_file, actions_file)
    sc.run()
