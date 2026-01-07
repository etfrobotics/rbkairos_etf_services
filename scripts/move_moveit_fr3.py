#!/usr/bin/env python3
import rospy
import threading
import time

from geometry_msgs.msg import Pose
from rbkairos_etf_services.srv import MoveArm, MoveArmResponse

import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander


class MoveArmServiceNode:
    def __init__(self):
        rospy.init_node("move_arm_service_node", anonymous=False)

        self.group_name = rospy.get_param("~move_group", "arm")
        self.ee_link = rospy.get_param("~ee_link", None)  # optional
        self.max_vel = rospy.get_param("~max_velocity_scaling", 0.3)
        self.max_acc = rospy.get_param("~max_acceleration_scaling", 0.3)

        moveit_commander.roscpp_initialize([])
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander(self.group_name)

        if self.ee_link:
            self.move_group.set_end_effector_link(self.ee_link)

        self.move_group.set_max_velocity_scaling_factor(self.max_vel)
        self.move_group.set_max_acceleration_scaling_factor(self.max_acc)

        self._lock = threading.Lock()

        self.srv = rospy.Service("move_arm", MoveArm, self.handle_move_arm)
        rospy.loginfo("MoveArm service ready on ~move_arm (service name: /move_arm if in global ns).")

    def _execute_pose(self, pose: Pose, done_event: threading.Event, result_holder: dict):
        """
        Runs MoveIt planning/execution in a thread so we can enforce a timeout from the service callback.
        """
        try:
            with self._lock:
                self.move_group.set_pose_target(pose)
                ok = self.move_group.go(wait=True)
                self.move_group.stop()
                self.move_group.clear_pose_targets()

            result_holder["ok"] = bool(ok)
            result_holder["msg"] = "Reached target pose." if ok else "MoveIt failed to reach target pose."
        except Exception as e:
            result_holder["ok"] = False
            result_holder["msg"] = f"Exception during MoveIt execution: {e}"
        finally:
            done_event.set()

    def handle_move_arm(self, req):
        if req.timeout <= 0.0:
            timeout_s = 30.0
        else:
            timeout_s = float(req.timeout)

        if len(req.targets) == 0:
            return MoveArmResponse(False, "No targets provided.")

        start_time = time.time()

        for i, target_pose in enumerate(req.targets):
            remaining = timeout_s - (time.time() - start_time)
            if remaining <= 0.0:
                return MoveArmResponse(False, f"Timeout before executing target {i}.")

            done = threading.Event()
            result_holder = {"ok": False, "msg": ""}

            th = threading.Thread(target=self._execute_pose, args=(target_pose, done, result_holder))
            th.daemon = True
            th.start()

            if not done.wait(timeout=remaining):
                # Timed out: attempt to stop
                try:
                    with self._lock:
                        self.move_group.stop()
                        self.move_group.clear_pose_targets()
                except Exception:
                    pass
                return MoveArmResponse(False, f"Timeout while executing target {i}. Stopped motion.")

            if not result_holder["ok"]:
                return MoveArmResponse(False, f"Target {i} failed: {result_holder['msg']}")

        return MoveArmResponse(True, f"Successfully executed {len(req.targets)} target pose(s).")


if __name__ == "__main__":
    MoveArmServiceNode()
    rospy.spin()
