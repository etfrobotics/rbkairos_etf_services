#!/usr/bin/env python3
import rospy
import threading
import time

from rbkairos_etf_services.srv import Pick, PickResponse

import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander

import actionlib
import franka_gripper.msg


class PickServiceNode:
    def __init__(self):
        rospy.init_node("pick_service_node", anonymous=False)

        self.group_name = rospy.get_param("~move_group", "arm")
        self.gripper_move_action = rospy.get_param("~gripper_move_action", "/robot/arm/franka_gripper/move")
        self.gripper_speed = rospy.get_param("~gripper_speed", 0.1)

        moveit_commander.roscpp_initialize([])
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.move_group = MoveGroupCommander(self.group_name)

        # Your example "home"
        self.joint_goal_home = [
            -1.0 * 3.1415/180, -51.0 * 3.1415/180, 0.0 * 3.1415/180,
            -156.0 * 3.1415/180, -2.0 * 3.1415/180, 107.0 * 3.1415/180, 48.0 * 3.1415/180
        ]

        self._lock = threading.Lock()

        self.srv = rospy.Service("pick", Pick, self.handle_pick)
        rospy.loginfo("Pick service ready (service name: /pick).")

    def _move_joints(self, joints, done_event: threading.Event, result_holder: dict):
        try:
            with self._lock:
                ok = self.move_group.go(joints, wait=True)
                self.move_group.stop()
            result_holder["ok"] = bool(ok)
            result_holder["msg"] = "Joint motion done." if ok else "MoveIt joint motion failed."
        except Exception as e:
            result_holder["ok"] = False
            result_holder["msg"] = f"Exception during joint motion: {e}"
        finally:
            done_event.set()

    def _move_gripper(self, width: float, timeout_s: float) -> (bool, str):
        try:
            client = actionlib.SimpleActionClient(self.gripper_move_action, franka_gripper.msg.MoveAction)
            if not client.wait_for_server(rospy.Duration(min(5.0, timeout_s))):
                return False, "Gripper action server not available."

            goal = franka_gripper.msg.MoveGoal(width=width, speed=self.gripper_speed)
            client.send_goal(goal)

            if not client.wait_for_result(rospy.Duration(timeout_s)):
                client.cancel_goal()
                return False, "Timeout waiting for gripper result."
            return True, "Gripper moved."
        except Exception as e:
            return False, f"Exception controlling gripper: {e}"

    def _pick_routine(self, timeout_s: float, done_event: threading.Event, result_holder: dict):
        """
        A simple sequence:
          1) go home
          2) approach pose (joint_goal_1)
          3) close gripper
          4) lift/retreat (joint_goal_2)
          5) open gripper slightly
          6) back home
        """
        start = time.time()

        def remaining():
            return max(0.0, timeout_s - (time.time() - start))

        # Example joints from your script
        joint_goal_1 = [
            -3.0 * 3.1415/180, 0.0 * 3.1415/180, 3.0 * 3.1415/180,
            -98.0 * 3.1415/180, -2.0 * 3.1415/180, 98.0 * 3.1415/180, 43.0 * 3.1415/180
        ]
        joint_goal_2 = [
            -4.0 * 3.1415/180, -31.0 * 3.1415/180, 5.0 * 3.1415/180,
            -104.0 * 3.1415/180, -2.0 * 3.1415/180, 74.0 * 3.1415/180, 47.0 * 3.1415/180
        ]

        # 1) Home
        if remaining() <= 0.0:
            result_holder.update(ok=False, msg="Timeout before starting.")
            done_event.set()
            return
        ok = self.move_group.go(self.joint_goal_home, wait=True)
        self.move_group.stop()
        if not ok:
            result_holder.update(ok=False, msg="Failed to go home.")
            done_event.set()
            return

        # 2) Approach
        if remaining() <= 0.0:
            result_holder.update(ok=False, msg="Timeout before approach.")
            done_event.set()
            return
        ok = self.move_group.go(joint_goal_1, wait=True)
        self.move_group.stop()
        if not ok:
            result_holder.update(ok=False, msg="Failed approach motion.")
            done_event.set()
            return

        # 3) Close gripper
        r = remaining()
        if r <= 0.0:
            result_holder.update(ok=False, msg="Timeout before closing gripper.")
            done_event.set()
            return
        okg, msgg = self._move_gripper(width=0.0, timeout_s=min(r, 10.0))
        if not okg:
            result_holder.update(ok=False, msg=f"Gripper close failed: {msgg}")
            done_event.set()
            return

        # 4) Lift/retreat
        if remaining() <= 0.0:
            result_holder.update(ok=False, msg="Timeout before retreat.")
            done_event.set()
            return
        ok = self.move_group.go(joint_goal_2, wait=True)
        self.move_group.stop()
        if not ok:
            result_holder.update(ok=False, msg="Failed retreat motion.")
            done_event.set()
            return

        # 5) Open slightly (optional)
        r = remaining()
        if r <= 0.0:
            result_holder.update(ok=False, msg="Timeout before opening gripper.")
            done_event.set()
            return
        okg, msgg = self._move_gripper(width=0.1, timeout_s=min(r, 10.0))
        if not okg:
            result_holder.update(ok=False, msg=f"Gripper open failed: {msgg}")
            done_event.set()
            return

        # 6) Back home
        if remaining() <= 0.0:
            result_holder.update(ok=False, msg="Timeout before returning home.")
            done_event.set()
            return
        ok = self.move_group.go(self.joint_goal_home, wait=True)
        self.move_group.stop()
        if not ok:
            result_holder.update(ok=False, msg="Failed to return home.")
            done_event.set()
            return

        result_holder.update(ok=True, msg="Pick routine completed.")
        done_event.set()

    def handle_pick(self, req):
        timeout_s = float(req.timeout) if req.timeout > 0.0 else 30.0

        done = threading.Event()
        result_holder = {"ok": False, "msg": ""}

        th = threading.Thread(target=self._pick_routine, args=(timeout_s, done, result_holder))
        th.daemon = True
        th.start()

        if not done.wait(timeout=timeout_s):
            # timed out: stop arm if possible
            try:
                with self._lock:
                    self.move_group.stop()
            except Exception:
                pass
            return PickResponse(False, "Timeout executing pick routine. Stopped arm motion.")

        return PickResponse(bool(result_holder["ok"]), str(result_holder["msg"]))


if __name__ == "__main__":
    PickServiceNode()
    rospy.spin()
