#!/usr/bin/env python3
import time
import rospy

from geometry_msgs.msg import Pose
from rbkairos_etf_services.srv import MoveArm, MoveArmResponse

import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander


class MoveArmServiceNode:
    def __init__(self):
        rospy.init_node("move_arm_service_node", anonymous=False)

        self.default_timeout = 10 # seconds
        self.default_planning_time = 2

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group_name = "arm"  # Replace with your MoveIt group name
        self.move_group = MoveGroupCommander(self.group_name)
        self.move_group.set_planning_time(self.default_planning_time)
        self.move_group.set_max_acceleration_scaling_factor(0.8)
        self.move_group.set_max_velocity_scaling_factor(0.6)

        self.world_frame = "fr3_link0"


        self.srv = rospy.Service("move_arm", MoveArm, self.handle_move_arm)
        rospy.loginfo("MoveArm service ready on /move_arm")

    def handle_move_arm(self, req):
        timeout_s = float(req.timeout) if req.timeout > 0.0 else float(self.default_timeout)

        if not req.targets:
            return MoveArmResponse(False, "No targets provided.")

        start_time = time.time()

        for i, target_pose in enumerate(req.targets):

            target_pose.header.frame_id = self.world_frame
            target_pose.header.stamp = rospy.Time.now()

            elapsed = time.time() - start_time
            remaining = timeout_s - elapsed
            if remaining <= 0.0:
                return MoveArmResponse(False, f"Timeout before executing target {i}.")

            # Limit planning time based on remaining time (but don't set it too low)
            planning_time = max(1.0, min(self.default_planning_time, remaining))
            self.move_group.set_planning_time(planning_time)

            try:
                self.move_group.set_pose_target(target_pose)
                ok = self.move_group.go(wait=True)
            except Exception as e:
                self.move_group.stop()
                self.move_group.clear_pose_targets()
                return MoveArmResponse(False, f"Exception while executing target {i}: {e}")
            finally:
                # Always clean up targets
                self.move_group.stop()
                self.move_group.clear_pose_targets()

            if not ok:
                return MoveArmResponse(False, f"Target {i} failed: MoveIt could not reach target pose.")

        return MoveArmResponse(True, f"Successfully executed {len(req.targets)} target pose(s).")


if __name__ == "__main__":
    MoveArmServiceNode()
    rospy.spin()
