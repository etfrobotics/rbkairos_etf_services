#!/usr/bin/env python3
import rospy
import time
import actionlib
import tf.transformations as tf_trans

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

from rbkairos_etf_services.srv import MoveBase, MoveBaseRequest, MoveArm, MoveArmRequest
from rbkairos_etf_services.srv import ActionServer, ActionServerResponse

import franka_gripper.msg


class ActionManagerNode:
    def __init__(self):
        rospy.init_node("action_manager_node", anonymous=False)

        # ---------- Params ----------
        self.world_frame = rospy.get_param("~world_frame", "fr3_link0")

        # Gripper
        self.gripper_action_name = rospy.get_param("~gripper_action", "/robot/arm/franka_gripper/move")
        self.gripper_speed = float(rospy.get_param("~gripper_speed", 0.1))

        # Gripper widths (meters)
        self.grasp_width = float(rospy.get_param("~grasp_width", 0.0))       # closed
        self.open_width  = float(rospy.get_param("~open_width", 0.08))       # open

        # Default timeouts
        self.default_timeout = float(rospy.get_param("~default_timeout", 30.0))
        self.arm_step_timeout = float(rospy.get_param("~arm_step_timeout", 10.0))
        self.base_step_timeout = float(rospy.get_param("~base_step_timeout", 30.0))
        self.gripper_timeout = float(rospy.get_param("~gripper_timeout", 10.0))

        # Example constant poses (fill these with correct ones for your system)
        # Format: [x, y, z, R, P, Y]
        self.HOME_POSE = rospy.get_param("~home_pose", [0.4, 0.0, 0.4, 3.14, 0.0, 0.0])

        # "Load to bin" pose: constant position with negative x value (as you said)
        self.LOAD_BIN_POSE = rospy.get_param("~load_bin_pose", [-0.3, 0.2, 0.35, 3.14, 0.0, 1.57])

        # "Unload" pose (can be the same as load bin pose or separate)
        self.UNLOAD_POSE = rospy.get_param("~unload_pose", [-0.3, -0.2, 0.35, 3.14, 0.0, -1.57])

        # ---------- Clients ----------
        rospy.wait_for_service("/move_base")
        rospy.wait_for_service("/move_arm")

        self.move_base_srv = rospy.ServiceProxy("/move_base", MoveBase)
        self.move_arm_srv  = rospy.ServiceProxy("/move_arm", MoveArm)

        self.gripper_client = actionlib.SimpleActionClient(
            self.gripper_action_name,
            franka_gripper.msg.MoveAction
        )
        rospy.loginfo(f"Waiting for gripper action server: {self.gripper_action_name}")
        self.gripper_client.wait_for_server()
        rospy.loginfo("Gripper action server connected.")

        # ---------- Feedback Subscriber ----------
        self.feedback_sub = rospy.Subscriber("/action_feedback", Bool, self.feedback_callback)
        self.feedback = False

        # ---------- Service ----------
        self.srv = rospy.Service("action_server", ActionServer, self.handle_action)
        rospy.loginfo("ActionOrchestrator service ready on /action_server")

    def feedback_callback(self, msg):
        self.feedback = msg.data

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def euler_to_quat_xyzw(self, R, P, Y):

        q = tf_trans.quaternion_from_euler(R, P, Y)  # (x,y,z,w)

        return q[1], q[2], q[3], q[0] # returns to a MoveIT preffered form

    def pose_stamped_from_array(self, arr6):
        """
        arr6: [x, y, z, R, P, Y]
        returns PoseStamped in world frame
        """
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.header.stamp = rospy.Time.now()

        ps.pose.position.x = float(arr6[0])
        ps.pose.position.y = float(arr6[1])
        ps.pose.position.z = float(arr6[2])

        qx, qy, qz, qw = self.euler_to_quat_xyzw(arr6[3], arr6[4], arr6[5])
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        return ps

    def call_move_base(self, x, y, theta, timeout):
        req = MoveBaseRequest()
        req.x = float(x)
        req.y = float(y)
        req.theta = float(theta)
        req.timeout = float(timeout)
        resp = self.move_base_srv(req)
        return resp.success, resp.message

    def call_move_arm_single(self, pose_stamped, timeout):
        """
        Calls /move_arm with a single PoseStamped target.
        Assumes your MoveArm service uses an array `targets`.
        """
        req = MoveArmRequest()
        req.targets = [pose_stamped]
        req.timeout = float(timeout)
        resp = self.move_arm_srv(req)
        return resp.success, resp.message

    def move_gripper(self, width, timeout):
        """
        Franka gripper move action.
        """
        goal = franka_gripper.msg.MoveGoal(width=float(width), speed=float(self.gripper_speed))
        self.gripper_client.send_goal(goal)

        ok = self.gripper_client.wait_for_result(rospy.Duration(timeout))
        if not ok:
            self.gripper_client.cancel_goal()
            return False, f"Gripper timeout after {timeout}s"

        result = self.gripper_client.get_result()
        # Franka MoveAction result typically includes `success`
        if hasattr(result, "success"):
            return bool(result.success), "Gripper move done."
        return True, "Gripper move done (no success field in result)."

    # -------------------------------------------------------------------------
    # Macro actions
    # -------------------------------------------------------------------------
    def action_navigate(self, arr6, timeout):
        x = arr6[0]
        y = arr6[1]
        theta = arr6[2]  # z used as theta
        return self.call_move_base(x, y, theta, timeout)

    def action_grasp(self, arr6, timeout):
        """
        Grasp sequence:
        1) go HOME
        2) move to target pose
        3) close gripper
        4) go HOME
        """
        start = time.time()

        def remaining():
            return max(0.0, timeout - (time.time() - start))

        # 1) Home
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to go HOME: {msg}"

        # 2) Target pose
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(arr6),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to reach target: {msg}"

        # 3) Close gripper
        ok, msg = self.move_gripper(self.grasp_width, timeout=min(self.gripper_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to close gripper: {msg}"

        # 4) Back home
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to return HOME: {msg}"

        return True, "GRASP: success."

    def action_load_to_bin(self, action_id, timeout):
        """
        Always the same sequence:
        1) move arm to LOAD_BIN_POSE
        2) open gripper
        3) (optional) return HOME
        """
        start = time.time()

        def remaining():
            return max(0.0, timeout - (time.time() - start))

        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.LOAD_BIN_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: failed to reach load bin pose: {msg}"

        ok, msg = self.move_gripper(self.open_width, timeout=min(self.gripper_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: failed to open gripper: {msg}"

        # Optional: go home
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: opened but failed returning HOME: {msg}"

        return True, f"{action_id}: success."

    def action_unload(self, action_id, timeout):
        """
        Similar to load-to-bin but potentially different pose.
        """
        start = time.time()

        def remaining():
            return max(0.0, timeout - (time.time() - start))

        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.UNLOAD_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: failed to reach unload pose: {msg}"

        ok, msg = self.move_gripper(self.open_width, timeout=min(self.gripper_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: failed to open gripper: {msg}"

        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: opened but failed returning HOME: {msg}"

        return True, f"{action_id}: success."

    # -------------------------------------------------------------------------
    # Service handler
    # -------------------------------------------------------------------------

    def handle_action(self, req):
        action_id = req.action_id.strip().upper()
        arr = list(req.input)

        if len(arr) != 6:
            self.feedback_pub.publish(False)
            return ActionServerResponse(False, "Input must be float64[6] = [x,y,z,R,P,Y]")

        timeout_s = float(req.timeout) if req.timeout > 0.0 else self.default_timeout

        rospy.loginfo(f"ActionServer request: action_id={action_id}, input={arr}, timeout={timeout_s}")

  
        ok = False
        msg = ""
            
        if action_id == "navigate":
            ok, msg = self.action_navigate(arr, timeout=min(self.base_step_timeout, timeout_s))

        elif action_id == "grasp_fruit":
            ok, msg = self.action_grasp(arr, timeout=timeout_s)

        elif action_id == "load_to_bin":
            ok, msg = self.action_load_to_bin(action_id, timeout=timeout_s)

        elif action_id == "navigate_to_unloading_station":
            # exactly like NAVIGATE, but you can keep it separate if you want different logging/logic
            ok, msg = self.action_navigate(arr, timeout=min(self.base_step_timeout, timeout_s))

        elif action_id == "unload":
            ok, msg = self.action_unload(action_id, timeout=timeout_s)

        else:
            ok = False
            msg = f"Unknown action_id '{action_id}'. Supported: navigate, grasp_fruit, load_to_bin, navigate_to_unloading_station, unload"

            # Checks if the service performed the action in the software, 
            # and also if the current action was perfomed by the robot in the real world
            
            # Waits for the feedback from the task detection
            time.sleep(1)

            # If the feedback is not received in 1 second, the action is considered as failed
            real_ok = ok and self.feedback
        
            return ActionServerResponse(real_ok, msg)

if __name__ == "__main__":
    ActionManagerNode()
    rospy.spin()
