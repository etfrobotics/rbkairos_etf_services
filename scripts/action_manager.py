#!/usr/bin/env python3
import rospy
import time
import actionlib
import tf.transformations as tf_trans

from geometry_msgs.msg import PoseStamped, PointStamped
from std_msgs.msg import Bool

from rbkairos_etf_services.srv import MoveBase, MoveBaseRequest, MoveArm, MoveArmRequest
from rbkairos_etf_services.srv import ActionServer, ActionServerResponse

import franka_gripper.msg


class ActionManagerNode:
    def __init__(self):
        rospy.init_node("action_manager_node", anonymous=False)

        # ---------- Params ----------
        # robot_namespace identifies which physical robot this node controls.
        # The node advertises /<robot_namespace>/action_server and connects to
        # /<robot_namespace>/move_base and /<robot_namespace>/move_arm.
        self.robot_ns = rospy.get_param("~robot_namespace", "robot")

        self.world_frame = rospy.get_param("~world_frame", "fr3_link0")

        # Gripper
        self.gripper_action_name = rospy.get_param(
            "~gripper_action", f"/{self.robot_ns}/arm/franka_gripper/move"
        )
        self.gripper_speed = float(rospy.get_param("~gripper_speed", 0.1))

        # Gripper widths (meters)
        self.grasp_width = float(rospy.get_param("~grasp_width", 0.04))       # closed
        self.open_width  = float(rospy.get_param("~open_width", 0.08))       # open

        # Default timeouts
        self.default_timeout = float(rospy.get_param("~default_timeout", 30.0))
        self.arm_step_timeout = float(rospy.get_param("~arm_step_timeout", 10.0))
        self.base_step_timeout = float(rospy.get_param("~base_step_timeout", 30.0))
        self.gripper_timeout = float(rospy.get_param("~gripper_timeout", 10.0))

        # Example constant poses (fill these with correct ones for your system)
        # Format: [x, y, z, R, P, Y]
        #self.HOME_POSE = rospy.get_param("~home_pose", [0.3, 0.0, 0.5, 0.707, 0.0, 3.14])
        self.HOME_POSE = [0.25,0.0,0.6, 3.14, 0.0, 2.355]
 
        # "Load to bin" pose: constant position with negative x value (as you said)
        #self.LOAD_BIN_POSE = rospy.get_param("~load_bin_pose", [-0.3, 0.2, 0.35, -0.707, 0.0, 3.14])
        self.LOAD_BIN_POSE = [-0.308, 0.192, 0.4,3.095, 0.087, 2.405]

        # "Grasp" positions
        self.GRASP_POSE_1 = [0.032, -0.040, 0.811, 2.104, 0.850, 1.867]
        self.GRASP_POSE_2 = [0.231, -0.061, 0.803, 1.761, 0.761, 0.188]

        # ---------- Vision latest target ----------
        self.vision_topic = rospy.get_param("~vision_topic", "/vision/latest_fruit_point")
        self.vision_timeout = float(rospy.get_param("~vision_timeout", 2.0))
        self.vision_wait_rate = float(rospy.get_param("~vision_wait_rate", 10.0))

        # Two camera scan poses, chosen from sign of arr[1]
        # Replace these with your real calibrated scan poses
        self.SCAN_POSE_POSITIVE = rospy.get_param("~scan_pose_positive", [0.032, -0.040, 0.811, 2.104, 0.850, 1.867])
        self.SCAN_POSE_NEGATIVE = rospy.get_param("~scan_pose_negative", [0.231, -0.061, 0.803, 1.761, 0.761, 0.188])

        # Offset/orientation used for grasping after vision gives fruit center
        self.VISION_GRASP_OFFSET_1 = rospy.get_param("~vision_grasp_offset_1", [-0.09, 0.0, 0.04])
        self.VISION_GRASP_OFFSET_2 = rospy.get_param("~vision_grasp_offset_2", [-0.05, 0.05, 0.01])
        self.VISION_GRASP_RPY_1 = rospy.get_param("~vision_grasp_rpy_1", [2.104, 0.850, 1.867])
        self.VISION_GRASP_RPY_2 = rospy.get_param("~vision_grasp_rpy_2", [1.761, 0.761, 0.188])

        self.latest_vision_point = None
        self.latest_vision_stamp = rospy.Time(0)
        self.vision_sub = rospy.Subscriber(self.vision_topic, PointStamped, self.vision_callback)

        # ---------- Clients ----------
        move_base_service = f"/{self.robot_ns}/move_base"
        move_arm_service  = f"/{self.robot_ns}/move_arm"
        rospy.wait_for_service(move_base_service)
        rospy.wait_for_service(move_arm_service)

        self.move_base_srv = rospy.ServiceProxy(move_base_service, MoveBase)
        self.move_arm_srv  = rospy.ServiceProxy(move_arm_service,  MoveArm)

        self.gripper_client = actionlib.SimpleActionClient(
            self.gripper_action_name,
            franka_gripper.msg.MoveAction
        )

        # ---------- Feedback Subscriber ----------
        self.feedback_sub = rospy.Subscriber(
            f"/{self.robot_ns}/action_feedback", Bool, self.feedback_callback
        )
        self.feedback = False

        # ---------- Service ----------
        # Advertised as /<robot_namespace>/action_server so the service_caller
        # can address each robot independently.
        action_server_name = f"/{self.robot_ns}/action_server"
        self.srv = rospy.Service(action_server_name, ActionServer, self.handle_action)
        rospy.loginfo(f"ActionOrchestrator service ready on {action_server_name}")

    def feedback_callback(self, msg):
        self.feedback = msg.data

    def vision_callback(self, msg):
        self.latest_vision_point = msg.point
        self.latest_vision_stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def euler_to_quat_xyzw(self, R, P, Y):

        q = tf_trans.quaternion_from_euler(R, P, Y)  # (x,y,z,w)
        return q
#        return q[1], q[2], q[3], q[0] # returns to a MoveIT preffered form

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

    def select_scan_pose(self, arr):
        # arr[1] decides which scan pose to use
        if arr[1] >= 0.0:
            return self.SCAN_POSE_POSITIVE
        return self.SCAN_POSE_NEGATIVE

    def get_latest_pose(self, min_stamp, timeout):
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        rate = rospy.Rate(self.vision_wait_rate)

        while not rospy.is_shutdown():
            if self.latest_vision_point is not None and self.latest_vision_stamp > min_stamp:
                return self.latest_vision_point, "Fresh vision target received."

            if rospy.Time.now() > deadline:
                return None, "Timed out waiting for fresh vision target."

            rate.sleep()

        return None, "ROS shutdown while waiting for vision target."

    def build_grasp_pose_from_vision(self, pt, arr):
        if arr[1] >= 0.0: 
            x = pt.x + float(self.VISION_GRASP_OFFSET_1[0])
            y = pt.y + float(self.VISION_GRASP_OFFSET_1[1])
            z = pt.z + float(self.VISION_GRASP_OFFSET_1[2])
            roll, pitch, yaw = self.VISION_GRASP_RPY_1
        else:
            x = pt.x + float(self.VISION_GRASP_OFFSET_2[0])
            y = pt.y + float(self.VISION_GRASP_OFFSET_2[1])
            z = pt.z + float(self.VISION_GRASP_OFFSET_2[2])
            roll, pitch, yaw = self.VISION_GRASP_RPY_2
        return [x, y, z, roll, pitch, yaw]

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
        #ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
        #                                    timeout=min(self.arm_step_timeout, remaining()))
        #if not ok:
        #    return False, f"GRASP: failed to go HOME: {msg}"

        # 2) Target pose
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(arr6),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to reach target: {msg}"

        # 3) Close gripper
        #ok, msg = self.move_gripper(self.grasp_width, timeout=min(self.gripper_timeout, remaining()))
        #if not ok:
        #    return False, f"GRASP: failed to close gripper: {msg}"
        rospy.sleep(3.0)

        # 4) Back home
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to return HOME: {msg}"
        rospy.loginfo("GRASP Success")
        return True, "GRASP: success."

    def action_grasp_latest_fruit(self, arr, timeout):
        """
        Sequence:
        1) choose camera scan pose from sign of arr[1]
        2) move to scan pose
        3) wait for fresh vision target
        4) build grasp pose from latest fruit point
        5) execute normal grasp action
        """
        start = time.time()

        def remaining():
            return max(0.0, timeout - (time.time() - start))

        scan_pose = self.select_scan_pose(arr)

        # clear old target so we force a fresh one after moving to scan pose
        self.latest_vision_point = None
        self.latest_vision_stamp = rospy.Time(0)

        # move to scan pose
        ok, msg = self.call_move_arm_single(
            self.pose_stamped_from_array(scan_pose),
            timeout=min(self.arm_step_timeout, remaining())
        )
        if not ok:
            return False, f"GRASP_LATEST_FRUIT: failed to reach scan pose: {msg}"

        # wait for fresh target after arriving to scan pose
        min_stamp = rospy.Time.now()
        fruit_point, info = self.get_latest_pose(min_stamp=min_stamp, timeout=min(self.vision_timeout, remaining()))
        if fruit_point is None:
            return False, f"GRASP_LATEST_FRUIT: {info}"

        grasp_pose = self.build_grasp_pose_from_vision(fruit_point, arr)
        rospy.loginfo(f"GRASP_LATEST_FRUIT: {info} grasp_pose={grasp_pose}")

        # reuse your existing grasp sequence
        return self.action_grasp(grasp_pose, timeout=remaining())

    
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

        #ok, msg = self.move_gripper(self.open_width, timeout=min(self.gripper_timeout, remaining()))
        #if not ok:
        #    return False, f"{action_id}: failed to open gripper: {msg}"

        # Optional: go home
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.HOME_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: opened but failed returning HOME: {msg}"

        return True, f"{action_id}: success."

    # -------------------------------------------------------------------------
    # Service handler
    # -------------------------------------------------------------------------

    def handle_action(self, req):
        action_id = req.action_id.strip()
        arr = list(req.input)

        if len(arr) != 6:
            return ActionServerResponse(False, "Input must be float64[6] = [x,y,z,R,P,Y]", 0.0, 0.0)

        timeout_s = float(req.timeout) if req.timeout > 0.0 else self.default_timeout

        rospy.loginfo(f"ActionServer request: action_id={action_id}, input={arr}, timeout={timeout_s}")

  
        ok = False
        msg = ""
            
        if action_id == "navigate":
            ok, msg = self.action_navigate(arr, timeout=min(self.base_step_timeout, timeout_s))

        elif action_id == "grasp_fruit":
            ok, msg = self.action_grasp_latest_fruit(arr, timeout=timeout_s)

        elif action_id == "grasp_fruit_without_vision":
            ok, msg = self.action_grasp(arr, timeout=timeout_s)

        elif action_id == "load_to_bin":
            ok, msg = self.action_load_to_bin(action_id, timeout=timeout_s)

        elif action_id == "unload":
            # exactly like NAVIGATE, but you can keep it separate if you want different logging/logic
            ok, msg = self.action_navigate(arr, timeout=min(self.base_step_timeout, timeout_s))

        elif action_id == "NOOP":
            ok, msg = True, "NOOP: success."

        else:
            ok = False
            msg = f"Unknown action_id '{action_id}'. Supported: navigate, grasp_fruit, grasp_latest_fruit, load_to_bin, unload"

            # Checks if the service performed the action in the software, 
            # and also if the current action was perfomed by the robot in the real world
            
            # Waits for the feedback from the task detection
            time.sleep(1)

            # If the feedback is not received in 1 second, the action is considered as failed
            real_ok = ok and self.feedback
        
        return ActionServerResponse(ok, msg, 0.0, 0.0)
        

if __name__ == "__main__":
    ActionManagerNode()
    rospy.spin()