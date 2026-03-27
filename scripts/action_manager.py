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

import moveit_commander
from moveit_commander import RobotCommander, PlanningSceneInterface, MoveGroupCommander

class ActionManagerNode:
    def __init__(self):
        rospy.init_node("action_manager_node", anonymous=False)

        # ---------- Params ----------
        self.world_frame = rospy.get_param("~world_frame", "fr3_link0")
        # Gripper
        self.gripper_action_name = rospy.get_param("~gripper_action", "/robot/arm/franka_gripper/grasp")
        self.gripper_speed = float(rospy.get_param("~gripper_speed", 0.03))

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
        #self.HOME_POSE = rospy.get_param("~home_pose",[0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785])
        self.HOME_POSE = rospy.get_param("~home_pose", [0.351, 0.094, 0.836, 3.14, 0.0, 3.14-0.785])        #self.HOME_POSE = [0.4, 0.0, 0.5, -3.14, 0.0, -3.14]
        # "Load to bin" pose: constant position with negative x value (as you said)
        self.LOAD_BIN_POSE = rospy.get_param("~load_bin_pose", [0.336, 0.4, 0.65, 3.14, 0, 3.14-0.785])

        # ---------- Clients ----------
        rospy.wait_for_service("/robot/move_base")
        rospy.wait_for_service("/robot/move_arm")
        
        self.move_base_srv = rospy.ServiceProxy("/robot/move_base", MoveBase)
        self.move_arm_srv  = rospy.ServiceProxy("/robot/move_arm", MoveArm)
        
        self.gripper_client = actionlib.SimpleActionClient(
            self.gripper_action_name,
            franka_gripper.msg.GraspAction
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
        return q[0],q[1],q[2],q[3]
        #return q[1], q[2], q[3], q[0] # returns to a MoveIT preffered form

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
    def pose_direct_from_array(self,arr7):
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.header.stamp = rospy.Time.now()

        ps.pose.position.x = float(arr6[0])
        ps.pose.position.y = float(arr6[1])
        ps.pose.position.z = float(arr6[2])

        ps.pose.orientation.x = float(arr7[3])
        ps.pose.orientation.y = float(arr7[4])
        ps.pose.orientation.z = float(arr7[5])
        ps.pose.orientation.w = float(arr7[6])
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
   
    def move_gripper(self, width, timeout):


        # move_gripper is now using GraspGoal instead of MoveGoal for better control.

        goal = franka_gripper.msg.GraspGoal()
        
        goal.width = float(width)
        goal.speed = float(self.gripper_speed)
        
        goal.force = 0.8
        
        goal.epsilon.inner = 0.005
        goal.epsilon.outer = 0.08 

        self.gripper_client.send_goal(goal)

        ok = self.gripper_client.wait_for_result(rospy.Duration(timeout))
        if not ok:
            self.gripper_client.cancel_goal()
            return False, f"Gripper timeout (grasp) after {timeout}s"
        
        result = self.gripper_client.get_result()
        
        if hasattr(result, "success"):
            if result.success:
                return True, "(Grasp success)."
            else:
                return False, "Gripper closed, but object not found"
                
        return True, "Grasp action complete."
    # -------------------------------------------------------------------------
    # Macro actions
    # -------------------------------------------------------------------------
    def action_navigate(self, arr6, timeout):
        x = arr6[0]
        y = arr6[1]
        theta = arr6[2]  # z used as theta
        return self.call_move_base(x, y, theta, timeout)

    def move_to_home(self,arr6timetout):
        return True, "Move to pos: Success"

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


        # 0) open gripper
        ok,msg = self.move_gripper(self.open_width, timeout = min(self.gripper_timeout, remaining()))
        if not ok:
            return False, f"GRASP: failed to open Gripper: {msg}"

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
        time.sleep(1)

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

        # 1) Load bin pose
        ok, msg = self.call_move_arm_single(self.pose_stamped_from_array(self.LOAD_BIN_POSE),
                                            timeout=min(self.arm_step_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: failed to reach load bin pose: {msg}"

        # 2) Open gripper
        ok, msg = self.move_gripper(self.open_width, timeout=min(self.gripper_timeout, remaining()))
        if not ok:
            return False, f"{action_id}: failed to open gripper: {msg}"
        time.sleep(1)

        # 3) Back home
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

        elif action_id == "grasp_fruit_direct":
            ok, msg = self.action_grasp_direct(arr, timeout=timeout_s)

        elif action_id == "load_to_bin":
            ok, msg = self.action_load_to_bin(action_id, timeout=timeout_s)

        elif action_id == "unload":
            # exactly like NAVIGATE, but you can keep it separate if you want different logging/logic
            ok, msg = self.action_navigate(arr, timeout=min(self.base_step_timeout, timeout_s))

        elif action_id == "NOOP":
            ok, msg = True, "NOOP: success."

        else:
            ok = False
            msg = f"Unknown action_id '{action_id}'. Supported: navigate, grasp_fruit, load_to_bin, unload"

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
