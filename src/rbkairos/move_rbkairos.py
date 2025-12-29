#!/usr/bin/env python3
import rospy
import threading
import time

from rbkairos_etf_services.srv import MoveBase, MoveBaseResponse

from robot_simple_command_manager_msgs.msg import CommandString
from robot_simple_command_manager_msgs.msg import CommandManagerStatus


class MoveBaseServiceNode:
    def __init__(self):
        rospy.init_node("move_base_service_node", anonymous=False)

        self.cmd_topic = rospy.get_param("~command_topic", "/robot/command_manager/command")
        self.status_topic = rospy.get_param("~status_topic", "/robot/command_manager/status")

        self._status_lock = threading.Lock()
        self._last_status_code = None
        self._last_status_cmd = None
        self._status_event = threading.Event()

        self.cmd_pub = rospy.Publisher(self.cmd_topic, CommandString, queue_size=1)
        self.status_sub = rospy.Subscriber(self.status_topic, CommandManagerStatus, self._status_cb)

        self.srv = rospy.Service("move_base", MoveBase, self.handle_move_base)
        rospy.loginfo("MoveBase service ready (service name: /move_base).")

    def _status_cb(self, msg: CommandManagerStatus):
        with self._status_lock:
            # Common fields: msg.code, msg.command (per your comments)
            self._last_status_code = getattr(msg, "code", None)
            self._last_status_cmd = getattr(msg, "command", None)
            self._status_event.set()

    def handle_move_base(self, req):
        timeout_s = float(req.timeout) if req.timeout > 0.0 else 30.0

        cmd_str = f"GOTO {req.x} {req.y} {req.theta}"

        # Clear previous status signal
        self._status_event.clear()

        msg = CommandString()
        msg.command = cmd_str
        self.cmd_pub.publish(msg)

        start = time.time()
        saw_active = False

        while (time.time() - start) < timeout_s and not rospy.is_shutdown():
            # Wait for any status update
            remaining = timeout_s - (time.time() - start)
            if remaining <= 0.0:
                break

            got = self._status_event.wait(timeout=min(0.5, remaining))
            if not got:
                continue

            with self._status_lock:
                code = self._last_status_code
                scmd = self._last_status_cmd
                self._status_event.clear()

            # If your status doesn't include command string, we just react to codes.
            # If it DOES include it, we can optionally require it matches our cmd.
            if scmd is not None and isinstance(scmd, str) and (scmd.strip() != cmd_str.strip()):
                continue

            # Typical values in many setups: "ACTIVE", "SUCCEEDED", "FAILED"
            if code == "ACTIVE":
                saw_active = True
            elif code == "SUCCEEDED":
                return MoveBaseResponse(True, "Base movement SUCCEEDED.")
            elif code == "FAILED":
                return MoveBaseResponse(False, "Base movement FAILED.")
            else:
                # Unknown code; keep waiting
                pass

        if saw_active:
            return MoveBaseResponse(False, "Timeout waiting for base movement to finish (was ACTIVE).")
        return MoveBaseResponse(False, "Timeout waiting for base movement status.")

if __name__ == "__main__":
    MoveBaseServiceNode()
    rospy.spin()
