#!/usr/bin/env python3

import rospy

from prost_ros.srv import StartPlanning, SubmitObservation

# Package Imports

from sim_interface import SimInterface

class ServiceCaller:

    def __init__(self, sim_id):
        rospy.init_node("PROST service caller")

        rospy.wait_for_service('/prost_bridge/start_planning')
        rospy.wait_for_service('/prost_bridge/submit_observation')

        self.start_planning = rospy.ServiceProxy('/prost_bridge/start_planning', StartPlanning)
        self.submit_obs = rospy.ServiceProxy('/prost_bridge/submit_observation', SubmitObservation)

        with open("domain.rddl", 'r') as f:
            self.domain = f.read()
        with open("instance.rddl", 'r') as f:
            self.instance = f.read()

        rospy.loginfo("Sending the Planning request")
        self.resp = self.start_planning(self.domain, self.instance, 60)
        if not self.resp.success:
            rospy.logerr("PROST server failed")

        self.sim = SimInterface(sim_id) 

        # Initial action state
        
        self.count = 0
        self.obs = self.sim.get_obs()
        rospy.loginfo("Sending init state . . .")
        self.act_resp = self.submit_obs(self.obs, 0.0) # Init reward is 0.0
        
        self.HORIZON_LENGTH = 40
        self.TERMINATION_STRING = "ROUND_END"
        
    def run(self):

        rospy.loginfo(self.act_resp.action_name)
        while self.act_resp.action_name != self.TERMINATION_STRING and count < self.HORIZON_LENGTH:

            self.sim.step(self.act_resp.action_name, self.act_resp.action_params, self.count)
            obs = self.sim.get_obs()
            
            # Reward: 0 if goal reached, -1 otherwise (as per domain file)
            # reward = [sum_{?x : xpos, ?y : ypos} -(GOAL(?x,?y) ^ ~robot-at(?x,?y))]; 
            # So if NOT at goal, reward is -1. If at goal, reward is 0 (assuming goal is unique).
            
            reward = -1.0, # TODO: Reward should be modified  and implemented.
            if self.sim.terminate:
                reward = 0.0
                rospy.loginfo("Sim finished")
                break
                
            act_resp = self.submit_obs(obs, reward)
            count += 1
            rospy.loginfo(act_resp.action_name)


if __name__ == "__main__":
    robot = "Robotnik"

    sc = ServiceCaller(robot=robot)
    sc.run()