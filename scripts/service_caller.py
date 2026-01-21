#!/usr/bin/env python3

import rospy

from prost_ros.srv import StartPlanning, SubmitObservation


# Usage: rosrun prost_ros prost_bridge.py _prost_path:=/home/ruzamladji/catkin_ws/src/prost_ros/prost/prost.py

# Package Imports

from evaluator import FruitHarvestingRewardEvaluator

class ServiceCaller:

    def __init__(self, robot_id, domain_file, instance_file):
        rospy.init_node("prost_service_caller")

        rospy.wait_for_service('/prost_bridge/start_planning')
        rospy.wait_for_service('/prost_bridge/submit_observation')

        self.start_planning = rospy.ServiceProxy('/prost_bridge/start_planning', StartPlanning)
        self.submit_obs = rospy.ServiceProxy('/prost_bridge/submit_observation', SubmitObservation)

        with open(domain_file, 'r') as f:
            self.domain = f.read()
        with open(instance_file, 'r') as f:
            self.instance = f.read()

        rospy.loginfo("Sending the Planning request")
        self.resp = self.start_planning(self.domain, self.instance, 60)
        if not self.resp.success:
            rospy.logerr("PROST server failed")

        self.evaluator = FruitHarvestingRewardEvaluator()

        # Initial action state
        self.obs = self.evaluator.create_observation_template()

        self.obs['fruit_at'] = np.array(self.evaluator.env.model._state_fluents['fruit_at'])
        self.obs['fruit_collected'] = np.array(self.evaluator.env.model._state_fluents['fruit_collected'])
        self.obs['fruit_in_bin'] = np.array(self.evaluator.env.model._state_fluents['fruit_in_bin'])
        self.obs['fruits_unloaded'] = np.array(self.evaluator.env.model._state_fluents['fruits_unloaded'])
        self.obs['robot_at'] = np.array(self.evaluator.env.model._state_fluents['robot_at'])
        self.obs['position_visited'] = np.array(self.evaluator.env.model._state_fluents['position_visited'])
        
        self.count = 0
        rospy.loginfo("Sending init state . . .")
        self.act_resp = self.submit_obs(self.obs, 0.0) # Init reward is 0.0
        
        self.HORIZON_LENGTH = 150
        
        
    def run(self):

        while self.count < self.HORIZON_LENGTH:

            self.evaluator.step(self.act_resp.action_name, self.act_resp.action_params, self.count)
            
            # Reward: 0 if goal reached, -1 otherwise (as per domain file)
            # reward = [sum_{?x : xpos, ?y : ypos} -(GOAL(?x,?y) ^ ~robot-at(?x,?y))]; 
            # So if NOT at goal, reward is -1. If at goal, reward is 0 (assuming goal is unique).
            
            reward = -1.0 
            if self.evaluator.env.model._state_fluents['goal_reached']:
                reward = 0.0
                rospy.loginfo("Sim finished")
                break
                
            self.act_resp = self.submit_obs(self.obs, reward)
            self.count += 1

if __name__ == "__main__":

    #TODO: Automate this
    # domain_file = "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_domain.rddl"
    # instance_file = "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_inst.rddl"

    domain_file = "/home/etf-robotics/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_domain.rddl"
    instance_file = "/home/etf-robotics/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_inst.rddl"
    
    sc = ServiceCaller("Robotnik", domain_file, instance_file)
    sc.run()