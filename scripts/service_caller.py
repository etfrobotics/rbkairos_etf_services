#!/usr/bin/env python3

import rospy

from prost_ros.srv import StartPlanning, SubmitObservation


# Usage: rosrun prost_ros prost_bridge.py _prost_path:=/home/ruzamladji/catkin_ws/src/prost_ros/prost/prost.py

# Package Imports

from evaluator import FruitHarvestingRewardEvaluator
import numpy as np

class ServiceCaller:
    def __init__(self, robot_id, domain_file, instance_file):
        rospy.init_node("prost_service_caller")

        rospy.wait_for_service('/prost_bridge/start_planning')
        rospy.wait_for_service('/prost_bridge/submit_observation')

        rospy.wait_for_service('/action_server')

        self.start_planning = rospy.ServiceProxy('/prost_bridge/start_planning', StartPlanning)
        self.submit_obs = rospy.ServiceProxy('/prost_bridge/submit_observation', SubmitObservation)

        self.perform_action = rospy.ServiceProxy('/action_server', ActionServer)

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

        self.reward = 0.0

        # Load the actions.json action to experiment data
        with open("actions.json", "r", encoding="utf-8") as f:
            self.ACTION_DATA = json.load(f)

        
    def run(self):

        while True:

            rospy.loginfo("Sending Observations and rewards to the planner . . .")
            
            action_to_take = self.submit_obs(self.obs, self.reward)
            #TODO: CHECK THE OUTPUT OF THE PLANNER
            graph_idx = action_to_take.action_params
            
            action_to_take_data = self.ACTION_DATA[action_to_take.action_name][np.where(graph_idx == True)]
            
            rospy.loginfo("Performing action . . .")
            success = self.perform_action(action_to_take.action_name, action_to_take_data)

            observed_action = self.evaluator.create_observation_template()
            observed_action[action_to_take.action_name][np.where(graph_idx == True)] = success

            next_obs = self.evaluator.step(self.obs, observed_action)
    
            reward = self.evaluator.evaluate_reward(self.obs, next_obs)


            if self.evaluator.env.model._state_fluents['goal_reached']:
                reward = 0.0
                rospy.loginfo("Sim finished")
                break

            # Update the current observation and the previous reward for the planner at the start of the next iteration
            self.obs = next_obs
            self.reward = reward



if __name__ == "__main__":

    #TODO: Automate this
    # domain_file = "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_domain.rddl"
    # instance_file = "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_inst.rddl"

    domain_file = "/home/etf-robotics/catkin_ws/src/rbkairos_etf_services/problem_data/domain.rddl"
    instance_file = "/home/etf-robotics/catkin_ws/src/rbkairos_etf_services/problem_data/instance.rddl"
    
    sc = ServiceCaller("Robotnik", domain_file, instance_file)
    sc.run()