#!/usr/bin/env python3

import rospy

from prost_ros.srv import StartPlanning, SubmitObservation
from prost_ros.msg import KeyValue
from rbkairos_etf_services.srv import ActionServer


# Usage: rosrun prost_ros prost_bridge.py _prost_path:=/home/ruzamladji/catkin_ws/src/prost_ros/prost/prost.py

# Package Imports
from rbkairos_etf_services.evaluator import FruitHarvestingRewardEvaluator
import numpy as np
import rospkg
import os
import json

def get_problem_data_paths(
    package_name="rbkairos_etf_services",
    domain_name="domain.rddl",
    instance_name="instance.rddl",
    actions_name="actions.json",
    ):

    rp = rospkg.RosPack()
    pkg_path = rp.get_path(package_name)  
    problem_data = os.path.join(pkg_path, "problem_data")
    return (
            os.path.join(problem_data, domain_name),
            os.path.join(problem_data, instance_name),
            os.path.join(problem_data, actions_name),
        )

class ServiceCaller:
    def __init__(self, robot_id, domain_file, instance_file, actions_file):
        rospy.init_node("prost_service_caller")

        rospy.wait_for_service('/prost_bridge/start_planning')
        rospy.wait_for_service('/prost_bridge/submit_observation')

        # rospy.wait_for_service('/action_server') # TODO: Uncomment this when the action server is ready

        self.start_planning = rospy.ServiceProxy('/prost_bridge/start_planning', StartPlanning)
        self.submit_obs = rospy.ServiceProxy('/prost_bridge/submit_observation', SubmitObservation)

        # self.perform_action = rospy.ServiceProxy('/action_server', ActionServer)

        with open(domain_file, 'r') as f:
            self.domain = f.read()
        with open(instance_file, 'r') as f:
            self.instance = f.read()

        rospy.loginfo("Sending the Planning request")
        self.resp = self.start_planning(self.domain, self.instance, 60)
        if not self.resp.success:
            rospy.logerr("PROST server failed")

        self.evaluator = FruitHarvestingRewardEvaluator(domain_file, instance_file)

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
        with open(actions_file, "r", encoding="utf-8") as f:
            self.ACTION_DATA = json.load(f)

        # Create mappings from indices to object names (and vice-versa) based on the environment model
        # This assumes pyRDDLGym state vectors are ordered by the object indices.
        env_model = self.evaluator.env.model
        
        # Helper to get objects sorted by their internal index
        def get_sorted_objects(type_name):
            objs = [o for o, t in env_model._object_to_type.items() if t == type_name]
            return sorted(objs, key=lambda x: env_model._object_to_index[x])

        self.positions = get_sorted_objects('aisle_position')
        self.locations = get_sorted_objects('location')
        
        # Reverse mappings for action interpretation
        self.pos_to_idx = {name: i for i, name in enumerate(self.positions)}
        self.loc_to_idx = {name: i for i, name in enumerate(self.locations)}

    def run(self):

        while True:

            rospy.loginfo("Sending Observations and rewards to the planner . . .")

            obs_to_submit = []
            
            # Helper to add true fluents to observation
            def add_fluents(name, values, obj_list):
                 for i, val in enumerate(values):
                     if val:
                         obs_to_submit.append(KeyValue(f"{name}({obj_list[i]})", "true"))

            add_fluents("robot_at", self.obs['robot_at'], self.positions)
            add_fluents("fruit_at", self.obs['fruit_at'], self.locations)
            add_fluents("fruit_collected", self.obs['fruit_collected'], self.locations)
            add_fluents("fruit_in_bin", self.obs['fruit_in_bin'], self.locations)
            add_fluents("fruits_unloaded", self.obs['fruits_unloaded'], self.locations)
            add_fluents("position_visited", self.obs['position_visited'], self.positions) 
            
            action_to_take = self.submit_obs(obs_to_submit, self.reward)

            print(action_to_take)
            #TODO: CHECK THE OUTPUT OF THE PLANNER

            rospy.loginfo("Performing action . . .")
            
            # Should receive (success, message) or similar, but perform_action returns ActionServerResponse (bool success, string message)
            response = self.perform_action(action_name, action_data, 0.0)
            success = response.success
            rospy.loginfo(f"Action finished with success: {success} | {response.message}")

            observed_action = self.evaluator.create_observation_template()
            
            # Update the observed action for the evaluator
            if action_name == "unload":
                observed_action[action_name] = success
            else:
                 for idx in updated_indices:
                     # Update the boolean array at the specific index
                     observed_action[action_name][idx] = success

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

    domain_file, instance_file, actions_file = get_problem_data_paths()


    print(domain_file)
    print(instance_file)

    # domain_file = "/problem_data/domain.rddl"
    # instance_file = "/problem_data/instance.rddl"
    
    sc = ServiceCaller("Robotnik", domain_file, instance_file, actions_file)
    sc.run()