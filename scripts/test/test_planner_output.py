#!/usr/bin/env python3

import rospy

from prost_ros.srv import StartPlanning, SubmitObservation


# Usage: rosrun prost_ros prost_bridge.py _prost_path:=/home/ruzamladji/catkin_ws/src/prost_ros/prost/prost.py

# Package Imports


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
    
    def run(self):
        print(self.resp)
        rospy.spin()




if __name__ == "__main__":

    #TODO: Automate this
    # domain_file = "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_domain.rddl"
    # instance_file = "/home/ruzamladji/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_inst.rddl"

    domain_file = "/home/etf-robotics/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_domain.rddl"
    instance_file = "/home/etf-robotics/catkin_ws/src/rbkairos_etf_services/problem_data/fruit_collection_inst.rddl"
    
    sc = ServiceCaller("Robotnik", domain_file, instance_file)
    sc.run()