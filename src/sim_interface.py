import rospy

class RobotnikSim:

    def __init__(self):

        rospy.wait_for_service('') # Movebas
        rospy.wait_for_service('') # Moveit PickPlace and Grasp
        rospy.wait_for_service('') # Storage

        ## Add additional services and or topics, for the get_obs

        ## Map locations

        self.fruit_locations = []

        ## Available fruit 

        self.picked_fruit = []

        ## Current Grasp

        self.grasp_success = False

    def step(self):
        pass

    def get_obs(self):
        pass


class SimInterface:

    def __init__(self, strategy):
        self.strategy = strategy
    
    def robot_sim(self):

        if self.strategy == "Robotnik":
            return RobotnikSim()
        else:
            rospy.loginfo("ERROR: Unsupported sim")
            return 