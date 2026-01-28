location : {l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25, l26, l27, l28, l29, l30, l31, l32, l33, l34, l35, l36, l37, l38, l39, l40, l41, l42}; 
aisle_position : {a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, unload1, unload2};

These are the two objects the planner is working with. The robot can navigate to 21 spots, each with 2 fruit locations. 
Critical part of defining the actions.json file is to add the two unload1, and unload2 locations as last! So the indicies of unload1, and unload2 would 
in this case 22 and 23. 

The actions_params will be empty in the case of unload action, because there isn't a location for the robot to unload. 

locations are tied to the grasp_fruit actions. 
aisle_position is tied to the navigate action. 

load_to_bin
unload = > These are the actions which have a constant value for the postions of the gripper

NOOP = > this is an idle action as is handled in the service caller.