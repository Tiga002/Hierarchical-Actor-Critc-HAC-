import numpy as np
from environment import Environment
from utils import check_validity
from agent import Agent

def agent_and_env(FLAGS):
    """
    1. Design Parameters of the agent
    - Key hyperparameters for constructing the agent
        i. Number of levels of policies in the hierarchy
        ii. Max sequence length that each policy will specialize
        iii. Max numner of atomic actions(primitive action) allowed in an episode
        iv. Environment timesteps per atomic action
    """
    # Number of levels of policies in the hierarchy
    FLAGS.layers = 3 
    # Max sequence length that each policy will specialize
    FLAGS.time_scale = 10
    # Max numner of atomic actions(primitive action) allowed in an episode
    max_actions = FLAGS.time_scale**(FLAGS.layers-1)*6
    # Environment timesteps per atomic action
    timesteps_per_action = 15

    """
    2. Design Parameters of the environment
    - A. Original UMDP(S,A,T,G,R) must be provided
        - The S,A,T componenets can be fulfilled by providing the MuJoCo XML Model
        - Initial State space must be specidied separately
        - G can be provided by specifying the End Goal Space
        - R, which by default uses a shortest path {-1,0} reward function, can be implemented by specifying two components: 
            (i) a function that maps the state space to the end goal space and 
            (ii) the end goal achievement thresholds for each dimensions of the end goal.
    - B.  In order to convert the original UMDP into a hierarchy of k UMDPs, the followings must be provided
        - The subgoal action space, A_i, for all higher-level UMDPs i > 0
        - R_i for levels that try to achieve goals in the subgoal space
            - As in the original UMDP, R_i can be implemented by providing two components:
                - (i) a function that maps the state space to the subgoal space and 
                - (ii) the subgoal achievement thresholds. 
    - C. Designer should also provide 
        - subgoal and end goal visualization functions in order to show video of training.  
        - These can be updated in "display_subgoal" and "display_end_goal" methods in the "environment.py" file.
    """
    model_name = "locobot_reach.xml"

    # Provide initial state space consisting of 
    #   - the ranges for all joint angles and velocities.  
    #   - if we have 4 joints
    #       - size ([8,2]) 4pos,4vel
    # In the UR5 Reacher task, we use a random initial shoulder position and use fixed values for the remainder.  
    # Initial joint velocities are set to 0.
    initial_joint_pos = np.array([-1.22487753e-05, 6.71766300e-03, 7.30874027e-03, 3.80183559e-03])
    # reshape the vector in shape of [4,1]
    initial_joint_pos = np.reshape(initial_joint_pos, (len(initial_joint_pos),1))
    # Matrix of the initial joint position range [4,2]
    initial_joint_ranges = np.concatenate((initial_joint_pos, initial_joint_pos),1)
    # Matrix of the initial state space, including joint positions and velocities
    initial_state_space = np.concatenate((initial_joint_ranges,np.zeros((len(initial_joint_ranges),2))),0)

    # Provide end goal space.  
    # The code supports two types of end goal spaces.
    # If user would like to train on a larger end goal space.  
    # If user needs to make additional customizations to the end goals, the "get_next_goal" method in "environment.py" can be updated.

    # For LocoBot Reach task, end goal will be the desired end effector position (x,y,z)
    
