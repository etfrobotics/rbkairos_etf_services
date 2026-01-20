import numpy as np
from typing import Dict, List, Any

from pyRDDLGym.core.env import RDDLEnv
import pyRDDLGym


class FruitHarvestingRewardEvaluator:
    """
    Evaluates the reward function for fruit harvesting based on real-world observations.
    """
    
    def __init__(self, env_model):
        """
        Initialize the reward evaluator with the environment model.
        
        Args:
            env_model: The RDDLLiftedModel object from pyRDDLGym (env.model)
        """
        self.model = env_model
        
        # Extract key parameters from non-fluents
        self.max_capacity = env_model._non_fluents['MAX_CAPACITY']
        self.capacity_threshold = env_model._non_fluents['CAPACITY_THRESHOLD']
        self.fruit_weights = env_model._non_fluents['fruit_weight']
        self.fruit_ripe = env_model._non_fluents['fruit_ripe']
        
        # Object mappings
        self.location_to_idx = {loc: idx for loc, idx in env_model._object_to_index.items() 
                                if env_model._object_to_type[loc] == 'location'}
        self.position_to_idx = {pos: idx for pos, idx in env_model._object_to_index.items() 
                                if env_model._object_to_type[pos] == 'aisle_position'}
        
        self.num_locations = len(self.location_to_idx)
        self.num_positions = len(self.position_to_idx)
        
        # Store discount factor
        self.discount = env_model._discount
        
    def create_observation_template(self) -> Dict[str, Any]:
        """
        Create a template observation dictionary that can be filled with real-world data.
        
        Returns:
            Dictionary with all required observation fields initialized to default values
        """
        obs = {
            # State fluents
            'fruit_at': np.zeros(self.num_locations, dtype=bool),
            'fruit_collected': np.zeros(self.num_locations, dtype=bool),
            'fruit_in_bin': np.zeros(self.num_locations, dtype=bool),
            'fruits_unloaded': np.zeros(self.num_locations, dtype=bool),
            'robot_at': np.zeros(self.num_positions, dtype=bool),
            'position_visited': np.zeros(self.num_positions, dtype=bool),
            
            # Action fluents (for the current timestep)
            'grasp_fruit': np.zeros(self.num_locations, dtype=bool),
            'load_to_bin': np.zeros(self.num_locations, dtype=bool),
            'navigate': np.zeros(self.num_positions, dtype=bool),
            'unload': False,
        }
        return obs
    
    def calculate_bin_load(self, fruit_in_bin: np.ndarray) -> float:
        """
        Calculate the current bin load based on fruits in the bin.
        
        Args:
            fruit_in_bin: Boolean array indicating which fruits are in the bin
            
        Returns:
            Total weight in the bin
        """
        bin_load = 0.0
        for loc_idx, in_bin in enumerate(fruit_in_bin):
            if in_bin:
                bin_load += self.fruit_weights[loc_idx]
        return bin_load
    
    def evaluate_reward(self, observation: Dict[str, Any]) -> float:
        """
        Evaluate the reward function based on the current observation.
        
        The reward function typically includes:
        1. Positive reward for unloading ripe fruits
        2. Penalty for unloading unripe fruits
        3. Penalty for bin overload
        
        Args:
            observation: Dictionary containing current state and action observations
            
        Returns:
            Computed reward value
        """
        reward = 0.0
        
        # Extract relevant observations
        fruits_unloaded = observation['fruits_unloaded']
        fruit_in_bin = observation['fruit_in_bin']
        unload_action = observation['unload']
        
        # Calculate bin load
        bin_load = self.calculate_bin_load(fruit_in_bin)
        
        # Reward for unloading fruits
        if unload_action:
            for loc_idx, unloaded in enumerate(fruits_unloaded):
                if unloaded:
                    if self.fruit_ripe[loc_idx]:
                        # Positive reward for ripe fruits
                        reward += 10.0
                    else:
                        # Penalty for unripe fruits
                        reward -= 20.0
        
        # Penalty for bin overload
        if bin_load > self.capacity_threshold:
            overload = bin_load - self.capacity_threshold
            reward -= overload * 5.0  # Penalty proportional to overload
        
        return reward
    
    def evaluate_trajectory(self, trajectory: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a complete trajectory of observations.
        
        Args:
            trajectory: List of observation dictionaries, one per timestep
            
        Returns:
            Dictionary containing:
                - rewards: List of rewards at each timestep
                - total_reward: Sum of all rewards
                - discounted_reward: Discounted sum of rewards
        """
        rewards = []
        
        for obs in trajectory:
            reward = self.evaluate_reward(obs)
            rewards.append(reward)
        
        total_reward = sum(rewards)
        discounted_reward = sum([r * (self.discount ** t) for t, r in enumerate(rewards)])
        
        return {
            'rewards': rewards,
            'total_reward': total_reward,
            'discounted_reward': discounted_reward,
            'num_timesteps': len(rewards)
        }
    
    def get_observation_summary(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a human-readable summary of the current observation.
        
        Args:
            observation: Observation dictionary
            
        Returns:
            Dictionary with summarized information
        """
        bin_load = self.calculate_bin_load(observation['fruit_in_bin'])
        
        summary = {
            'fruits_at_locations': int(np.sum(observation['fruit_at'])),
            'fruits_collected': int(np.sum(observation['fruit_collected'])),
            'fruits_in_bin': int(np.sum(observation['fruit_in_bin'])),
            'fruits_unloaded': int(np.sum(observation['fruits_unloaded'])),
            'bin_load': bin_load,
            'bin_overloaded': bin_load > self.capacity_threshold,
            'robot_position': np.where(observation['robot_at'])[0].tolist(),
            'current_action': self._get_current_action(observation)
        }
        return summary
    
    def _get_current_action(self, observation: Dict[str, Any]) -> str:
        """Helper to identify the current action being taken."""
        if observation['unload']:
            return 'unload'
        elif np.any(observation['navigate']):
            pos_idx = np.where(observation['navigate'])[0][0]
            return f'navigate_to_position_{pos_idx}'
        elif np.any(observation['grasp_fruit']):
            loc_idx = np.where(observation['grasp_fruit'])[0][0]
            return f'grasp_fruit_at_location_{loc_idx}'
        elif np.any(observation['load_to_bin']):
            loc_idx = np.where(observation['load_to_bin'])[0][0]
            return f'load_to_bin_from_location_{loc_idx}'
        else:
            return 'no_action'


if __name__ == "__main__":

    env = RDDLEnv(
        domain="problem_data/fruit_collection_domain.rddl",
        instance="problem_data/fruit_collection_inst.rddl"
    )

    
    print("Fruit Harvesting Reward Evaluator")
    print("=" * 50)
    print("\nUsage Example:")
   
    # Initialize the evaluator
    evaluator = FruitHarvestingRewardEvaluator(env.model)
    
    # Create observation template
    obs = evaluator.create_observation_template()
    
    # Fill with real-world data at timestep t
    obs['fruit_at'][0] = True  # Fruit at location 0
    obs['robot_at'][5] = True  # Robot at position 5
    obs['grasp_fruit'][0] = True  # Grasping fruit at location 0
    
    # Evaluate reward for this timestep
    reward = evaluator.evaluate_reward(obs)
    print(f"Reward at timestep: {reward}")
    
    # Get observation summary
    summary = evaluator.get_observation_summary(obs)
    print("Observation Summary:", summary)
    
    # Evaluate complete trajectory
    # trajectory = [obs1, obs2, obs3, ...]  # List of observations
    # results = evaluator.evaluate_trajectory(trajectory)
    # print(f"Total Reward: {results['total_reward']}")
    # print(f"Discounted Reward: {results['discounted_reward']}")
