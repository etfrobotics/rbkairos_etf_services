import numpy as np
from typing import Dict, List, Any

from pyRDDLGym.core.env import RDDLEnv
import pyRDDLGym


class FruitHarvestingRewardEvaluator:
    """
    Evaluates the reward function for fruit harvesting based on real-world observations.
    """
    
    def __init__(self, domain_file, instance_file):
        """
        Initialize the reward evaluator with the environment model.
        
        Args:
            env_model: The RDDLLiftedModel object from pyRDDLGym (env.model)
        """

        self.env = RDDLEnv(
            domain=domain_file,
            instance=instance_file
        )

        env_model = self.env.model
        
        # Extract key parameters from non-fluents
        self.max_capacity = env_model._non_fluents['MAX_CAPACITY']
        self.capacity_threshold = env_model._non_fluents['CAPACITY_THRESHOLD']
        self.fruit_weights = np.array(env_model._non_fluents['fruit_weight'])
        self.fruit_ripe = np.array(env_model._non_fluents['fruit_ripe'])
        
        # Extract adjacency and distance matrices
        # These are flattened, need to reshape
        num_positions = len([k for k in env_model._object_to_type.keys() 
                            if env_model._object_to_type[k] == 'aisle_position'])
        num_locations = len([k for k in env_model._object_to_type.keys() 
                            if env_model._object_to_type[k] == 'location'])
        
        self.num_locations = num_locations
        self.num_positions = num_positions
        
        # Reshape adjacent and distance to (num_positions, num_positions)
        adjacent_flat = np.array(env_model._non_fluents['adjacent'])
        self.adjacent = adjacent_flat.reshape(num_positions, num_positions)
        
        distance_flat = np.array(env_model._non_fluents['distance'])
        self.distance = distance_flat.reshape(num_positions, num_positions)
        
        # Reshape reachable_from to (num_locations, num_positions)
        reachable_flat = np.array(env_model._non_fluents['reachable_from'])
        self.reachable_from = reachable_flat.reshape(num_locations, num_positions)
        
        # Unload station positions
        self.unload_station = np.array(env_model._non_fluents['unload_station'])
        
        # Object mappings
        self.location_to_idx = {loc: idx for loc, idx in env_model._object_to_index.items() 
                                if env_model._object_to_type[loc] == 'location'}
        self.position_to_idx = {pos: idx for pos, idx in env_model._object_to_index.items() 
                                if env_model._object_to_type[pos] == 'aisle_position'}
        
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
    
    def bin_load(self, fruit_in_bin: np.ndarray) -> float:
        """Calculate current bin load."""
        return float(np.sum(fruit_in_bin.astype(float) * self.fruit_weights))

    def evaluate_reward(self, obs: Dict[str, Any], next_obs: Dict[str, Any]) -> float:
        """
        obs contains unprimed state at time t and action fluents for t (as in your template).
        next_obs contains primed state at time t+1.
        """

        # --- Unprimed (t) ---
        fruit_collected = np.asarray(obs["fruit_collected"], dtype=bool)
        fruit_in_bin = np.asarray(obs["fruit_in_bin"], dtype=bool)
        fruit_at = np.asarray(obs["fruit_at"], dtype=bool)
        robot_at = np.asarray(obs["robot_at"], dtype=bool)

        # Actions at t
        grasp_fruit = np.asarray(obs["grasp_fruit"], dtype=bool)
        load_to_bin = np.asarray(obs["load_to_bin"], dtype=bool)
        navigate = np.asarray(obs["navigate"], dtype=bool)
        unload_action = bool(obs["unload"])

        # --- Primed (t+1) ---
        fruit_collected_p = np.asarray(next_obs["fruit_collected"], dtype=bool)
        fruit_in_bin_p = np.asarray(next_obs["fruit_in_bin"], dtype=bool)
        fruit_at_p = np.asarray(next_obs["fruit_at"], dtype=bool)
        robot_at_p = np.asarray(next_obs["robot_at"], dtype=bool)
        fruits_unloaded_p = np.asarray(next_obs["fruits_unloaded"], dtype=bool)

        reward = 0.0

        # ---------------------------------------------------------------------
        # + (sum_l [fruit_collected'(l) ^ ~fruit_collected(l) ^ fruit_ripe(l)]) * 30
        newly_collected_ripe = fruit_collected_p & (~fruit_collected) & self.fruit_ripe
        reward += float(np.sum(newly_collected_ripe)) * 30.0

        # + (sum_l [fruit_in_bin'(l) ^ ~fruit_in_bin(l) ^ fruit_ripe(l)]) * 10
        newly_in_bin_ripe = fruit_in_bin_p & (~fruit_in_bin) & self.fruit_ripe
        reward += float(np.sum(newly_in_bin_ripe)) * 10.0

        # + (sum_l [fruit_in_bin'(l)]) * 0.2
        reward += float(np.sum(fruit_in_bin_p)) * 0.2

        # ---------------------------------------------------------------------
        # + (sum_a [robot_at'(a) ^ unload_station(a) ^
        #          (forall_l [~fruit_at'(l) | ~fruit_ripe(l)]) ^
        #          (forall_l [(fruit_collected'(l) ^ fruit_ripe(l)) => fruits_unloaded'(l)])]) * 200
        #
        # forall_l [~fruit_at'(l) | ~fruit_ripe(l)]  <=>  no ripe fruit is at any location in next state
        no_ripe_fruit_at_next = np.all((~fruit_at_p) | (~self.fruit_ripe))

        # forall_l [(fruit_collected'(l) ^ fruit_ripe(l)) => fruits_unloaded'(l)]
        # implication P=>Q is (~P) | Q
        P = fruit_collected_p & self.fruit_ripe
        collected_ripe_implies_unloaded = np.all((~P) | fruits_unloaded_p)

        cond_unload_bonus = no_ripe_fruit_at_next & collected_ripe_implies_unloaded
        if cond_unload_bonus:
            # count positions a where robot_at'(a) & unload_station(a)
            reward += float(np.sum(robot_at_p & self.unload_station)) * 200.0

        # ---------------------------------------------------------------------
        # + (sum_a [navigate(a) ^ unload_station(a) ^
        #          (bin_load >= CAPACITY_THRESHOLD)]) * 500
        bin_load_now = self.bin_load(fruit_in_bin)
        if bin_load_now >= self.capacity_threshold:
            reward += float(np.sum(navigate & self.unload_station)) * 500.0

        # ---------------------------------------------------------------------
        # - (sum_{a1,a2} [robot_at(a1) ^ navigate(a2) ^ adjacent(a1,a2)] * distance(a1,a2)) * 0.3
        # robot_at and navigate are one-hot in typical RDDL encodings; we'll compute generally.
        # Build outer product mask of chosen move(s)
        move_mask = np.outer(robot_at.astype(bool), navigate.astype(bool)) & self.adjacent
        move_cost = float(np.sum(self.distance * move_mask))
        reward -= move_cost * 0.3

        # ---------------------------------------------------------------------
        # - (sum_l grasp_fruit(l)) * 0.05
        reward -= float(np.sum(grasp_fruit)) * 0.05

        # - (sum_l load_to_bin(l)) * 0.05
        reward -= float(np.sum(load_to_bin)) * 0.05

        # - unload * 0.05
        reward -= (0.05 if unload_action else 0.0)

        # ---------------------------------------------------------------------
        # - (sum_a [robot_at(a) ^ ~unload_station(a) ^
        #          (bin_load >= CAPACITY_THRESHOLD) ^
        #          ~exists_{a2} [navigate(a2) ^ unload_station(a2)]]) * 100
        if bin_load_now >= self.capacity_threshold:
            chose_unload_nav = bool(np.any(navigate & self.unload_station))
            if not chose_unload_nav:
                reward -= float(np.sum(robot_at & (~self.unload_station))) * 100.0

        # ---------------------------------------------------------------------
        # - (sum_{a1,a2} [robot_at(a1) ^ navigate(a2) ^ adjacent(a1,a2) ^
        #                ~unload_station(a1) ^ ~unload_station(a2) ^
        #                (bin_load < CAPACITY_THRESHOLD) ^
        #                exists_l [reachable_from(l,a1) ^ fruit_ripe(l) ^
        #                          ((fruit_at(l) ^ ~fruit_collected(l)) |
        #                           (fruit_collected(l) ^ ~fruit_in_bin(l))) ]]) * 12
        if bin_load_now < self.capacity_threshold:
            # exists_l ... at current a1
            # "work exists": ripe fruit either still at location and not collected,
            # or collected but not yet in bin.
            work_l = self.fruit_ripe & (
                (fruit_at & (~fruit_collected)) |
                (fruit_collected & (~fruit_in_bin))
            )  # shape (L,)

            # For each a1, does there exist an l reachable s.t. work_l[l] is True?
            # reachable_from: (L, A). We want (A,) for a1:
            exists_work_at_a1 = (self.reachable_from.T @ work_l.astype(int)) > 0  # (A,)

            # Now apply the big conjunction over (a1,a2)
            mask_a1a2 = (
                np.outer(robot_at, navigate) &
                self.adjacent &
                np.outer(~self.unload_station, ~self.unload_station) &
                np.outer(exists_work_at_a1, np.ones_like(exists_work_at_a1, dtype=bool))
            )

            reward -= float(np.sum(mask_a1a2)) * 12.0

        return float(reward)
    
    def step(self, obs: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate a simple deterministic step (for demonstration purposes).
        In reality, you would apply your actual transition dynamics.
        
        Args:
            obs: Current observation
            action: Action to take (grasp_fruit, load_to_bin, navigate, unload)
            
        Returns:
            next_obs: Resulting next observation
        """
        next_obs = self.create_observation_template()
        
        # Copy current state to next state (will be modified based on actions)
        next_obs['fruit_at'] = obs['fruit_at'].copy()
        next_obs['fruit_collected'] = obs['fruit_collected'].copy()
        next_obs['fruit_in_bin'] = obs['fruit_in_bin'].copy()
        next_obs['fruits_unloaded'] = obs['fruits_unloaded'].copy()
        next_obs['robot_at'] = obs['robot_at'].copy()
        next_obs['position_visited'] = obs['position_visited'].copy()
        
        # Apply grasp_fruit action
        if np.any(action['grasp_fruit']):
            for loc_idx in np.where(action['grasp_fruit'])[0]:
                if obs['fruit_at'][loc_idx] and not obs['fruit_collected'][loc_idx]:
                    # Fruit is grasped (with success probability, here deterministic)
                    next_obs['fruit_collected'][loc_idx] = True
                    next_obs['fruit_at'][loc_idx] = False
        
        # Apply load_to_bin action
        if np.any(action['load_to_bin']):
            for loc_idx in np.where(action['load_to_bin'])[0]:
                if obs['fruit_collected'][loc_idx] and not obs['fruit_in_bin'][loc_idx]:
                    next_obs['fruit_in_bin'][loc_idx] = True
        
        # Apply navigate action
        if np.any(action['navigate']):
            target_pos = np.where(action['navigate'])[0][0]
            current_pos = np.where(obs['robot_at'])[0][0]
            
            # Check if adjacent
            if self.adjacent[current_pos, target_pos]:
                next_obs['robot_at'] = np.zeros(self.num_positions, dtype=bool)
                next_obs['robot_at'][target_pos] = True
                next_obs['position_visited'][target_pos] = True
        
        # Apply unload action
        if action['unload']:
            current_pos = np.where(obs['robot_at'])[0][0]
            if self.unload_station[current_pos]:
                # Unload all fruits in bin
                next_obs['fruits_unloaded'] = obs['fruit_in_bin'].copy()
                next_obs['fruit_in_bin'] = np.zeros(self.num_locations, dtype=bool)
        
        return next_obs

if __name__ == "__main__":
    
    # ev = get_evaluator("domain.rddl", "instance.rddl")
    print("evaluator.py")
    