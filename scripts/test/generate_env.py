from pyRDDLGym.core.env import RDDLEnv

import sys

import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent


def main(episodes=1, seed=42):
    
    # create the environment
    # env = pyRDDLGym.make(domain, instance, enforce_action_constraints=True)
    env = RDDLEnv(
        domain="problem_data/fruit_collection_domain.rddl",
        instance="problem_data/fruit_collection_inst.rddl"
    )

    actions = env.action_space

    print("action space")
    print(env.action_space)
    
    print("navigate___a1")
    print(actions["navigate___a1"])


    print("model")

    print(env.model)

    with open("output.txt", "w") as f:
        f.write(f"{env.model}")
    
    # # set up a random policy
    # agent = RandomAgent(action_space=env.action_space,
    #                     num_actions=env.max_allowed_actions,
    #                     seed=seed)
    
    # agent.evaluate(env, episodes=episodes, verbose=True, render=True, seed=seed)
    
    # important when logging to save all traces
    env.close()


if __name__ == "__main__":
    main()