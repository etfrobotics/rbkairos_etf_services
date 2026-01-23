import json
import random

from evaluator import FruitHarvestingRewardEvaluator


NUM_LOCATIONS = 42
ALLOWED_ACTIONS = ["navigate", "grasp_fruit", "load_to_bin", "unload"]


def generate_actions_json(num_locations: int, seed: int = 0):
    rng = random.Random(seed)

    load_to_bin_const = [0.5, -0.25, 1.0, 0.0, 0.0, 0.75]

    def rand6():
        # 6 random floats (adjust range as you like)
        return [rng.uniform(-1.0, 1.0) for _ in range(6)]


    data = {
        "navigate":    [rand6() for _ in range(num_locations)],
        "grasp_fruit": [rand6() for _ in range(num_locations)],
        # same constant array repeated num_locations times:
        "load_to_bin": [load_to_bin_const],
        "unload":      [rand6() for _ in range(num_locations)],
    }
    return data

# --- usage ---

data = generate_actions_json(num_locations=NUM_LOCATIONS, seed=42)

with open("actions.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Wrote actions.json with num_locations =", num_locations)
