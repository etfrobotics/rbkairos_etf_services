import rospy

import rospy
import random
from prost_bridge.msg import KeyValue


class RobotnikSim:
    """
    Reward-consistent simulation for your fruit_collection RDDL.

    Implements:
      - step(action_name, action_params, t) -> returns reward
      - get_obs() -> KeyValue[]
      - self.terminate
      - self.last_reward

    The reward computation mirrors your RDDL reward expression.
    """

    TERMINATION_STRING = "ROUND_END"

    def __init__(self, robot_id=None):
        # ------------------------------
        # Non-fluents (from instance)
        # ------------------------------
        self.unload_stations = {"unload1", "unload2"}

        # adjacency + distances for the zigzag layout
        self.adjacent = set()
        self.distance = {}

        self._init_graph()

        # reachable_from(location, aisle_position)
        self.reachable_from = {}
        self._init_reachability()

        # fruit weights
        self.fruit_weight = self._init_weights()

        # fruit ripe flags
        self.fruit_ripe = self._init_ripeness()

        # constants
        self.GRASP_SUCCESS_PROB = 0.9
        self.MAX_CAPACITY = 100.0
        self.CAPACITY_THRESHOLD = 90.0

        # ------------------------------
        # State fluents (init-state)
        # ------------------------------
        self.robot_pos = "a1"
        self.position_visited = {f"a{i}": False for i in range(1, 22)}
        self.position_visited["unload1"] = False
        self.position_visited["unload2"] = False

        # fruit states: fruit_at, fruit_collected, fruit_in_bin, fruits_unloaded
        self.fruit_state = {}
        for i in range(1, 43):
            l = f"l{i}"
            self.fruit_state[l] = {
                "fruit_at": True,
                "fruit_collected": False,
                "fruit_in_bin": False,
                "fruits_unloaded": False,
            }

        # episode control
        self.terminate = False
        self.last_reward = 0.0
        self.t = 0

    # ======================================================================
    # Public API (SimInterface compatibility)
    # ======================================================================

    def step(self, action_name, action_params, t):
        """
        Apply action, update state, compute reward (RDDL reward mirror).
        Returns reward and stores it in self.last_reward.
        """
        self.t = t

        if action_name == self.TERMINATION_STRING:
            self.terminate = True
            self.last_reward = 0.0
            return self.last_reward

        # Snapshot "pre" state (needed for reward terms with primes)
        pre = self._snapshot_state()

        # Execute action (update state)
        self._apply_action(action_name, action_params)

        # Snapshot "post" state (prime state)
        post = self._snapshot_state()

        # Compute reward exactly according to your domain expression
        r = self._compute_reward(pre, post, action_name, action_params)

        self.last_reward = r
        self._check_termination()  # optional termination condition

        return r

    def get_obs(self):
        """
        Return KeyValue[] representing state fluents.
        """
        obs = []

        # robot_at(a)
        for a in self.position_visited.keys():
            obs.append(self._kv(f"robot_at({a})", "true" if a == self.robot_pos else "false"))

        # position_visited(a)
        for a, v in self.position_visited.items():
            obs.append(self._kv(f"position_visited({a})", "true" if v else "false"))

        # fruits
        for l, st in self.fruit_state.items():
            obs.append(self._kv(f"fruit_at({l})", "true" if st["fruit_at"] else "false"))
            obs.append(self._kv(f"fruit_collected({l})", "true" if st["fruit_collected"] else "false"))
            obs.append(self._kv(f"fruit_in_bin({l})", "true" if st["fruit_in_bin"] else "false"))
            obs.append(self._kv(f"fruits_unloaded({l})", "true" if st["fruits_unloaded"] else "false"))

        return obs

    # ======================================================================
    # Action application (deterministic sim version of your CPFs)
    # ======================================================================

    def _apply_action(self, action_name, params):
        if action_name == "navigate":
            if not params:
                return
            target = params[0]
            self._do_navigate(target)

        elif action_name == "grasp_fruit":
            if not params:
                return
            loc = params[0]
            self._do_grasp(loc)

        elif action_name == "load_to_bin":
            if not params:
                return
            loc = params[0]
            self._do_load(loc)

        elif action_name == "unload":
            self._do_unload()

        elif action_name == "" or action_name.lower() == "noop":
            pass

        else:
            rospy.logwarn(f"[RobotnikSim] Unknown action: {action_name} {params}")

    def _do_navigate(self, target):
        """
        Navigate only if adjacent; otherwise ignore (planner should not send invalid moves).
        """
        if target == self.robot_pos:
            return

        if (self.robot_pos, target) not in self.adjacent:
            # Invalid move; ignore the move (planner shouldn't do this if preconditions correct)
            rospy.logwarn(f"[RobotnikSim] Invalid navigate: {self.robot_pos} -> {target} (not adjacent)")
            return

        self.robot_pos = target

    def _do_grasp(self, loc):
        if loc not in self.fruit_state:
            return

        st = self.fruit_state[loc]

        # Must be reachable, present, ripe, not already in bin
        if not st["fruit_at"]:
            return
        if not self.fruit_ripe.get(loc, True):
            return
        if st["fruit_in_bin"]:
            return

        if not self._is_reachable(loc, self.robot_pos):
            return

        # Capacity check (matches precondition spirit)
        if self._bin_weight() + self.fruit_weight.get(loc, 1.0) > self.MAX_CAPACITY:
            return

        # stochastic grasp success
        success = (random.random() <= self.GRASP_SUCCESS_PROB)
        if success:
            st["fruit_at"] = False
            st["fruit_collected"] = True

    def _do_load(self, loc):
        if loc not in self.fruit_state:
            return
        st = self.fruit_state[loc]

        if not self._is_reachable(loc, self.robot_pos):
            return

        if st["fruit_collected"] and (not st["fruit_at"]) and (not st["fruit_in_bin"]):
            # Capacity check
            if self._bin_weight() + self.fruit_weight.get(loc, 1.0) <= self.MAX_CAPACITY:
                st["fruit_in_bin"] = True
                # position_visited' CPF: when robot_at(a) and load_to_bin at reachable_from(l,a)
                self.position_visited[self.robot_pos] = True

    def _do_unload(self):
        """
        If at unload station and anything in bin: unload all.
        """
        if self.robot_pos not in self.unload_stations:
            return
        if not self._exists_in_bin():
            return

        for l, st in self.fruit_state.items():
            if st["fruit_in_bin"]:
                st["fruit_in_bin"] = False
                st["fruits_unloaded"] = True
                # In your CPF fruit_collected' becomes false when unloading.
                # BUT your reward uses fruit_collected'(l) in the end condition.
                # In your domain: fruit_collected' resets to false at unload.
                # We'll mirror that:
                st["fruit_collected"] = False

    # ======================================================================
    # Reward computation (mirrors your RDDL reward expression)
    # ======================================================================

    def _compute_reward(self, pre, post, action_name, action_params):
        """
        Mirrors your domain reward:

        + 30  per newly collected ripe fruit
        + 10  per newly loaded ripe fruit
        + 0.2 per fruit in bin (post)
        + 200 for being at unload station in post AND finishing criteria
        + 500 for navigating to unload station with capacity >= threshold (pre state)
        - 0.3 * distance for navigation step (if valid adjacent move)
        - 0.05 per grasp
        - 0.05 per load
        - 0.05 unload
        - 100 penalty for being non-unload with capacity >= threshold and not navigating to unload
        - 12 penalty for navigating away (adjacent) while leaving ripe pending work at old position and not full
        """

        r = 0.0

        # ------------------------------
        # Helper lambdas
        # ------------------------------
        def is_true(d, key):
            return bool(d.get(key, False))

        def bin_weight_from(state):
            return sum(
                (1.0 if state["fruit_in_bin"][l] else 0.0) * self.fruit_weight.get(l, 1.0)
                for l in self.fruit_state.keys()
            )

        # ------------------------------
        # +30 newly collected ripe fruits
        # fruit_collected'(l) ^ ~fruit_collected(l) ^ fruit_ripe(l)
        # ------------------------------
        for l in self.fruit_state.keys():
            if post["fruit_collected"][l] and (not pre["fruit_collected"][l]) and self.fruit_ripe.get(l, True):
                r += 30.0

        # ------------------------------
        # +10 newly loaded ripe fruits
        # fruit_in_bin'(l) ^ ~fruit_in_bin(l) ^ fruit_ripe(l)
        # ------------------------------
        for l in self.fruit_state.keys():
            if post["fruit_in_bin"][l] and (not pre["fruit_in_bin"][l]) and self.fruit_ripe.get(l, True):
                r += 10.0

        # ------------------------------
        # +0.2 per fruit in bin (post)
        # ------------------------------
        r += 0.2 * sum(1.0 for l in self.fruit_state.keys() if post["fruit_in_bin"][l])

        # ------------------------------
        # +200 end bonus condition:
        # robot_at'(a) ^ unload_station(a) ^
        #   (forall l: ~fruit_at'(l) | ~fruit_ripe(l)) ^
        #   (forall l: (fruit_collected'(l) ^ ripe(l)) => fruits_unloaded'(l))
        #
        # In your CPFs, fruit_collected'(l) becomes false when unloading,
        # so the implication is trivially satisfied after unload. We mirror that.
        # ------------------------------
        post_robot_pos = post["robot_pos"]
        if post_robot_pos in self.unload_stations:
            no_ripe_left_on_shelf = True
            for l in self.fruit_state.keys():
                if self.fruit_ripe.get(l, True) and post["fruit_at"][l]:
                    no_ripe_left_on_shelf = False
                    break

            collected_implies_unloaded = True
            for l in self.fruit_state.keys():
                if post["fruit_collected"][l] and self.fruit_ripe.get(l, True):
                    if not post["fruits_unloaded"][l]:
                        collected_implies_unloaded = False
                        break

            if no_ripe_left_on_shelf and collected_implies_unloaded:
                r += 200.0

        # ------------------------------
        # +500 for navigating to unload station with capacity >= threshold
        # navigate(a) ^ unload_station(a) ^ (sum fruit_in_bin * fruit_weight >= threshold)
        # NOTE: This uses PRE-STATE fruit_in_bin (unprimed) in your reward.
        # ------------------------------
        if action_name == "navigate" and action_params:
            target = action_params[0]
            if target in self.unload_stations:
                if bin_weight_from(pre) >= self.CAPACITY_THRESHOLD:
                    r += 500.0

        # ------------------------------
        # -0.3 * distance for navigation (only when robot_at(a1) & navigate(a2) & adjacent(a1,a2))
        # ------------------------------
        if action_name == "navigate" and action_params:
            a1 = pre["robot_pos"]
            a2 = action_params[0]
            if (a1, a2) in self.adjacent:
                dist = self.distance.get((a1, a2), 1.0)
                r -= 0.3 * dist

        # ------------------------------
        # -0.05 action costs
        # ------------------------------
        if action_name == "grasp_fruit":
            r -= 0.05
        if action_name == "load_to_bin":
            r -= 0.05
        if action_name == "unload":
            r -= 0.05

        # ------------------------------
        # -100 penalty:
        # robot_at(a) ^ ~unload_station(a) ^ (bin_weight >= threshold) ^
        #   ~exists a2 [navigate(a2) ^ unload_station(a2)]
        # ------------------------------
        if pre["robot_pos"] not in self.unload_stations:
            if bin_weight_from(pre) >= self.CAPACITY_THRESHOLD:
                going_to_unload = (action_name == "navigate" and action_params and action_params[0] in self.unload_stations)
                if not going_to_unload:
                    r -= 100.0

        # ------------------------------
        # -12 penalty for leaving position when there is ripe work remaining at old position:
        # (robot_at(a1) ^ navigate(a2) ^ adjacent(a1,a2) ^ not unload stations both ^
        #  bin_weight < threshold ^
        #  exists l reachable_from(l,a1) ^ ripe(l) ^ ((fruit_at & ~collected) | (collected & ~in_bin)))
        # ------------------------------
        if action_name == "navigate" and action_params:
            a1 = pre["robot_pos"]
            a2 = action_params[0]

            if (a1, a2) in self.adjacent:
                if (a1 not in self.unload_stations) and (a2 not in self.unload_stations):
                    if bin_weight_from(pre) < self.CAPACITY_THRESHOLD:
                        # check "exists ripe pending work at a1"
                        pending = False
                        for l in self.fruit_state.keys():
                            if not self._is_reachable(l, a1):
                                continue
                            if not self.fruit_ripe.get(l, True):
                                continue

                            fruit_at = pre["fruit_at"][l]
                            collected = pre["fruit_collected"][l]
                            in_bin = pre["fruit_in_bin"][l]

                            if (fruit_at and (not collected)) or (collected and (not in_bin)):
                                pending = True
                                break

                        if pending:
                            r -= 12.0

        return float(r)

    # ======================================================================
    # Termination condition (optional)
    # ======================================================================

    def _check_termination(self):
        """
        Optional: end when all ripe fruits have been unloaded.
        This is NOT in RDDL directly, but practical for simulation.
        """
        for l in self.fruit_state.keys():
            if self.fruit_ripe.get(l, True) and (not self.fruit_state[l]["fruits_unloaded"]):
                return
        self.terminate = True

    # ======================================================================
    # State snapshots for reward calculation
    # ======================================================================

    def _snapshot_state(self):
        """
        Returns dict containing:
          robot_pos
          fruit_at[l], fruit_collected[l], fruit_in_bin[l], fruits_unloaded[l]
        """
        snap = {
            "robot_pos": self.robot_pos,
            "fruit_at": {},
            "fruit_collected": {},
            "fruit_in_bin": {},
            "fruits_unloaded": {},
        }
        for l, st in self.fruit_state.items():
            snap["fruit_at"][l] = bool(st["fruit_at"])
            snap["fruit_collected"][l] = bool(st["fruit_collected"])
            snap["fruit_in_bin"][l] = bool(st["fruit_in_bin"])
            snap["fruits_unloaded"][l] = bool(st["fruits_unloaded"])
        return snap

    # ======================================================================
    # Non-fluent helpers
    # ======================================================================

    def _is_reachable(self, loc, aisle_pos):
        return self.reachable_from.get((loc, aisle_pos), False)

    def _exists_in_bin(self):
        return any(st["fruit_in_bin"] for st in self.fruit_state.values())

    def _bin_weight(self):
        return sum(
            (1.0 if self.fruit_state[l]["fruit_in_bin"] else 0.0) * self.fruit_weight.get(l, 1.0)
            for l in self.fruit_state.keys()
        )

    def _kv(self, k, v):
        kv = KeyValue()
        kv.key = k
        kv.value = v
        return kv

    # ======================================================================
    # Instance-derived nonfluents (hardcoded from your provided file)
    # ======================================================================

    def _add_edge(self, u, v, d):
        self.adjacent.add((u, v))
        self.distance[(u, v)] = float(d)

    def _init_graph(self):
        # Aisle 1 forward: a1-a2-a3-a4-a5-a6-a7
        for i in range(1, 7):
            self._add_edge(f"a{i}", f"a{i+1}", 1.0)
            self._add_edge(f"a{i+1}", f"a{i}", 1.0)

        # Aisle 2 backward: a14-a13-a12-a11-a10-a9-a8
        for i in range(14, 8, -1):
            self._add_edge(f"a{i}", f"a{i-1}", 1.0)
            self._add_edge(f"a{i-1}", f"a{i}", 1.0)

        # Aisle 3 forward: a15-a16-a17-a18-a19-a20-a21
        for i in range(15, 21):
            self._add_edge(f"a{i}", f"a{i+1}", 1.0)
            self._add_edge(f"a{i+1}", f"a{i}", 1.0)

        # Zigzag transitions
        self._add_edge("a7", "a14", 5.0)
        self._add_edge("a14", "a7", 5.0)
        self._add_edge("a8", "a15", 5.0)
        self._add_edge("a15", "a8", 5.0)

        # Unload stations
        self._add_edge("unload1", "a1", 2.0)
        self._add_edge("a1", "unload1", 2.0)

        self._add_edge("a21", "unload2", 2.0)
        self._add_edge("unload2", "a21", 2.0)

    def _init_reachability(self):
        # Exactly as in your instance file:
        # l1,l2 reachable from a1; l3,l4 from a2; ... etc.

        mapping = {
            "a1": ["l1", "l2"],
            "a2": ["l3", "l4"],
            "a3": ["l5", "l6"],
            "a4": ["l7", "l8"],
            "a5": ["l9", "l10"],
            "a6": ["l11", "l12"],
            "a7": ["l13", "l14"],

            "a8": ["l15", "l16"],
            "a9": ["l17", "l18"],
            "a10": ["l19", "l20"],
            "a11": ["l21", "l22"],
            "a12": ["l23", "l24"],
            "a13": ["l25", "l26"],
            "a14": ["l27", "l28"],

            "a15": ["l29", "l30"],
            "a16": ["l31", "l32"],
            "a17": ["l33", "l34"],
            "a18": ["l35", "l36"],
            "a19": ["l37", "l38"],
            "a20": ["l39", "l40"],
            "a21": ["l41", "l42"],
        }

        for a, locs in mapping.items():
            for l in locs:
                self.reachable_from[(l, a)] = True

    def _init_weights(self):
        # Taken from your instance file
        w = {
            "l1": 0.9, "l2": 1.8, "l3": 1.3, "l4": 1.7, "l5": 1.9, "l6": 1.7,
            "l7": 0.6, "l8": 1.6, "l9": 1.0, "l10": 1.3, "l11": 0.7, "l12": 1.8,
            "l13": 1.4, "l14": 1.6, "l15": 0.6, "l16": 1.4, "l17": 1.6, "l18": 1.5,
            "l19": 1.9, "l20": 0.6, "l21": 1.3, "l22": 1.0, "l23": 0.8, "l24": 1.3,
            "l25": 0.7, "l26": 1.3, "l27": 1.3, "l28": 1.0, "l29": 1.7, "l30": 1.8,
            "l31": 0.6, "l32": 0.6, "l33": 1.9, "l34": 1.5, "l35": 1.5, "l36": 1.6,
            "l37": 0.8, "l38": 0.9, "l39": 1.7, "l40": 0.9, "l41": 1.4, "l42": 1.0
        }
        return w

    def _init_ripeness(self):
        # Taken from your instance file
        ripe = {f"l{i}": True for i in range(1, 43)}
        for bad in ["l8", "l18", "l19", "l26", "l30"]:
            ripe[bad] = False
        return ripe



class SimInterface:
    
    def robot_sim(self, robot_id):

        if robot_id == "Robotnik":
            return RobotnikSim()
        else:
            rospy.loginfo("ERROR: Unsupported sim")
            return 