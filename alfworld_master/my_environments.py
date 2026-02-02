import os
import json
import random

from tqdm import tqdm
from termcolor import colored

import textworld
import textworld.agents
import textworld.gym

from alfworld.agents.utils.misc import Demangler, add_task_to_grammar
from alfworld.agents.expert import HandCodedTWAgent, HandCodedAgentTimeout


TASK_TYPES = {1: "pick_and_place_simple",
              2: "look_at_obj_in_light",
              3: "pick_clean_then_place_in_recep",
              4: "pick_heat_then_place_in_recep",
              5: "pick_cool_then_place_in_recep",
              6: "pick_two_obj_and_place"}


class AlfredDemangler(textworld.core.Wrapper):

    def __init__(self, *args, shuffle=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.shuffle = shuffle

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)

        demangler = Demangler(game_infos=self._entity_infos, shuffle=self.shuffle)
        for info in self._entity_infos.values():
            info.name = demangler.demangle_alfred_name(info.id)


import os, uuid
import textworld

class AlfredInfos(textworld.core.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gamefile = None
        self._uid = None

    def load(self, *args, **kwargs):
        super().load(*args, **kwargs)
        self._gamefile = args[0]

        self._uid = f"{os.getpid()}:{uuid.uuid4().hex}"

    def _attach(self, state):
        state["extra.gamefile"] = self._gamefile
        state["extra.uid"] = self._uid
        return state

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        return self._attach(state)

    def step(self, command):
        state, reward, done = super().step(command)
        state = self._attach(state)
        return state, reward, done



# Enum for the supported types of AlfredExpert.
class AlfredExpertType:
    HANDCODED = "handcoded"
    PLANNER = "planner"


class AlfredExpert(textworld.core.Wrapper):

    def __init__(self, env=None, expert_type=AlfredExpertType.HANDCODED):
        super().__init__(env=env)

        self.expert_type = expert_type
        self.prev_command = ""

        if expert_type not in (AlfredExpertType.HANDCODED, AlfredExpertType.PLANNER):
            msg = "Unknown type of AlfredExpert: {}.\nExpecting either '{}' or '{}'."
            msg = msg.format(expert_type, AlfredExpertType.HANDCODED, AlfredExpertType.PLANNER)
            raise ValueError(msg)

    def _gather_infos(self):
        # Compute expert plan.
        if self.expert_type == AlfredExpertType.HANDCODED:
            self.state["extra.expert_plan"] = ["look"]
            try:
                # initialization
                if not self.prev_command:
                    
                    self._handcoded_expert.observe(self.state.get("feedback", ""))
                else:
                    try:
                        handcoded_expert_next_action = self._handcoded_expert.act(
                            self.state, 0, self.state.get("won", False), self.prev_command
                        )
                    except IndexError as e:
                        
                        handcoded_expert_next_action = "look"
                        self.state["extra.expert_error"] = f"Handcoded act IndexError: {repr(e)}"
                    except Exception as e:
                        
                        handcoded_expert_next_action = "look"
                        self.state["extra.expert_error"] = f"Handcoded act Error: {repr(e)}"

                    if handcoded_expert_next_action in self.state.get("admissible_commands", []):
                        self.state["extra.expert_plan"] = [handcoded_expert_next_action]

            except HandCodedAgentTimeout:
                raise Exception("Timeout")

        elif self.expert_type == AlfredExpertType.PLANNER:
            self.state["extra.expert_plan"] = self.state["policy_commands"]
        else:
            raise NotImplementedError("Unknown type of AlfredExpert: {}.".format(self.expert_type))


    def load(self, gamefile):
        super().load(gamefile)
        self.gamefile = gamefile
        self.request_infos.policy_commands = self.request_infos.policy_commands or (self.expert_type == AlfredExpertType.PLANNER)
        self.request_infos.facts = self.request_infos.facts or (self.expert_type == AlfredExpertType.HANDCODED)
        self._handcoded_expert = HandCodedTWAgent(max_steps=200)

    def step(self, command):
        self.state, reward, done = super().step(command)
        self.prev_command = str(command)
        self._gather_infos()
        return self.state, reward, done

    def reset(self):
        self.state = super().reset()
        self._handcoded_expert.reset(self.gamefile)
        self.prev_command = ""
        self._gather_infos()
        return self.state


class AlfredTWEnv(object):
    '''
    Interface for Textworld Env
    '''

    def __init__(self, config, train_eval="train"):
        print("Initializing AlfredTWEnv...")
        self.config = config
        self.train_eval = train_eval

        if config["env"]["goal_desc_human_anns_prob"] > 0:
            msg = ("Warning! Changing `goal_desc_human_anns_prob` should be done with"
                   " the script `alfworld-generate`. Ignoring it and loading games as they are.")
            print(colored(msg, "yellow"))

        self.collect_game_files()
    

    def _build_catalog(self):
        self.catalog = []
        for i, p in enumerate(self.game_files):
            trial_path = os.path.dirname(p)
            task_path  = os.path.dirname(trial_path)
            trial = os.path.basename(trial_path)
            task  = os.path.basename(task_path)
            self.catalog.append({
                "idx": i,
                "path": p,
                "trial": trial,
                "task": task,
                "trial_path": trial_path,
                "task_path": task_path,
            })
    


    def subset_and_repeat(self, indices=None, trials=None, tasks=None, repeat: int = 1, shuffle=False):
        assert repeat >= 1
        if not hasattr(self, "catalog") or not self.catalog:
            self._build_catalog()

        picked_paths = []

        if indices:
            for i in indices:
                if i < 0 or i >= len(self.catalog):
                    raise IndexError(f"Index {i} out of range (0..{len(self.catalog)-1}).")
                picked_paths.append(self.catalog[i]["path"])

        if trials:
            trial_set = set(trials)
            for row in self.catalog:
                if row["trial"] in trial_set:
                    picked_paths.append(row["path"])

        if tasks:
            task_set = set(tasks)
            for row in self.catalog:
                if row["task"] in task_set:
                    picked_paths.append(row["path"])

        seen, unique_paths = set(), []
        for p in picked_paths:
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)

        repeated = [p for p in unique_paths for _ in range(repeat)]

        if shuffle:
            random.shuffle(repeated)

        self.game_files = repeated
        self.num_games = len(self.game_files)
        self._build_catalog()
        print(f"[subset_and_repeat] Unique games: {len(unique_paths)}, repeat={repeat} -> total {self.num_games}.")



    def collect_game_files(self, verbose=False):
        def log(info):
            if verbose:
                print(info)

        self.game_files = []

        if self.train_eval == "train":
            data_path = os.path.expandvars(self.config['dataset']['data_path'])
        elif self.train_eval == "temp_train":
            data_path = os.path.expandvars(self.config['dataset']['temp_train_data_path'])
        elif self.train_eval == "syn_train":
            data_path = os.path.expandvars(self.config['dataset']['syn_train_data_path'])
        elif self.train_eval == "eval_in_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_id_data_path'])
        elif self.train_eval == "eval_out_of_distribution":
            data_path = os.path.expandvars(self.config['dataset']['eval_ood_data_path'])

        log("Collecting solvable games...")

        # get task types
        assert len(self.config['env']['task_types']) > 0
        task_types = []
        for tt_id in self.config['env']['task_types']:
            if tt_id in TASK_TYPES:
                task_types.append(TASK_TYPES[tt_id])

        count = 0
        for root, dirs, files in tqdm(list(os.walk(data_path, topdown=False))):
            if 'traj_data.json' in files:
                count += 1

                # Filenames
                json_path = os.path.join(root, 'traj_data.json')
                game_file_path = os.path.join(root, "game.tw-pddl")

                if 'movable' in root or 'Sliced' in root:
                    log("Movable & slice trajs not supported %s" % (root))
                    continue

                # Get goal description
                with open(json_path, 'r') as f:
                    traj_data = json.load(f)

                # Check for any task_type constraints
                if not traj_data['task_type'] in task_types:
                    log("Skipping task type")
                    continue

                # Check if a game file exists
                if not os.path.exists(game_file_path):
                    log(f"Skipping missing game! {game_file_path}")
                    continue

                with open(game_file_path, 'r') as f:
                    gamedata = json.load(f)

                # Check if previously checked if solvable
                if 'solvable' not in gamedata:
                    print(f"-> Skipping missing solvable key! {game_file_path}")
                    continue

                if not gamedata['solvable']:
                    log("Skipping known %s, unsolvable game!" % game_file_path)
                    continue

                # Add to game file list
                self.game_files.append(game_file_path)

        print(f"Overall we have {len(self.game_files)} games in split={self.train_eval}")
        self.num_games = len(self.game_files)

        if self.train_eval == "train":
            num_train_games = self.config['dataset']['num_train_games'] if self.config['dataset']['num_train_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_train_games]
            self.num_games = len(self.game_files)
            print("Training with %d games" % (len(self.game_files)))
        else:
            num_eval_games = self.config['dataset']['num_eval_games'] if self.config['dataset']['num_eval_games'] > 0 else len(self.game_files)
            self.game_files = self.game_files[:num_eval_games]
            self.num_games = len(self.game_files)
            print("Evaluating with %d games" % (len(self.game_files)))

    def get_game_logic(self):
        self.game_logic = {
            "pddl_domain": open(os.path.expandvars(self.config['logic']['domain'])).read(),
            "grammar": open(os.path.expandvars(self.config['logic']['grammar'])).read()
        }

    # use expert to check the game is solvable
    def is_solvable(self, env, game_file_path,
                    random_perturb=True, random_start=10, random_prob_after_state=0.15):
        done = False
        steps = 0
        trajectory = []
        try:
            env.load(game_file_path)
            game_state = env.reset()
            if env.expert_type == AlfredExpertType.PLANNER:
                return game_state["extra.expert_plan"]

            while not done:
                expert_action = game_state['extra.expert_plan'][0]
                random_action = random.choice(game_state.admissible_commands)

                command = expert_action
                if random_perturb:
                    if steps <= random_start or random.random() < random_prob_after_state:
                        command = random_action

                game_state, _, done = env.step(command)
                trajectory.append(command)
                steps += 1
        except Exception as e:
            print("Unsolvable: %s (%s)" % (str(e), game_file_path))
            return None

        return trajectory

    def init_env(self, batch_size):
        domain_randomization = self.config["env"]["domain_randomization"]
        if self.train_eval != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        # Register a new Gym environment.
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "uid"])
        expert_type = self.config["env"]["expert_type"]
        training_method = self.config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = self.config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = self.config["dagger"]["training"]["max_nb_steps_per_episode"]

            expert_plan = True if self.train_eval == "train" else False
            if expert_plan:
                #wrappers.append(AlfredExpert(expert_type))
                wrappers.append(AlfredExpert(expert_type=expert_type))
                request_infos.extras.append("expert_plan")

        else:
            raise NotImplementedError

        env_id = textworld.gym.register_games(self.game_files, request_infos,
                                              batch_size=batch_size,
                                              asynchronous=True,
                                              max_episode_steps=max_nb_steps_per_episode,
                                              wrappers=wrappers)
        # Launch Gym environment.
        env = textworld.gym.make(env_id)
        return env