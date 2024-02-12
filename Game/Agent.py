import threading
import uuid
from pathlib import Path
from typing import List

import ray
from ray.rllib.algorithms.ppo import PPOConfig
import asyncio
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn, TensorType
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.torch_utils import FLOAT_MIN
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override

from Game import Map, Agent, Village
from Game.Buildings import *

class Agent:

    def __init__(self,name=""):
        self.name = name

    def choose_action(self,actions: list) -> str:
        return actions[0][0]


class MapEnv(gym.Env):
    def __init__(self, env_config):
        self.worker_id = ray.runtime_context.get_runtime_context().get_worker_id()
        self.max_steps = env_config["max_steps"] or 1000
        self.speed = env_config["speed"] or 0.1
        self.agent_wait = env_config["agent_wait"] or 0.
        self.valid_actions = ['headquarters','timber', 'clay', 'iron','first_church','rally_point','farm', 'warehouse','hiding_place',
                       'watchtower', 'wall','market','smithy', 'academy',
                      'church', 'workshop', 'stable', 'barracks', 'idle']
        self.action_record = np.zeros(len(self.valid_actions))

        # Since we are just going to consider building objects,
        # for now we are going to take a discrete approach on all building scenarios
        max_avail_actions = len(self.valid_actions)
        self.action_space = spaces.Discrete(max_avail_actions)
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(max_avail_actions,), dtype=np.int8),
            "observations": spaces.Box(low=0, high=np.inf, shape=(3+3+max_avail_actions-1,))
        })
        """"observations": spaces.Dict({
            "resources": spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.int32),  # wood,iron,clay
            "building_levels": spaces.Box(low=0, high=30, shape=(max_avail_actions - 1,), dtype=np.int32),
            # all_buildings -idle
            # Maybe resources production rate?
        })"""

    def reset(self, seed=None, options=None, **kwargs):
        """
        Game starting conditions:
            Loot: (500w,500c,400i)
            Production: (5w,5c,5i)
            Buildings: [headquarters,first_church, rally_point,farm,warehouse,hiding_place]
            Available to build: [timber,clay,iron,]
            Coordinates: Random
            Points: 36p
        :param seed:
        :param options:
        :param kwargs:
        :return:
        """
        self.loop = None
        self.village_task = None
        self._initialize_async_loop()
        #Storing past actions
        self.action_record = np.zeros(len(self.valid_actions))
        self.past_actions = []
        self.point_at_time = []

        self.current_step = 0
        self.agent = Agent()
        self.new_village = self._create_starting_village()
        print("STARTING")
        #Running
        self.start_game()
        # self.map = Map(self.agent, [new_village])
        observation,future_rewards = self._generate_observations_and_rewards()
        print("ST OBSERVATION",observation)
        assert (observation["action_mask"][4]==0 and observation["action_mask"][5]==0),\
            f"{self.new_village.all_buildings[4]} and {self.new_village.all_buildings[5]} can be updated"
        return observation,{}

    def step(self, action):
        """
        :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
          if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.

        :param action:
        :return:
        """
        #Storing past actions
        self.past_actions.append(action)
        self.point_at_time.append(self.new_village.compute_points())
        self.action_record[action]+=1
        # 1. Process the action
        self.current_step += 1
        action_name = self.valid_actions[action]
        upgrades, rewards = self.new_village.get_available_upgrades()
        assert action_name=='idle' or action_name in rewards,\
            f"Invalid action {action}: {action_name}"
        self.action_record[action]+=1
        print(f"ACTION {self.current_step}: {action_name}")
        status, reward = self.new_village.perform_action(action_name)  # self.valid_actions[action]
        print("Status",status,"Reward",reward)
        asyncio.run(asyncio.sleep(self.agent_wait))
        # 2. Update the Observation and 3. Calculate the Reward
        observation,future_rewards = self._generate_observations_and_rewards()
        # 4. Check Termination Conditions
        terminated = self.new_village.check_max_village()
        # 5. Check Truncation - assuming a maximum number of steps for simplicity
        truncated = self.current_step >= self.max_steps  # Example condition
        print("OBSERVATION",observation)
        assert (observation["action_mask"][4]==0 and observation["action_mask"][5]==0),\
            f"{self.new_village.all_buildings[4]} and {self.new_village.all_buildings[5]} can be updated"
        if terminated or truncated:
            np.save(f"./stats/Action_record_{self.worker_id[:4]}",self.action_record)
        return observation, reward * 0.3, terminated, truncated, {}

    # def close(self):
    def start_game(self):
        if self.loop is None or self.loop.is_closed():
            self.initialize_async_loop()

            # Run village tasks asynchronously without waiting for them to complete
        if self.village_task:
            self.village_task.cancel()  # Cancel previous tasks if needed
        asyncio.run_coroutine_threadsafe(self.new_village.run(self.speed), self.loop)
    def _initialize_async_loop(self):
        """Initialize the asyncio event loop in a separate thread."""
        def start_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        t = threading.Thread(target=start_loop)
        t.start()
    def _generate_observations_and_rewards(self):
        max_avail_actions = len(self.valid_actions)
        #CARE REWARDS IS A DICTIONARY
        upgrades, rewards = self.new_village.get_available_upgrades()
        action_mask = np.array(list(upgrades.values())+[1], dtype=np.int8)
        resources = np.array(list(self.new_village.loot.values()))
        production = np.array(list(self.new_village.production.values()))
        building_levels = np.array(list(self.new_village.get_buildings().values()))
        assert ((max_avail_actions,) == action_mask.shape
            and (3,) == resources.shape
            and (max_avail_actions - 1,)== building_levels.shape),"Dimension missmatch"
        observation = {
            "action_mask": action_mask,
            "observations": np.concatenate((resources, building_levels,production)),
        }
        """{
            "resources": resources,
            "building_levels": building_levels,
        }"""
        return observation,rewards

    def _create_starting_village(self):
        new_village = Village(name=f"village{self.worker_id[:3]}")
        buildings = set()
        buildings.add(Headquarters(level=1, village=new_village))
        buildings.add(First_church(level=1, village=new_village))
        buildings.add(Rally_point(level=1, village=new_village))
        buildings.add(Farm(level=1, village=new_village))
        buildings.add(Warehouse(level=1, village=new_village))
        buildings.add(Hiding_place(level=1, village=new_village))
        new_village.set_buildings(buildings)
        return new_village

class ActionMaskModel(TorchModelV2, nn.Module):
    """PyTorch version of above ActionMaskingModel."""

    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        ), "This assertion is wrong"

        TorchModelV2.__init__(
            self, obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=model_config,name=name, **kwargs
        )
        nn.Module.__init__(self)
        self.internal_model = TorchFC(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask
        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()

    def _get_size(self,obs_space):
        return get_preprocessor(obs_space)(obs_space).size
