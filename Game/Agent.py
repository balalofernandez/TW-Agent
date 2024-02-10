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
from ray.rllib.utils.framework import try_import_torch

from Game import Map, Agent, Village
from Game.Buildings import *

class Agent:

    def __init__(self,name=""):
        self.name = name

    def choose_action(self,actions: list) -> str:
        return actions[0][0]


class MapEnv(gym.Env):
    def __init__(self, env_config):
        self.max_steps = env_config["max_steps"] or 1000
        self.valid_actions = ['headquarters', 'first_church', 'rally_point', 'farm', 'warehouse', 'hiding_place',
                              'timber', 'clay', 'iron', 'watchtower', 'wall', 'market', 'smithy', 'academy',
                              'church', 'workshop', 'stable', 'barracks', 'idle']
        # Since we are just going to consider building objects,
        # for now we are going to take a discrete approach on all building scenarios
        max_avail_actions = len(self.valid_actions)
        self.action_space = spaces.Discrete(max_avail_actions)
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=(max_avail_actions,), dtype=np.int8),
            "observations": spaces.Box(low=0, high=np.inf, shape=(3+max_avail_actions-1,), dtype=np.int32)
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

        self.current_step = 0
        self.agent = Agent()
        self.new_village = self._create_starting_village()
        print("STARTING")
        #Running
        self.start_game()
        # self.map = Map(self.agent, [new_village])
        observation,future_rewards = self._generate_observations_and_rewards()
        print("ST OBSERVATION",observation)
        return observation,{}

    def step(self, action):
        """
        :meth:`step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions,
          if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.

        :param action:
        :return:
        """
        # 1. Process the action
        self.current_step += 1
        action_name = self.valid_actions[action]
        print("ACTION",action_name)
        status, reward = self.new_village.perform_action(action_name)  # self.valid_actions[action]
        print("Status",status,"Reward",reward)
        # 2. Update the Observation and 3. Calculate the Reward
        observation,future_rewards = self._generate_observations_and_rewards()
        # 4. Check Termination Conditions
        terminated = self.new_village.check_max_village()
        # 5. Check Truncation - assuming a maximum number of steps for simplicity
        truncated = self.current_step >= self.max_steps  # Example condition
        print("OBSERVATION",observation)
        return observation, reward * 0.1, terminated, truncated, {}

    # def close(self):
    def start_game(self):
        if self.loop is None or self.loop.is_closed():
            self.initialize_async_loop()

            # Run village tasks asynchronously without waiting for them to complete
        if self.village_task:
            self.village_task.cancel()  # Cancel previous tasks if needed
        asyncio.run_coroutine_threadsafe(self.new_village.run(0.1), self.loop)
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
        resources = np.array(list(self.new_village.loot.values()), dtype=np.int32)
        building_levels = np.array(list(self.new_village.get_buildings().values()), dtype=np.int32)
        assert ((max_avail_actions,) == action_mask.shape
            and (3,) == resources.shape
            and (max_avail_actions - 1,)== building_levels.shape),"Dimension missmatch"
        observation = {
            "action_mask": action_mask,
            "observations": np.concatenate((resources, building_levels)),

        }
        """{
            "resources": resources,
            "building_levels": building_levels,
        }"""
        return observation,rewards

    def _create_starting_village(self):
        new_village = Village(name="village1")
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
        print("Observation space:", obs_space)
        print("Shape of observation space:", obs_space.shape)

        orig_space = getattr(obs_space, "original_space", obs_space)
        print("Is instance of Dict:", isinstance(orig_space, Dict))
        print("Has action_mask:", "action_mask" in orig_space.spaces)
        print("Has observations:", "observations" in orig_space.spaces)
        print("ORIG_SPACE:", orig_space)
        assert (
                isinstance(orig_space, Dict)
                and "action_mask" in orig_space.spaces
                and "observations" in orig_space.spaces
        ), "This assertion is wrong"

        TorchModelV2.__init__(
            self, obs_space=obs_space, action_space=action_space, num_outputs=num_outputs, model_config=model_config,name=name, **kwargs
        )
        nn.Module.__init__(self)
        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        """
        self.obs_size = self._get_size(obs_space)
        self.fc1 = nn.Sequential(nn.Linear(self.obs_size,256),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256,256),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(256,num_outputs),nn.ReLU())
        """
        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):

        # Extract the available actions tensor from the observation.
        print("INPUT DICT",input_dict["obs"]["observations"]["building_levels"].shape)
        print("STATE",state)
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})
        """fc1_out = self.fc1(input_dict["obs"]["observations"])
        fc2_out = self.fc2(fc1_out)
        logits = self.fc3(fc2_out)"""

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

class FullyConnectedNetwork(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        print("FC_net", activation,no_final_linear,self.vf_share_layers,self.free_log_std)

        layers = []
        print("OBS_SPACE AGAIN",obs_space)
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None
        print(prev_layer_size)
        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation,
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict,
        state,
        seq_lens,
    ):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            out = self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            out = self._value_branch(self._features).squeeze(1)
        return out