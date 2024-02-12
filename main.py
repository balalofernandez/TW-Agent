import asyncio

import ray
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from Game import *
from ray.rllib.algorithms.ppo import PPOConfig
from Game.Agent import MapEnv,ActionMaskModel
import torch

if __name__ == '__main__':
    new_village = Village(name="village1")
    buildings = set()
    buildings.add(Headquarters(level=3, village=new_village))
    buildings.add(First_church(level=1, village=new_village))
    buildings.add(Rally_point(level=1, village=new_village))
    buildings.add(Timber(level=20, village=new_village))
    buildings.add(Clay(level=2, village=new_village))
    buildings.add(Iron(level=1, village=new_village))
    buildings.add(Farm(level=2, village=new_village))
    buildings.add(Warehouse(level=1, village=new_village))
    buildings.add(Hiding_place(level=1, village=new_village))
    new_village.set_buildings(buildings)
    new_village2 = Village(name="village2")
    buildings = set()
    buildings.add(Headquarters(level=3, village=new_village2))
    buildings.add(First_church(level=1, village=new_village2))
    buildings.add(Rally_point(level=1, village=new_village2))
    buildings.add(Timber(level=10, village=new_village2))
    buildings.add(Clay(level=10, village=new_village2))
    buildings.add(Iron(level=10, village=new_village2))
    buildings.add(Farm(level=2, village=new_village2))
    buildings.add(Warehouse(level=1, village=new_village2))
    buildings.add(Hiding_place(level=1, village=new_village2))
    new_village2.set_buildings(buildings)
    agent = Agent()
    map = Map(agent,[new_village], [new_village2])
    #Let's run the game
    #asyncio.run(map.run_game())
    ray.init(num_gpus=1,num_cpus=10)
    env_config = {
        'agent_wait': .2, 'max_steps': 3000, 'debug': True,'map':map,'speed':0.0001,
    }
    ModelCatalog.register_custom_model("action_mask_model", TorchActionMaskModel)
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .debugging(log_level="DEBUG")
        .environment(MapEnv,env_config=env_config)
        .rollouts(num_rollout_workers=8)  # 2
        .framework("torch")
        .training(model={
                "custom_model": "action_mask_model",}, gamma=0.9, lr=0.01, kl_coeff=0.3,
                  train_batch_size=512)
        .resources(num_gpus=1)
        .framework("torch")
        .evaluation(evaluation_num_workers=1,evaluation_interval=100)
    )
    algo = config.build()
    for i in range(1000):
      results = algo.train()
      print("RESULTS",results)
      if i % 200 == 0:
          evaluation_result = algo.evaluate()
          with open('output.txt', 'a') as file:
              file.write(f"{str(evaluation_result)}\n")

    save_result = algo.save()
    path_to_checkpoint = save_result.checkpoint.path
    print(
        "An Algorithm checkpoint has been created inside directory: "
        f"'{path_to_checkpoint}'."
    )