import asyncio

from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog

from Game import *
from ray.rllib.algorithms.ppo import PPOConfig
from Game.Agent import MapEnv,ActionMaskModel

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
    env_config = {
        'action_freq': 2, 'max_steps': 2000, 'debug': True,'map':map
    }
    ModelCatalog.register_custom_model("action_mask_model", ActionMaskModel)
    config = (  # 1. Configure the algorithm,
        PPOConfig()
        .debugging(log_level="ERROR")
        .environment(MapEnv,env_config=env_config)
        .rollouts(num_rollout_workers=1)  # 2
        .framework("torch")
        .training(model={
                "custom_model": "action_mask_model",}, gamma=0.9, lr=0.01, kl_coeff=0.3,
                  train_batch_size=128)
        .resources(num_gpus=1)
        .evaluation(evaluation_num_workers=1)
        .framework(framework="torch")
    )
    algo = config.build()
    for i in range(2000):
        print(f"Episode {i}")
        results = algo.train()
        print(results)
