import pandas as pd
import os
import asyncio

from Game import Building
from Game.Buildings import create_building


class Village():
    all_troops = ['spear','swordsman','axeman','archer','scout','light','mounted','heavy','ram','catapult','paladin','nobleman','militia']
    excluded_buildings = ['statue',]
    all_buildings = ['headquarters','timber', 'clay', 'iron','first_church','rally_point','farm', 'warehouse','hiding_place',
                       'watchtower', 'wall','market','smithy', 'academy',
                      'church', 'workshop', 'stable', 'barracks']
    flags = {"wood_upgrade": False,
              "iron_upgrade": False,
              "clay_upgrade": False,
                  }
    max_levels = {
        'headquarters': 30,
        'barracks': 25,
        'stable': 20,
        'workshop': 15,
        'academy': 1,
        'smithy': 20,
        'rally_point': 1,
        'statue': 1,
        'market': 25,
        'timber_camp': 30,
        'clay_pit': 30,
        'iron_mine': 30,
        'farm': 30,
        'warehouse': 30,
        'hiding_place': 10,
        'wall': 20
    }
    lock = asyncio.Lock()

    def __init__(self, buildings=[],name=''):
        self.name = name
        self.buildings = {}
        for building in buildings:
            self.buildings[building.name] = building
        self.troops = []
        self.loot = {"wood":500,
                     "clay":500,
                     "iron":400}
        self.production = {"wood":5,
                     "clay":5,
                     "iron":5}
        self.coordinates = [0.,0.]
        self.storage_capacity = 1000
        self.max_farm = 240
        self.current_farm = 12
        self.point_table = self.read_points()

        #self.points = self.compute_points() #This may be a good way to encourage the agent to improve

    async def run(self,speed):
        print("Starting Village",self.name)
        task1 = asyncio.create_task(self._produce_wood(speed),name="wood")
        task2 = asyncio.create_task(self._produce_clay(speed),name="clay")
        task3 = asyncio.create_task(self._produce_iron(speed),name="iron")
        await asyncio.gather(task1, task2, task3)

    def perform_action(self,action):
        print(f"Resources of {self.name}: {self.loot}")
        print(f"Production of {self.name}: {self.production}")
        if action!="idle":
            return self.upgrade_building(action)
        return True,0.

    def set_buildings(self,buildings):
        self.buildings = {}
        for building in buildings:
            self.buildings[building.name] = building

    def get_buildings(self):
        buildings_dict = {}
        for building_name in self.all_buildings:
            if building_name in self.buildings:
                buildings_dict[building_name] = self.buildings[building_name].level
            else:
                buildings_dict[building_name] = 0
        return buildings_dict

    def check_max_village(self):
        for building_name in self.all_buildings:
            if not(building_name in self.buildings):
                return False
            if self.max_levels[building_name] != self.buildings[building_name].level:
                return False
        return True

    @staticmethod
    def read_points():
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, f"Points/points.csv")
        table = pd.read_csv(file_path).astype(float)
        return table
    def compute_points(self):
        total_points = 0
        for building in self.buildings.values():
            if building.name in self.point_table.columns and not self.point_table[building.name].empty:
                #print(f"{building.name},level {building.level}:", table[building.name][building.level-1].sum())
                total_points += self.point_table[building.name][:building.level].sum()
        return total_points

    def upgrade_building(self,building_name):
        """
        This method supposes that you have run get_available_upgrades before
        :param building:
        :return:
        """
        if building_name in self.buildings.keys():
            return self.buildings[building_name].upgrade()
        building = create_building(building_name,level=0,village=self)
        correct,reward = building.upgrade()
        if correct:
            self.buildings[building_name] = building
            return correct,reward
        return False,-1.

    def get_available_upgrades(self):
        """
        :return:
            - available_upgrades contains a mask [0,1] with all the buildings that can be upgraded
            - upgrade_rewards is a dictionary with the rewards for each upgradeable building
        """
        available_upgrades = {}
        upgrade_rewards = {}
        for building_name in self.all_buildings:
            available_upgrades[building_name] = 0
            #Check if building is already built
            if building_name in self.buildings:
                if self.buildings[building_name].max_level:
                    continue
                available,reward = self._building_available_upgrade(self.buildings[building_name])
                if available:
                    available_upgrades[building_name] = 1
                    upgrade_rewards[building_name] = reward
            else:
                available,reward = self._building_available_construction(building_name)
                if available:
                    available_upgrades[building_name] = 1
                    upgrade_rewards[building_name] = reward
        return available_upgrades,upgrade_rewards

    def _building_available_construction(self,building_name):
        available = (False,0)
        building = create_building(building_name)
        if not building.building_requirements == None:
            for constraint in building.building_requirements:
                if constraint in self.buildings:
                    #Check if we have the required level of the building
                    if self.buildings[constraint].level<building.building_requirements[constraint]:
                        return available
                else:
                    return available
        next_lvl_requirements = building.requirements[building.requirements['level'] == building.level+1].iloc[0].to_dict()
        if self.loot["wood"]<next_lvl_requirements["wood"]:
            return available
        if self.loot["iron"]<next_lvl_requirements["iron"]:
            return available
        if self.loot["clay"]<next_lvl_requirements["clay"]:
            return available
        return (True,building.next_reward)


    def _building_available_upgrade(self,building):
        #Check loot requirements
        available = (False,0)
        next_lvl_requirements = building.requirements[building.requirements['level'] == building.level+1].iloc[0].to_dict()
        if self.loot["wood"]<next_lvl_requirements["wood"]:
            return available
        if self.loot["iron"]<next_lvl_requirements["iron"]:
            return available
        if self.loot["clay"]<next_lvl_requirements["clay"]:
            return available
        return (True,building.next_reward)

    #RUNNING METHODS
    async def _produce_wood(self,speed):
        #see how to cancel this when storage full
        while True:
            second_production = self.production["wood"]/(60*60)
            time_span_one_ore = 1/second_production*speed
            await asyncio.sleep(time_span_one_ore)
            if not(self.flags["wood_upgrade"] or self.storage_capacity<=self.loot["wood"]):
                async with self.lock:  # Ensure safe access to the shared resource
                    self.loot["wood"] += 1
    async def _produce_clay(self,speed):
        while True:
            second_production = self.production["clay"]/(60*60)
            time_span_one_ore = 1/second_production*speed
            await asyncio.sleep(time_span_one_ore)
            if not(self.flags["clay_upgrade"] or self.storage_capacity<=self.loot["clay"]):
                async with self.lock:  # Ensure safe access to the shared resource
                    self.loot["clay"] += 1
    async def _produce_iron(self,speed):
        while True:
            second_production = self.production["iron"]/(60*60)
            time_span_one_ore = 1/second_production*speed
            await asyncio.sleep(time_span_one_ore)
            if not(self.flags["iron_upgrade"] or self.storage_capacity<=self.loot["iron"]):
                async with self.lock:  # Ensure safe access to the shared resource
                    self.loot["iron"] += 1
    #METHODS:
        #construct building
        #available constructions
        #train troops