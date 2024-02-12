import pandas as pd
import os
class Building:
    def __init__(self, name, level,village=None,*args,**kwargs):
        self.name = name
        self.level = level
        self.max_level = False
        self.rewards = self._get_reward_list()
        self.next_reward = 0
        if len(self.rewards)>self.level:
            self.next_reward = self.rewards[self.level]
        else:
            self.max_level = True
        self.requirements = self.read_requirements()
        self.building_requirements = self._check_building_requirements()
        if village != None:
            self.village = village

    def _get_reward_list(self):
        # Let's get a list of the rewards for each level
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, f"Points/points.csv")
        table = pd.read_csv(file_path)
        if self.name in table.columns and not table[self.name].empty:
                return table[self.name].dropna().to_list()
        return []

    def read_requirements(self):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, f"Buildings/building_requirements/{self.name}.csv")
        table = pd.read_csv(file_path)
        table["wood"] = table["wood"].astype(float)
        table["iron"] = table["iron"].astype(float)
        table["clay"] = table["clay"].astype(float)
        return table

    def _check_building_requirements(self):
        buildings_with_requirements = {
            "wall":{"barracks":1},
            "market":{"headquarters":3,"warehouse":2},
            "smithy":{"headquarters":5,"barracks":1},
            "academy":{"headquarters":20,"market":10,"smithy":20},
            "watchtower":{"headquarters":5,"farm":5},
            "workshop":{"headquarters":10,"smithy":10},
            "stable":{"headquarters":10,"barracks":5,"smithy":5},
        }
        if self.name in buildings_with_requirements:
            return buildings_with_requirements[self.name]
        return {}

    #On improve we have to update the next_reward and max.
    def upgrade(self,current_level_requirements=None):
        new_reward = self.next_reward
        if self.max_level:
            print("ERROR:", self.name, self.level)
            return False, 0
        if not current_level_requirements:
            current_level_requirements = self.requirements[self.requirements['level'] == self.level+1].iloc[0].to_dict()
        #async with self.lock:  # Ensure safe access to the shared resource
        self.village.loot["wood"] -= current_level_requirements["wood"]
        self.village.loot["iron"] -= current_level_requirements["iron"]
        self.village.loot["clay"] -= current_level_requirements["clay"]
        assert (self.village.loot["wood"]>=0 and self.village.loot["clay"]>=0 and self.village.loot["iron"]>=0),"not enough resources"
        self.level +=1
        if len(self.rewards)>self.level:
            self.next_reward = self.rewards[self.level]
        else:
            self.max_level = True
        return True,new_reward

    def __eq__(self, other):
        if isinstance(other, Building):
            return self.name == other.name
        return False
    def __hash__(self):
        return hash(self.name)

