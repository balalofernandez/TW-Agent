import pandas as pd
import os
class Building:
    def __init__(self, name, level,improvement_requirements=None,village=None,*args,**kwargs):
        self.name = name
        self.level = level
        self.max_level = False
        self.rewards = self._get_reward_list()
        self.next_reward = 0
        if len(self.rewards)>self.level:
            self.next_reward = self.rewards[self.level]
        else:
            self.max_level = True
        #We need to add building constraints to the requirements
        if improvement_requirements == None:
            improvement_requirements = self.read_requirements(self.level)
        self.requirements = improvement_requirements
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
    def read_requirements(self,level=1):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, f"Buildings/building_requirements/{self.name}.csv")
        table = pd.read_csv(file_path)
        try:
            requirements = table[table['level'] == level+1].iloc[0].to_dict()
            requirements["wood"] = float(requirements["wood"])
            requirements["iron"] = float(requirements["iron"])
            requirements["clay"] = float(requirements["clay"])
        except:
            self.max_level = True
            requirements = None
        return requirements

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
    def upgrade(self):
        new_reward = self.next_reward
        if self.max_level:
            return False
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

