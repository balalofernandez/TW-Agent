import pandas as pd
import os
from .Building import Building
class Troop:
    #spear,swordsman,axeman,archer,scout,light,mounted,heavy,ram,catapult,paladin,nobleman,militia
    def __init__(self, name):
        self.name = name
        self.stats = self.read_troop(name)
        self.attack_power = 0
        self.general_defense = 0
        self.cavalry_defense = 0
        self.archer_defense = 0

    @staticmethod
    def read_troop(self,name):
        script_dir = os.path.dirname(__file__)
        file_path = os.path.join(script_dir, f"Troops/all_troops.csv")
        table = pd.read_csv(file_path)
        return table[table['unit'] == name].iloc[0].to_dict()

    def __eq__(self, other):
        if isinstance(other, Building):
            return self.name == other.name
        return False
    def __hash__(self):
        return hash(self.name)

