import unittest
from Game import Village
from Game.Buildings import *


class TestVillage(unittest.TestCase):
    def setUp(self):
        self.new_village = Village()
        buildings = set()
        buildings.add(Headquarters(level=3,village=self.new_village))
        buildings.add(First_church(level=1,village=self.new_village))
        buildings.add(Rally_point(level=1,village=self.new_village))
        buildings.add(Timber(level=2,village=self.new_village))
        buildings.add(Clay(level=2,village=self.new_village))
        buildings.add(Iron(level=1,village=self.new_village))
        buildings.add(Farm(level=2,village=self.new_village))
        buildings.add(Warehouse(level=1,village=self.new_village))
        buildings.add(Hiding_place(level=1,village=self.new_village))
        self.new_village.set_buildings(buildings)

    def test_points(self):
        points = self.new_village.compute_points()
        self.assertEqual(points,61)

    def test_get_buildings(self):
        upgrades,rewards = self.new_village.get_available_upgrades()
        actual_list = list(rewards)
        expected_list = ['timber', 'clay', 'iron', 'hiding_place', 'warehouse', 'farm', 'barracks', 'headquarters']
        self.assertEqual(len(actual_list),len(expected_list))
        for expected_item in actual_list:
            self.assertEqual(expected_item in expected_list,True)
    def test_upgrade_building(self):
        upgrades,rewards = self.new_village.get_available_upgrades()
        available_upgrades = list(rewards)
        print(available_upgrades)
        previous_level = 0
        if available_upgrades[0] in self.new_village.buildings:
            previous_level = self.new_village.buildings[available_upgrades[0]].level
        if self.new_village.upgrade_building(available_upgrades[0]):
            self.assertGreater(self.new_village.buildings[available_upgrades[0]].level, previous_level)
        self.assertEqual(self.new_village.upgrade_building("first_church"), False)

    def test_not_maxed_village(self):
        self.assertEqual(self.new_village.check_max_village(),False)



