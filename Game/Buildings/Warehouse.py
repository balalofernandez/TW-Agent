from ..Building import Building
class Warehouse(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'warehouse'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)

	def upgrade(self):
		current_level_requirements = self.requirements[self.requirements['level'] == self.level + 1].iloc[0].to_dict()
		result, reward = super().upgrade()
		if not result:
			return result, reward
		self.village.storage_capacity = current_level_requirements["capacity"]
		return result, reward