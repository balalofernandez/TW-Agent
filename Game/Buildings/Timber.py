from ..Building import Building
class Timber(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'timber'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		if self.level > 0:
			self.village.production["wood"] = self.requirements[self.requirements['level'] == self.level].iloc[0].to_dict()["new_production"]

	def upgrade(self):
		current_level_requirements = self.requirements[self.requirements['level'] == self.level+1].iloc[0].to_dict()
		result,reward = super().upgrade()
		if not result:
			return result,reward
		self.village.production["wood"] = current_level_requirements["new_production"]
		return result,reward