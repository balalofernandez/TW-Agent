from ..Building import Building
class Watchtower(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'watchtower'
		self.level=level
		self.upgrade_requirements = super().read_requirements(level+1)
		super().__init__(self.name,self.level,self.upgrade_requirements,*args,**kwargs)
		self.building_requirements = {"headquarters":5,"farm":5}
