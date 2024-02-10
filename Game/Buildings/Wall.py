from ..Building import Building
class Wall(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'wall'
		self.level=level
		self.upgrade_requirements = super().read_requirements(level+1)
		super().__init__(self.name,self.level,self.upgrade_requirements,*args,**kwargs)
		self.building_requirements = {"barracks":1}
