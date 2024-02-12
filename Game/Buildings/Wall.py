from ..Building import Building
class Wall(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'wall'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"barracks":1}
