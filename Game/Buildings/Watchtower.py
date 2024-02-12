from ..Building import Building
class Watchtower(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'watchtower'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"headquarters":5,"farm":5}
