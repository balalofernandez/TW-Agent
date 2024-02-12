from ..Building import Building
class Barracks(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'barracks'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"headquarters":3}
