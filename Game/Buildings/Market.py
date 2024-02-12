from ..Building import Building
class Market(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'market'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"headquarters":3,"warehouse":2}
