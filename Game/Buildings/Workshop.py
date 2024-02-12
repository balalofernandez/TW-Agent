from ..Building import Building
class Workshop(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'workshop'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"headquarters":10,"smithy":10}
