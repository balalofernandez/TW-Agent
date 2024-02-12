from ..Building import Building
class Stable(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'stable'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"headquarters":10,"barracks":5,"smithy":5}
