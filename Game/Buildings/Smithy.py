from ..Building import Building
class Smithy(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'smithy'
		self.level=level
		self.upgrade_requirements = super().read_requirements(level+1)
		super().__init__(self.name,self.level,self.upgrade_requirements,*args,**kwargs)
		self.building_requirements = {"headquarters":5,"barracks":1}
