from ..Building import Building
class Rally_point(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'rally_point'
		self.level=level
		self.upgrade_requirements = super().read_requirements(level+1)
		super().__init__(self.name,self.level,self.upgrade_requirements,*args,**kwargs)
