from ..Building import Building
class Church(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'church'
		self.level=level
		self.upgrade_requirements = super().read_requirements(level+1)
		super().__init__(self.name,self.level,self.upgrade_requirements,*args,**kwargs)
