from ..Building import Building
class Clay(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'clay'
		self.level=level
		self.upgrade_requirements = super().read_requirements(level+1)
		self.current_requirements = super().read_requirements(level)
		super().__init__(self.name,self.level,self.upgrade_requirements,*args,**kwargs)
		if self.level > 0:
			self.village.production["clay"] = self.current_requirements["new_production"]
