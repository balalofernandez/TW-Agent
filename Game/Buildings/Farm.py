from ..Building import Building
class Farm(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'farm'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
