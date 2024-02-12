from ..Building import Building
class Headquarters(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'headquarters'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
