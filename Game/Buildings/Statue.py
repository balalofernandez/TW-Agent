from ..Building import Building
class Statue(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'statue'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
