from ..Building import Building
class Hiding_place(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'hiding_place'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
