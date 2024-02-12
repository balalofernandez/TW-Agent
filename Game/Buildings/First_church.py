from ..Building import Building
class First_church(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'first_church'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
