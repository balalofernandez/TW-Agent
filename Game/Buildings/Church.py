from ..Building import Building
class Church(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'church' 
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
