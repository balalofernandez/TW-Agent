from ..Building import Building
class Academy(Building):
	def __init__(self,level=0,*args,**kwargs):
		self.name = 'academy'
		self.level=level
		super().__init__(self.name,self.level,*args,**kwargs)
		self.building_requirements = {"headquarters":20,"market":10,"smithy":20}

