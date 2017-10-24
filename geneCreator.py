from gene import Gene
import random

class GeneCreator:
	
	def randomLearning(self):
		return ( random.random() / 100.0 )

	def randomSteps(self):
		return ( random.randint(100,10000) )
	
	def randomUnit(self):
		return ( random.randint(1,20) )

	def randomLevels(self):
		return ( random.randint(1,3) )

	def randomCreate(self):
		
		rl = self.randomLearning()
		rs = self.randomSteps()
		u = []
		levels = self.randomLevels()
		for i in range( 0 , levels ):
			u.append( self.randomUnit() )
		gene = Gene( rl, rs, u)
		return gene 	
	