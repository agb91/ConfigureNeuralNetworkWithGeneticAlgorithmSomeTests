from gene import Gene
import random

class GeneCreator:
	
	def randomSet(self):
		return [0,1,2,3,4,5,6] #todo... for the moment take all the existing infos.. 

	def randomLearning(self):
		return ( random.random() / 100.0 )

	def randomSteps(self):
		return ( random.randint(100,10000) )
	
	def randomUnits(self):
		return ( random.randint(1,20) )

	def randomLevelExists(self):
		return ( random.randint(0,1) )

	def randomCreate(self):
		
		rl = self.randomLearning()
		rs = self.randomSteps()
		u1 = self.randomUnits()
		u2 = self.randomUnits()
		u3 = self.randomUnits()
		l2 = 1
		l3 = self.randomLevelExists()
		rset = self.randomSet
		#for the moment i wanna force at least 1 level
		gene = Gene( rl, rs, u1, u2, u3, l2, l3, rset )
		return gene 	
	