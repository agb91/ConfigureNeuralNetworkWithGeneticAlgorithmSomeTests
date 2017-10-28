import pytest
from geneCreator import GeneCreator
from gene import Gene 

class testGeneCreator:

	def testRandoms(self):

		gc = GeneCreator()
		rl = gc.randomLearning()
		assert ( rl <= 1.0 )
		assert (rl >= 0.0)

		st = gc.randomSteps()
		assert( st >= 100 )
		assert( st <= 1000 )

		ru = gc.randomUnit()
		assert( ru >= 1 )
		assert (ru <= 20)

		rlev = gc.randomLevel()
		assert( rlev >= 1 )
		assert( rlev <=3 )

		g = gc.randomCreate()
		assert g.isAcceptable() == True