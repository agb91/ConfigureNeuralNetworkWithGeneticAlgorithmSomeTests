from breeder import Breeder
from gene import Gene
import pytest

class TestBreeder:

	def testFirstGen(self):
		nPop = 4
		breeder = Breeder()
		gen = list()
		gen = breeder.getFirstGeneration( nPop )
		assert len( gen ) == nPop 

		for i in range( 0, (nPop - 1) ):
			gene = gen[i]
			assert gene.isAcceptable() == True
	
	def testNewGen(self):
		nPop = 4
		breeder = Breeder()
		gen = list()
		gen = breeder.getFirstGeneration( nPop )
		gen = breeder.run( gen , 0 )
		
		newGen = list()
		newGen = breeder.getNewGeneration(gen , nPop)
		assert len(newGen) == nPop
		newGen = breeder.getNewGeneration(gen , (nPop - 2)  )
		assert len(newGen) == nPop - 2
		newGen = breeder.getNewGeneration(gen , (nPop + 7) )
		assert len(newGen) == nPop + 7
	
	def testGetSon(self):
		breeder = Breeder()
		g1 = Gene( 1, 100, [1,2,3] )
		g2 = Gene( 2, 200, [9,10,11,12] )
		son = breeder.getSon( [g1,g2] )
		
		count = 0
		if( son.LEARNING_RATE == g1.LEARNING_RATE ):
			count += 1
		if( son.LEARNING_RATE == g2.LEARNING_RATE ):
			count += 1	
		assert ( count == 1 )

		count = 0
		if( son.STEPS == g1.STEPS ):
			count += 1
		if( son.STEPS == g2.STEPS ):
			count += 1	
		assert ( count == 1 )

		count = 0
		if( son.UNITS == g1.UNITS ):
			count += 1
		if( son.UNITS == g2.UNITS ):
			count += 1	
		assert ( count == 1 )


