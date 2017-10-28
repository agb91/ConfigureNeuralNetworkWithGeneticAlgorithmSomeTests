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
	'''
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
	'''
	def testGetSon(self):
		breeder = Breeder()
		g1 = Gene( 1, 100, [1,2,3] )
		print( "\nparent1:" )
		g1.toStr()
		g2 = Gene( 2, 200, [9,10,11,12] )
		print( "\nparen2:" )
		g2.toStr()
		son = breeder.getSon( [g1,g2] )
		print( "\n son:" )
		son.toStr()
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

		major = len( g1.UNITS )
		if( len( g2.UNITS ) > major ):
			major = len( g2.UNITS )
		for i in range( 0, (major - 1) ):
			count = 0
			if( len(son.UNITS) > i and len(g1.UNITS) > i ):
				if( son.UNITS[i] == g1.UNITS[i] ):
					count += 1
			if( len(son.UNITS) > i and len(g2.UNITS) > i ):
				if( son.UNITS[i] == g2.UNITS[i] ):
					count += 1	
			assert ( count == 1 )

	def testOrderAndTakeGenes(self):
		breeder = Breeder()
		g1 = Gene( 1, 100, [1,2,3] )
		g1.setFitnessLevel(5.2)
		g2 = Gene( 1, 100, [1,2,3] )
		g2.setFitnessLevel(15.8)
		g3 = Gene( 1, 100, [1,2,3] )
		g3.setFitnessLevel(12.83)
		g4 = Gene( 1, 100, [1,2,3] )
		g4.setFitnessLevel(3.23)
		g5 = Gene( 1, 100, [1,2,3] )
		g5.setFitnessLevel(2.15)
		lg = [g1,g2,g3,g4,g5]
		ordered = list()
		ordered = breeder.orderGenes( lg )
		assert( len(ordered) == len(lg) )
		assert( ordered[0].level == 2.15 )
		assert( ordered[1].level == 3.23 )
		assert( ordered[2].level == 5.2 )
		assert( ordered[3].level == 12.83 )
		assert( ordered[4].level == 15.8 )

		nGoods = 3 
		goods = breeder.takeGoods( lg , nGoods )
		assert( len(goods) == nGoods )
		assert( goods[0].level == 2.15 )
		assert( goods[1].level == 3.23 )
		assert( goods[2].level == 5.2 )
	
		best = breeder.takeBest( lg )
		assert( best.level == 2.15 )