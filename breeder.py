from gene import Gene
from geneCreator import GeneCreator
from neuralAbalone import NeuralAbalone

import numpy as np
import random


class Breeder:

	def getNewGeneration( self, old, n):
		newGeneration = list()
		strongestN = 3
		if(n<3):
			strongestN = n
		goods = self.takeGoods( old , strongestN )
		for i in range( 0 , (n - 1) ):
			son = self.getSon( goods )
			newGeneration.append(son)

		return newGeneration

	def getSon( self, parents ):
		rlr = random.randint(0, (len(parents) - 1 ) )
		lr = parents[rlr].LEARNING_RATE
		rsteps = random.randint(0, (len(parents) - 1 ))
		steps = parents[rsteps].STEPS
		ru = random.randint(0, (len(parents) - 1 ))
		u = parents[ru].UNITS

		son = Gene( lr, steps, u )
		return son	

	def run(self, generation, verbose):
		runnedGeneration = list()
		
		for i in range( 0 , len(generation)):
			g = generation[i]
			nn = NeuralAbalone( g )
			g.setFitnessLevel( nn.run( verbose ) ) 
			#g.toStr()
			runnedGeneration.append(g)
		return runnedGeneration	

	def getFirstGeneration( self, n ):
		genes = list()
		creator = GeneCreator()
		for i in range( 0 , n):
			g = creator.randomCreate()
			genes.append(g)
		return genes

	def orderGenes( self , genes ):
		result = []
		result = sorted(genes, key=lambda x: x.level, reverse=False)
		return result

	def takeGoods( self, genes, n ):
		minLevel = 99 #level of error
		goods = []

		for i in range(0, len(genes) ):
			g = genes[i]
			goods.append(g)
			goods = self.orderGenes( goods )
			if( len( goods ) > n):
				goods = goods[ 0 : n ]
		return goods		    

	def takeBest( self, genes ):

		minLevel = 99 #level of error
		bestGene = None

		for i in range(0, len(genes) ):
			g = genes[i]
			if( g.level < minLevel ):
				bestGene = g
				minLevel = g.level

		return bestGene		