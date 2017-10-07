from gene import Gene
import numpy as np


class Breeder:

	def takeBest( self, genes ):

		maxLevel = 0
		maxGene = gene.randomCreate

		for ( g  in np.nditer(genes) ):
			if( g.level > maxLevel ):
				maxGene = g
				maxLevel = g.level

		return maxGene		