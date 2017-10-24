import random

class Gene:


	def __init__( self, lr, steps, units):
		self.LEARNING_RATE = lr
		self.STEPS = steps
		self.UNITS = units

	def toStr( self ):
		print( "gene: " + str( self.LEARNING_RATE ) + " -- " + str( self.STEPS )+ " -- " +
			str( self.UNITS ) + ";    fitness level: " + str( self.level ) )	

	def setFitnessLevel( self, l ):
		self.level = l	

	def isAcceptable( self ):
		if(self.LEARNING_RATE > 1 or self.LEARNING_RATE<0 ):
			return False
		if( self.STEPS > 40000 or self.STEPS< 0 ):
			return False	
		if( len( self.UNITS ) < 1 or len( self.UNITS ) > 5 ):
			return False	
		return True	

