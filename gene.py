class Gene:

	def __init__( self, lr, steps, u1, u2, u3, second, third, feat ):
		self.LEARNING_RATE = lr
		self.STEPS = steps
		self.UNITS1 = u1
		self.UNITS2 = u2
		self.UNITS3 = u3
		self.SECOND = second
		self.THIRD = third
		self.SET_OF_FEATURES = feat

	def isAcceptable(self):
		if(self.LEARNING_RATE > 1 or self.LEARNING_RATE<0 ):
			return False
		if( self.STEPS > 40000 or self.STEPS< 0 ):
			return False	
		if( self.UNITS1 < 1 or self.UNITS1 > 100 ):
			return False	
		if( self.UNITS2 < 1 or self.UNITS2 > 100 ):
			return False	
		if( self.UNITS3 < 1 or self.UNITS3 > 100 ):
			return False
		if(self.SECOND <0 or self.SECOND > 1):
			return False
		if(self.THIRD <0 or self.THIRD > 1):
			return False
		return True	
