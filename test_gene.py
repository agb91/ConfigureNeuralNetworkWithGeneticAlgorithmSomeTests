from gene import Gene
import pytest

class TestGene():

	def test_constructor(self):
		lr = 0.1
		steps = 1000
		units = [1,2,3]
		g = Gene( lr, steps, units )

		assert g.LEARNING_RATE == lr
		assert g.UNITS == units
		assert g.STEPS == steps

	def test_fl(self):
		lr = 0.1
		steps = 1000
		units = [1,2,3]
		
		g = Gene( lr, steps, units )
		g.setFitnessLevel(22)

		assert g.level == 22
		

	def test_accept(self):
		lr = 0.1
		lrKo = 12
		steps = 1000
		stepsKo = 100000
		units = [1,2,3]
		unitsKo = [1,1,1,1,1,1]

		gOk = Gene( lr, steps, units )
		gKo1 = Gene( lrKo, steps, units )
		gKo2 = Gene( lr, stepsKo, units )
		gKo3 = Gene( lr, steps, unitsKo )

		assert gOk.isAcceptable() == True
		assert gKo1.isAcceptable() == False
		assert gKo2.isAcceptable() == False
		assert gKo3.isAcceptable() == False 
		