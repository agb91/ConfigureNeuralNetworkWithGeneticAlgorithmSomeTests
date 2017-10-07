from neuralAbalone import NeuralAbalone 
from gene import Gene
from geneCreator import GeneCreator

if __name__ == "__main__":
  # Learning rate for the model
  LEARNING_RATE = 0.001
  # How many steps
  STEPS = 5000
  #How many units in each layer? (only if the layer exists obvious..)
  UNITS1 = 10
  UNITS2 = 10
  UNITS3 = 10
  # if 1 there is a second layer
  SECOND = 1
  # if 1 there is a third layer
  THIRD = 0

  SET_OF_FEATURES = [0,1,2,3,4,5,6]



  confs = Gene( LEARNING_RATE, STEPS, UNITS1, UNITS2, UNITS3, SECOND, THIRD, SET_OF_FEATURES )
  
  creator = GeneCreator()


  g1 = creator.randomCreate()
  g1.toStr()

  g2 = creator.randomCreate()
  g2.toStr()

  g3 = creator.randomCreate()
  g3.toStr()

  nn = NeuralAbalone( confs )
  loss = nn.run()

  print("we reach a loss of: " + str( loss ) )