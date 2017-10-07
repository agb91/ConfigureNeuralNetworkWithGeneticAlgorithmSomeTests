from neuralAbalone import neuralAbalone 

if __name__ == "__main__":
  # Learning rate for the model
  LEARNING_RATE = 0.01
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

  SET_OF_FEATURES = [0,1,2,3,5,6]
  nn = neuralAbalone( LEARNING_RATE, STEPS, UNITS1, UNITS2, UNITS3, SECOND, THIRD, SET_OF_FEATURES )
  nn.run()