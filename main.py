from neuralAbalone import NeuralAbalone 
#from standardEstimators import StandardEstimators
from gene import Gene
from geneCreator import GeneCreator
from breeder import Breeder

if __name__ == "__main__":
  # Learning rate for the model
  #LEARNING_RATE = 0.001
  # How many steps
  #STEPS = 5000
  #How many units in each layer? (only if the layer exists obvious..)
  #UNITS = [10 , 10, 10]
  
  #confs = Gene( LEARNING_RATE, STEPS, UNITS )
  
  population = 4
  nGenerations = 4

  creator = GeneCreator()
  breeder = Breeder()
  generation = breeder.getFirstGeneration( population )
  generation = breeder.run( generation , 1 )

  for i in range ( 0 , nGenerations ):
    print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
    generation = breeder.getNewGeneration(generation , population)
    generation = breeder.run( generation , 1 )
    best = breeder.takeBest( generation )
    #best.toStr()
    print("we reach a loss of: " + str( best.level) )

  #nn = NeuralAbalone( confs )
  #loss = nn.run()
  print( "\n\n\n########################## IN THE END ##########################")
    
  print("we reach a loss of: " + str( best.level) )

