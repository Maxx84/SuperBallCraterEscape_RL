
# Import libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib
import math
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# Connects Matlab to Pyhton
import transplant
# Imports Reinforcement learning class
from MultiAgentRLBruteForce import agent_class
import time
import timeit

# Starting Matlab
matlab = transplant.Matlab(arguments=['-desktop'])

matlab.addpath('tensegrityObjects')

#Define tspan=display time interval, wall position and wall height for Matlab simulation
tspan=0.05            # monitor refresh
wallPosition=0.76
wallHeight=5.0
delT=0.001             #time step

#Define how much the motors can spool in or out in one tspan.
deltaSpool=0.0005 # reduced from original 0.001

env=matlab.myEnvironmentSetup(tspan,deltaSpool,delT)

matlab.createGraph(env)


#Define some useful variables
lr=0.02                     # Learning rate
H=4                         # Size of the hidden layer
L=1                         # Number of hidden layers (exclude input and output layers)
gamma=0.99                  # Gamma used to decay the rewards, higher gamma values= future matters more
sessionTime=600           # Maximum number of cycles in each episode
error_cnt=0                 #Number of times the simulation crashed

MOTOR_NUMBER=24
#RENDER=False
RENDER=True
# Number of brute force approach cases
BRUTE_FORCE_CASES=1

maxCoord=[]
maxDistance=[]


# Create 1000 neural networks with random weight initialization and biases initialized to zero
# Neural netwrks are not trained, but saved as is and then loaded. The heigth of the center of mass is
# used as an estimator of how well each network is doing.
# This brute force approach is used to figure out which set of parameters gives the
# best result.

# Generate one agent per motor
agent={}
for i in range(BRUTE_FORCE_CASES):
    #graph=tf.Graph()
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(i)+"_"+str(m)]=agent_class(learning_rate=lr,actions_size=3,hidden_layer_size=H,features_size=MOTOR_NUMBER,gamma=gamma,L=L,n=i,m=m)
        #agent["A"+str(i)+str(m)].initializeVars()
        agent["A_"+str(i)+"_"+str(m)].saveSession()
print('Neural networks are ready')


print('Saved all the neural networks')
# Load each network and run it for the same time.
# Save the estimator of the netowork performance (height of center of mass)

for i in range(BRUTE_FORCE_CASES):
    # Load session
    for m in range(MOTOR_NUMBER):
        agent["A_"+str(i)+"_"+str(m)].loadSession()
    print('Loaded session '+str(i))

    #Start Evaluation Timer
    start = timeit.default_timer()
    #Observe initial rest lengths of the strings
    features = matlab.envReset(env,RENDER)

    features=np.reshape(features,(1,MOTOR_NUMBER))

    # For some reason, matlab.getCenterOfMass returns an array within an array.
    initialPosition = matlab.getCenterOfMass(env)[0]
    print("Initial position = ", initialPosition)
    
    #Initialize center of mass coordinates to 0
    centerMassCoord = initialPosition
    maxDistance=np.append(maxDistance,0)

    prtRefresh = 0

    for j in range(sessionTime):

        action=[]

        for m in range(MOTOR_NUMBER):
            action=np.append(action,agent["A_"+str(i)+"_"+str(m)].pick_action(features))
        observations= matlab.actionStep(env,action)

        # Assign the new features to the feature variable for the next cycle
        features=observations
        features=np.reshape(features,(1,MOTOR_NUMBER))

        if env.superBallDynamicsPlot.plotErrorFlag==1:
            error_cnt +=1
            print("ERROR count ",error_cnt)
            for m in range(MOTOR_NUMBER):
                agent["A_"+str(i)+"_"+str(m)].cancel_transition()
            break

        if RENDER:
            matlab.updateGraph(env)

        coordinate=matlab.getCenterOfMass(env)[0]

        centerMassCoord=np.append(centerMassCoord,coordinate)
        
        #print("Coordinate array: ", coordinate)
        

        # sqrt(x^2 + y^2)
        distanceTraveled = math.sqrt(math.pow(coordinate[0] - initialPosition[0],2) + math.pow(coordinate[1] - initialPosition[1],2))
        
        prtRefresh = prtRefresh + 1

        if prtRefresh > 10:
            print("Temporary distance: ", distanceTraveled)
            print("Elapsed Time: ", timeit.default_timer() - start)
            prtRefresh = 0

        if distanceTraveled>maxDistance[i]:
            maxDistance[i]=distanceTraveled


    stop = timeit.default_timer()
    
    # sqrt(x^2 + y^2)
    distanceTraveled = math.sqrt(math.pow(coordinate[0] - initialPosition[0],2) + math.pow(coordinate[1] - initialPosition[1],2))
    
    print("Max distance: ", maxDistance)
    print("Distance traveled: ", distanceTraveled)
    print ("Iteration time : ",stop - start )


print("Max Distance: ", maxDistance)
agent["A_"+str(i)+"_"+str(m)].saveCoordinates(maxDistance)
print(np.amax(maxDistance))
best=np.argmax(maxDistance)
print(best)

#Show best?

#Plot results
x=np.arange(0,len(maxDistance))
y1= maxDistance

fig= plt.figure(figsize=(20,10))
ax1=plt.subplot(211)
plt.scatter(x,y1)

plt.title('Max horizontal displacement of the center of mass for each Neural Network run')

plt.show()
