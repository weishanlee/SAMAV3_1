# -*- coding: utf-8 -*-
"""
Created on 
@author: weishan_lee

Sight-seeing order of Macao World Heritage Sites
Case 2: Travel distance or time for pairs of cities recorded in the following csv files: 
        (1) carTime.csv records time required for driving a car.
        (2) busTime.csv records time required for taking a bus.
        (3) pedestrianTime.csv records time required by walking between a pair of cities.
        The optimal route is found based on the Simulated Annealing and Metropolis Algorithm. 
Version 3_1: 1. Write to log.txt automatically.
             2. Add funcion definition plotRout
"""
from math import exp
import numpy as np
import random as rand
from vpython import * 
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import sys
import os.path

## Function Definitions

# Function to calculate the total distance or time of the tour
def distance(randomList, rCoor):
    s = 0.0
    for i in range(N):
        j = randomList[i-1]
        k = randomList[i]
        s += rCoor[j,k]
    j = randomList[-1]
    k = randomList[0]
    s += rCoor[j,k]
    return s

# output of the score (distance vs time steps)
def outPutScrVSTime(tRecord, scoreRecord):
    data = {'timeStep': tRecord,'score':scoreRecord}
    dfCSV = pd.DataFrame(data)
    dfCSV_file = open('./scoreVSTime.csv','w',newline='') 
    dfCSV.to_csv(dfCSV_file, sep=',', encoding='utf-8',index=False)
    dfCSV_file.close()
    
def outPutSitesOrder(randomList):
    ## Write randomList back to cities datafram
     
    sites["sitesOrder"] = randomList
    
    sitesOrder = pd.DataFrame(columns = ['sitesId', 'Name'])
    sitesOrder_file = open("./sightSeeingOrder.csv",'w',newline='') 

    for i in range(N+1):
        if i == N:
            integer = np.uint32(sites.loc[0].sitesOrder)
            sitesOrder.loc[i] = integer, sites.loc[integer].Name
        else:
            integer = np.uint32(sites.loc[i].sitesOrder)
            sitesOrder.loc[i] = integer, sites.loc[integer].Name

    sitesOrder.to_csv(sitesOrder_file, sep=',', encoding='utf-8', index=False) 
    sitesOrder_file.close()

def plotRoute(rr, sites):
    x = []
    y = []
    n = [int(num) for num in rCoor[:,3].tolist()]

    for i in range(N+1):
        if i == N:
            x.append( sites.loc[n[0]].X )
            y.append( sites.loc[n[0]].Y )
        else:
            x.append( sites.loc[n[i]].X )
            y.append( sites.loc[n[i]].Y )
    fig, ax = plt.subplots()
    ax.title.set_text("Optimal Tour Path")

    ax.plot(x,y,'k-')
    ax.scatter(x[0],y[0],c='blue')
    ax.scatter(x[1:-1],y[1:-1],c='red')

    for i, txt in enumerate(n):
        ax.annotate(txt, (x[i], y[i]))

    ax.set_xlabel("Longitude",size = 12)
    ax.set_ylabel("Latitude",size = 12)
    ax.ticklabel_format(useOffset=False)
    plt.grid(True)
    plt.savefig("optimalTourPath.eps")     
    
def writeLog(msg):
    with open('log.txt', 'a+') as the_file:
        print(msg)
        the_file.write(msg)
        
import os, psutil
def cpu_stats():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2. ** 30
    return 'Memory: ' + str(np.round(memory_use, 2)) + 'GB\t'

########################## Parameters and Options ############################
## If you need animation?
animation = False
## If you need to record score vs time step?
scoreVsTime = True

## Set up Case and load matrix of time or distance for each pair of cities.
# case = 1: car Time. 
# case = 2: bus time. Some pair of route may be replaced by pedestrian time.
# case = 3: pedestrin time.
# case = 4: car distance.
# cas3 = 5: pedestrian distance.
case = 1

## Parameters for Simulated annealing
Tmax = 1.0
Tmin = 1e-2
tau = 1e3
targetScore = 79 # carTime 79. busTime = 118 and  pedestrianTime = 116
                   # carDistance 14.1
###############################################################################

# Load world heritage sites locations
sites = pd.read_csv("./macauWHSLoc.csv")
R = 0.02
N = sites.shape[0]

## normalize data

sites['normX'] = min_max_scaler.fit_transform(sites.X.values.reshape(-1, 1))
sites['normY'] = min_max_scaler.fit_transform(sites.Y.values.reshape(-1, 1))

if case == 1:
    matrix_ = pd.read_csv("./carTime.csv")
elif case == 2:
    matrix_ = pd.read_csv("./busTime.csv")
elif case == 3:
    matrix_ = pd.read_csv("./pedestrianTime.csv")
elif case == 4:
    matrix_ = pd.read_csv("./carDistance.csv")
else:
    matrix_ = pd.read_csv("./pedestrianDistance.csv")

N = 25 # number of sites

# Set up the initial configuration
randomList = rand.sample(range(0, N), N)

## Change sites dataframe to rCoor array 
# rCoor could mean the time or distance of a pair of cities.

rCoor = np.empty([N,N])
for i in range(N):
    for j in range(N):
        rCoor[i,j] = matrix_.iloc[i][j]   # matrix value

## Change sites dataframe to rPlot array
rPlot = np.empty([N+1,4])
for i in range(N):
    j = randomList[i]
    rPlot[i,0] = sites.normX[j]
    rPlot[i,1] = sites.normY[j]
    rPlot[i,2] = 0.0
    rPlot[i,3] = sites.SiteId[j]
    
# Add one more ending site which is identical the starting site
rPlot[N,0] = rPlot[0,0]
rPlot[N,1] = rPlot[0,1]
rPlot[N,2] = rPlot[0,2]
rPlot[N,3] = rPlot[0,3]

#Calculate the initial distance

score = distance(randomList, rCoor)
initScore = score
minScore = initScore
msg = "Initial score = {:.5f}\n".format(initScore)
writeLog(msg)

# Set up the graphics
if animation == True:
    scene = canvas(center=vector(0.5,0.5,0.0), background = color.white)
    for i in range(N):
        if i == 0:
            sphere(pos=vector(rPlot[i,0],rPlot[i,1],0.0),radius=R,color = color.blue)
        else:
            sphere(pos=vector(rPlot[i,0],rPlot[i,1],0.0),radius=R,color = color.black)
    l = curve(pos=rPlot.tolist(),radius=R/4,color = color.red)

## Simulated annealing
## Main loop

tRecord = []
scoreRecord = []

t0=0 # setting up the beginning of the time "lump"
tRecord += [0]
scoreRecord += [score]

firstInitial = True

while (score>targetScore):
    
    if firstInitial == False: 
        # Set up another initial configuration
        randomList = rand.sample(range(0, N), N)

        
        ## Change sites dataframe to rCoor array
        rCoor = np.empty([N,N])
        for i in range(N):
            for j in range(N):
                rCoor[i,j] = matrix_.iloc[i][j] 
        
        #Calculate the initial distance
        score = distance(randomList, rCoor)

        ## Change sites dataframe to rPlot array
        rPlot = np.empty([N+1,4])
        for i in range(N):
            j = randomList[i]
            rPlot[i,0] = sites.normX[j]
            rPlot[i,1] = sites.normY[j]
            rPlot[i,2] = 0.0
            rPlot[i,3] = sites.SiteId[j]
    
        # Add one more ending site which is identical the starting site
        rPlot[N,0] = rPlot[0,0]
        rPlot[N,1] = rPlot[0,1]
        rPlot[N,2] = rPlot[0,2]
        rPlot[N,3] = rPlot[0,3]
        
        if animation == True:
            # Set up the graphics
            scene.delete()
            scene = canvas(center=vector(0.5,0.5,0.0), background = color.white)
            for i in range(N):
                if i == 0:
                    sphere(pos=vector(rPlot[i,0],rPlot[i,1],0.0),radius=R,color = color.blue)
                else:
                    sphere(pos=vector(rPlot[i,0],rPlot[i,1],0.0),radius=R,color = color.black)
            l = curve(pos=rPlot.tolist(),radius=R/4,color = color.red)

    T = Tmax
    t = 0
    while (T>Tmin):
        # Cooling
        t += 1
        T = Tmax*exp(-t/tau)

        # Choose two sites to swap and make sure they are distinct
        i,j = rand.randrange(1,N),rand.randrange(1,N)
        while i==j:
            i,j = rand.randrange(1,N),rand.randrange(1,N)
                
        # Swap them and calculate the change in score
        oldScore = score
        
        randomList[i], randomList[j] = randomList[j], randomList[i]
        
        rPlot[i,0],rPlot[j,0] = rPlot[j,0],rPlot[i,0]
        rPlot[i,1],rPlot[j,1] = rPlot[j,1],rPlot[i,1]
        rPlot[i,2],rPlot[j,2] = rPlot[j,2],rPlot[i,2]
        rPlot[i,3],rPlot[j,3] = rPlot[j,3],rPlot[i,3]
        
        score = distance(randomList, rCoor)        
        deltaScore = score - oldScore
        #print("deltaScore = {:.5f}".format(deltaScore))

        try:
            ans = np.exp(-deltaScore/T)
        except OverflowError:
            if -deltaScore/T > 0:
                ans = float('inf')
            else:
                ans = 0.0
    
        # If the move is rejected, swap them back again
        if rand.random() > ans:
            
            randomList[i], randomList[j] = randomList[j], randomList[i]
            
            rPlot[i,0],rPlot[j,0] = rPlot[j,0],rPlot[i,0]
            rPlot[i,1],rPlot[j,1] = rPlot[j,1],rPlot[i,1]
            rPlot[i,2],rPlot[j,2] = rPlot[j,2],rPlot[i,2]
            rPlot[i,3],rPlot[j,3] = rPlot[j,3],rPlot[i,3]
            score = oldScore
            if np.abs(score - distance(randomList, rCoor))>1e-5:
                msg = "score: {}".format(score)
                writeLog(msg)
                msg = "distance: {}".format(distance(randomList, rCoor))
                writeLog(msg)
                msg = "Error Line 262"
                writeLog(msg)
                sys.exit(msg)
        
        if score < minScore: 
            minScore = score
            outPutScrVSTime(tRecord, scoreRecord)
            outPutSitesOrder(randomList)
            dt = datetime.now()
            msg = str(dt.year) + '/' + str(dt.month)  + '/' + str(dt.day) + ' ' +\
                  str(dt.hour) + ':' + str(dt.minute) + ':' + str(dt.second) +'\t'
            writeLog(msg)
            msg = "Delta score = {:.5f}\t".format(deltaScore)
            writeLog(msg)
            msg = "New score = {:.5f}\n".format(score)
            writeLog(msg)
            
        if animation == True:    
            # Update the visualization every 100 moves
            if t%100==0:
                rate(25)
                for i in range(N+1):
                    pos = vector(rPlot[i,0],rPlot[i,1],0.0)
                    l.modify(i,pos)
    
        if scoreVsTime == True:
            #if t%1==0:
            tRecord += [t0+t]
            scoreRecord += [score]
        
        #writeLog(cpu_stats())
        
    t0 = t0 + t # go to next time "lump"
    firstInitial = False
# End of Main Loop
if case == 1 or case == 2 or case == 3:
    msg = "The initial total traveling time = {:.5f} min\n".format(initScore)
    writeLog(msg)
    msg = "The optimal total traveling time = {:.5f} min\n".format(score)
    writeLog(msg)
else:
    msg = "The initial total traveling distance = {:.5f} km\n".format(initScore)
    writeLog(msg)
    msg = "The optimal total traveling distance = {:.5f} km\n".format(score)
    writeLog(msg)

# plot score vs t
plt.figure()
plt.title("traveling time vs Iteration")
ax = plt.gca()
enVsTime = pd.read_csv( "./scoreVSTime.csv") 
plt.plot(enVsTime.timeStep,enVsTime.score,'k-')
plt.minorticks_on()
minorLocatorX = AutoMinorLocator(5) # number of minor intervals per major # inteval
minorLocatorY = AutoMinorLocator(5)
ax.set_xlabel("Iteration",size = 16)
if case == 1 or case == 2 or case == 3:
    ax.set_ylabel("Total traveling time (min)",size = 16)
else:
    ax.set_ylabel("Total traveling distance (km)",size = 16)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
plt.grid(True)
#plt.xlim(-20000,500000)
plt.savefig("fig1.eps")
plt.show()   

scoreCheck = distance(randomList, rCoor)
if case == 1 or case == 2 or case == 3:
    msg = "The checked optimal total traveling time = {:.5f} min".format(scoreCheck)
    writeLog(msg)
else:
    msg = "The checked optimal total traveling distance = {:.5f} km".format(scoreCheck)
    writeLog(msg)