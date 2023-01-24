from matplotlib.style import available
import torch
import random
import numpy as np
from collections import deque
from env import WIDTH,HEIGHT, simulator,distance
from model import DeepQNet, trainer
from plot import plot, plotWindowed

#800
#500

random.seed(15)

MAX_MEMORY=1000_000
BATCH_SIZE=10000
LR=0.001
MAX_PREY=10
AGENT_SIZE=20

class Agent:
    def __init__(self):
        self.simNum=0
        self.epsilon=0.95
        self.epsilon0=self.epsilon
        self.gamma=0.99#discount factor
        self.memory=deque(maxlen=MAX_MEMORY)#automatically popleft() if the size exceeds MAX_MEMORY
        self.model=DeepQNet(12,512,4)
        self.trainer=trainer(self.model,LR,gamma=self.gamma)
        if torch.cuda.is_available():
            dev="cuda:0"
            # dev="cpu"
        else:
            dev="cpu"
        self.device = torch.device(dev) 

    
    # def getState(self,env):
    #     preyList=env.preyList
    #     predator=env.predatorList[0]
    #     numPrey=env.numPrey
    #     idx=0
    #     jdx=0
    #     preyPos=np.zeros(MAX_PREY*2)#20
    #     preyPredDist=np.zeros(MAX_PREY)#10
    #     preyPreyDist=np.zeros(45)#45
    #     for prey in preyList:
    #         preyPos[jdx]=prey.x
    #         preyPos[jdx+1]=prey.y
    #         preyPredDist[idx]=distance(prey.x,prey.y,predator.x,predator.y)
    #         idx+=1
    #         jdx+=2
    #     count=0
    #     for i in range(0,len(preyList)):
    #         for j in range(i+1,len(preyList)):
    #             preyPreyDist[count]=distance(preyList[i].x,preyList[i].y,preyList[j].x,preyList[j].y)
    #             count+=1
    #     predatorPos=np.array([predator.x,predator.y])#2
        
    #     #size: 20+10+45+2+1=78
    #     state=np.concatenate((preyPos,preyPredDist,preyPreyDist,predatorPos,np.array([numPrey])))
    #     return state

    # def getState(self,env):
    #     preyList=env.preyList
    #     if len(preyList)==0:
    #         return np.zeros(15)
    #     predator=env.predatorList[0]
    #     numPrey=env.numPrey
    #     rlAgent=preyList[0]
    #     idx=0
    #     herdCenter=np.array([0,0])
    #     agentPreyDist=np.zeros(MAX_PREY-1)#9

    #     for prey in preyList:
    #         herdCenter[0]+=prey.x
    #         herdCenter[1]+=prey.y
    #         if prey.id!=rlAgent.id:
    #             agentPreyDist[idx]=distance(rlAgent.x,rlAgent.y,prey.x,prey.y)
    #             idx+=1
    #     herdCenter=herdCenter/numPrey
    #     herdPredDist=distance(herdCenter[0],herdCenter[1],predator.x,predator.y)#1
    #     rlAgentPos=np.array([rlAgent.x,rlAgent.y])#2
    #     predatorPos=np.array([predator.x,predator.y])#2
        
    #     #size: 2+2+9+2=15
    #     state=np.concatenate((rlAgentPos,predatorPos,agentPreyDist,np.array([herdPredDist,numPrey])))
    #     return state

    def getState(self,env):
        preyList=env.preyList
        if len(preyList)==0:
            return np.zeros(12)
        predator=env.predatorList[0]
        # numPrey=env.numPrey
        rlAgent=preyList[0]

        ptRight = (rlAgent.rect.centerx  + 2*AGENT_SIZE, rlAgent.rect.centery)
        ptLeft = (rlAgent.rect.centerx - 2*AGENT_SIZE, rlAgent.rect.centery)
        ptUP = (rlAgent.rect.centerx, rlAgent.rect.centery - 2*AGENT_SIZE)
        pointDown = (rlAgent.rect.centerx, rlAgent.rect.centery + 2*AGENT_SIZE)

        

        rlAgentPos=np.array([rlAgent.x,rlAgent.y])#2
        predatorPos=np.array([predator.x,predator.y])#2

        boundaryDanger=[
        env.preCollisionCheck(ptRight)*1, #boundary right
        env.preCollisionCheck(ptLeft)*1, #boundary left
        env.preCollisionCheck(ptUP)*1, #boundary up
        env.preCollisionCheck(pointDown)*1, #boundary down
        ]

        predatorDanger=[ 
        (predator.rect.centerx > rlAgent.rect.centerx)*min((1/max((abs(predator.rect.centerx-rlAgent.rect.centerx)/WIDTH),0.01))/2,10)/10,  # predator right
        (predator.rect.centerx < rlAgent.rect.centerx)*min((1/max((abs(rlAgent.rect.centerx-predator.rect.centerx)/WIDTH),0.01))/2,10)/10,  # predator left
        (predator.rect.centery < rlAgent.rect.centery)*min((1/max((abs(predator.rect.centery-rlAgent.rect.centery)/HEIGHT),0.01))/2,10)/10, # predator up
        (predator.rect.centery > rlAgent.rect.centery)*min((1/max((abs(rlAgent.rect.centery-predator.rect.centery)/HEIGHT),0.01))/2,10)/10  # predator down
        ]

        agentDirection=[
            (rlAgent.direction=='RIGHT')*1,
            (rlAgent.direction=='LEFT')*1,
            (rlAgent.direction=='UP')*1,
            (rlAgent.direction=='DOWN')*1,
        ]
        # print(distance(rlAgent.x,rlAgent.y,predator.x,predator.y))
        boundaryDanger=np.asarray(boundaryDanger)#4
        predatorDanger=np.asarray(predatorDanger)#4
        agentDirection=np.asarray(agentDirection)#4

        #size: 4+4+4=12
        state=np.concatenate((boundaryDanger,predatorDanger,agentDirection))
        # print(predatorDanger)
        # print(state)
        # boundaryCheck=state[:4]
        # predatorCheck=state[4:8]
        # print(boundaryDanger)
        # print(state.shape)
        return state

        
    def remember(self,state, action, reward, nextState, over):
        self.memory.append((state,action,reward,nextState,over))
   
    def trainLongMemory(self):
        if len(self.memory)>BATCH_SIZE:
            sample=random.sample(self.memory,BATCH_SIZE)#randomly sample memory buffer if we have enough memory
        else:
            sample=self.memory#take the whole memory if the memory is not big enough
        #zip all states, actions, rewards... into their respective groups
        states,actions,rewards,nextStates,overs=zip(*sample)
        states=np.asarray(states)
        actions=np.asarray(actions)
        rewards=np.asarray(rewards)
        nextStates=np.asarray(nextStates)
        overs=np.asarray(overs)
        self.trainer.trainStep(states,actions,rewards,nextStates,overs,self.simNum)
    
    def trainShortMemory(self,state,action,reward,nextState,over):
        self.trainer.trainStep(state,action,reward,nextState,over,self.simNum)
    
    # def getAction(self,state):#annealling greedy epsilon expolration
    #     decreaseConstant=250
    #     finalAction=[0,0,0,0]
    #     #0      1    2   3
    #     #right left up down
    #     #boundary, predator direction
        
    #     if random.random()<self.epsilon:#expolration
    #         if random.random()<0.35:#guided exploration
    #             boundaryCheck=state[:4]
    #             predatorCheck=state[4:8]
    #             availableDirCheck=boundaryCheck-predatorCheck
    #             # availableDirCheck=boundaryCheck
    #             # print(availableDirCheck)
    #             availableDir=[i for i, x in enumerate(availableDirCheck) if x == 0]
    #             # print(availableDir)
    #             if len(availableDir)==0:
    #                 escape=[i for i, x in enumerate(boundaryCheck) if x == 0]
    #                 # print('escape',escape)
    #                 idx=random.choice(escape)
    #                 # print(availableDir)
    #             else:
    #                 idx=random.choice(availableDir)
                
    #             # finalAction[idx]=1
    #         else:#random exploration with boundary check 
    #             boundaryCheck=state[:4]
    #             availableDir=[i for i, x in enumerate(boundaryCheck) if x == 0]
    #             idx=random.choice(availableDir)
    #             # idx=random.randint(0,3)
    #         # print(availableDir)
    #         finalAction[idx]=1
    #         # print("expolre")
    #     else:
    #         state0=torch.tensor(state,dtype=torch.float).to(self.device)
    #         pred=self.model(state0)#make prediction
    #         idx=torch.argmax(pred).item()
    #         finalAction[idx]=1
    #         # print("exploit")
    #     self.epsilon=self.epsilon0/(1+self.simNum/decreaseConstant)
    #     return finalAction
    
    def getAction(self,state):#annealling greedy epsilon expolration
        # decreaseConstant=250
        decreaseConstant=800
        finalAction=[0,0,0,0]
        # 0      1    2   3
        # right left up down
        # boundary, predator direction
        if random.random()<self.epsilon:#expolration
            idx=random.randint(0,3)
            # boundaryCheck=state[:4]
            # availableDir=[i for i, x in enumerate(boundaryCheck) if x == 0]
            # idx=random.choice(availableDir)
            # print(availableDir)
            finalAction[idx]=1
            # print("explore")
            # if random.random()<0.05:#guided exploration
            #     boundaryCheck=state[:4]
            #     predatorCheck=state[4:8]
            #     availableDir=[i for i, x in enumerate(boundaryCheck) if x == 0]
            #     if len(availableDir)==4:
            #         escape=[i for i, x in enumerate(predatorCheck) if x == 0]
            #         # print('escape',escape)
            #         idx=random.choice(escape)
            #         # print(availableDir)
            #     else:
            #         idx=random.choice(availableDir)
            # else:#random exploration with boundary check 
            #     boundaryCheck=state[:4]
            #     availableDir=[i for i, x in enumerate(boundaryCheck) if x == 0]
            #     idx=random.choice(availableDir)
            #     # idx=random.randint(0,3)
            # finalAction[idx]=1
            # # print("expolre")
        else:
            state0=torch.tensor(state,dtype=torch.float).to(self.device)
            pred=self.model(state0)#make prediction
            idx=torch.argmax(pred).item()
            finalAction[idx]=1
            # print("exploit")
        if self.epsilon>=0.05:
            self.epsilon=self.epsilon0/(1+self.simNum/decreaseConstant)
        return finalAction

    # def getAction(self, state):
    #     # random moves: tradeoff exploration / exploitation
    #     self.epsilon = 180 - self.simNum
    #     final_move = [0,0,0,0]
    #     if random.randint(0, 200) < self.epsilon:
    #         move = random.randint(0, 3)
    #         final_move[move] = 1
    #     else:
    #         state0 = torch.tensor(state, dtype=torch.float).to(self.device)
    #         prediction = self.model(state0)
    #         move = torch.argmax(prediction).item()
    #         final_move[move] = 1
    #         # print(self.epsilon)
    #     return final_move

        
def train():
    plotFramePt=[]
    plotFrameMean=[]
    totalFrame=0
    highFramePt=0

    plotScore=[]
    plotMean=[]
    totalScore=0
    highScore=0

    cumulativeReward=0
    plotCumulativeReward=[]
    totalReward=0
    plotMeanReward=[]

    windowedFrame=[]
    plotWindowedFrame=[]
    plotMeanWinFrame=[]

    saveFlag=0

    agent=Agent()
    env=simulator()
    while True:
        #get old state
        stateOld=agent.getState(env)
        #get move
        move=agent.getAction(stateOld)
        #perform move action and get new state 
        reward,over,frameCount,score=env.step(move)
        stateNew=agent.getState(env)
        #train short memory
        agent.trainShortMemory(stateOld,move,reward,stateNew,over)
        #remeber
        agent.remember(stateOld,move,reward,stateNew,over)
        # print(reward)
        # print(stateOld)
        # print(move)
        cumulativeReward+=reward
        # print(reward)
        if over:
            #experience replay
            env.reset()
            agent.simNum+=1
            agent.trainLongMemory()
            if score>highScore:
                highScore=score
                saveFlag+=1
                agent.model.save("modelHighScore.pth")
            if frameCount>highFramePt:
                highFramePt=frameCount
                saveFlag+=1
                agent.model.save("modelHighFrameLen.pth")
            
            if saveFlag==2:
                agent.model.save()
            # print(saveFlag)
            saveFlag=0#reset saveFlag
            
            plotScore.append(score)
            totalScore += score
            meanScore = totalScore/agent.simNum
            plotMean.append(meanScore)
            # plot(plotScore, meanScore)

            plotFramePt.append(frameCount)
            totalFrame+=frameCount
            meanFramePt=totalFrame/agent.simNum
            plotFrameMean.append(meanFramePt)

            windowedFrame.append(frameCount)

            plotCumulativeReward.append(cumulativeReward)
            totalReward+=cumulativeReward
            meanReward=totalReward/agent.simNum
            plotMeanReward.append(meanReward)
            # print(cumulativeReward)
            # print(totalReward)
            # print(meanReward)
            cumulativeReward=0
            
            plot(plotFramePt, plotFrameMean,plotScore,plotMean,plotCumulativeReward,plotMeanReward)
            # print("highest frame ",highFramePt)
            print((agent.simNum, agent.epsilon))
        if len(windowedFrame)==20:
            plotWindowedFrame.append(sum(windowedFrame)/len(windowedFrame))
            windowedFrame.clear()
            plotMeanWinFrame.append(sum(plotWindowedFrame)/len(plotWindowedFrame))
            plotWindowed(plotWindowedFrame,plotMeanWinFrame)

if __name__ =="__main__":
    train()
