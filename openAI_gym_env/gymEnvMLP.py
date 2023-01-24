import gym
from gym import spaces
import pygame
import random
import math
import numpy as np
from env import preyAgent,predatorAgent
from pygame.surfarray import array3d
import cv2

pygame.init()
WIDTH, HEIGHT= 640,480
WIN =pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Predator and Prey")

AGENT_SIZE=20
CLOCK_TICK=200#limit frame rate
PREY_SPEED=1.5
MANUAL_SPEED=10
# RL_SPEED=2
RL_SPEED=2
PREDATOR_SPEED=2
MAX_SPEED=5

#RGB COLOR 
WHITE=(255,255,255)
# RED=(200,0,0)
RED=(255,255,255)
BLUE1=(0,0,255)
BLUE2=(0,100,255)
BLACK=(0,0,0)

font=pygame.font.Font('arial.ttf',10)
scoreFont=pygame.font.Font('arial.ttf',25)
random.seed(15)

class predatorPreyCustomEnv(gym.Env):
    def __init__(self):
        super(predatorPreyCustomEnv, self).__init__()
        self.clock=pygame.time.Clock()#simulation clock
        n_actions=4
        self.action_space = spaces.Discrete(n_actions)#4 actions
        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(16,), dtype=np.float32)#input image as observation
    def reset(self):
        self.frameIter=0
        self.numPrey=1
        self.numPredator=1
        self.score=0#recording gobal score
        self.preyList=[]
        self.predatorList=[]
        #initial placement of the prey and predator
        for i in range(0,self.numPredator):
            predator=predatorAgent()
            pygame.draw.rect(WIN,RED,predator.rect)
            self.predatorList.append(predator)
        for i in range(0,self.numPrey):
            noverlapFlag=False
            prey=preyAgent()
            if(len(self.preyList)>=1):
                while noverlapFlag==False:
                    for agent in self.preyList:
                        if((abs(agent.x-prey.x)<=AGENT_SIZE) and (abs(agent.y-prey.y)<=AGENT_SIZE)):
                            prey=preyAgent() #remkae agent
                            noverlapFlag=False
                            break
                        noverlapFlag=True #no overlapping agents
            pygame.draw.rect(WIN,BLUE1,prey.rect)
            self.preyList.append(prey)
            image=array3d(pygame.display.get_surface())
            state=self.getStateMLP()
        # return self.imageProcess(image)
        return state
    def step(self,action):
        reward=0.03
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()
        over=False
        state=self.getStateMLP()
        if len(self.preyList)==0 or self.frameIter>10000:
            image=array3d(pygame.display.get_surface())
            state=self.getStateMLP()
            reward+=self.frameIter/600
            over=True
            info={'frameIter': self.frameIter}
            
            return state,reward,over,info

        if action==0:
            actionPrey=[1,0,0,0]
        elif action==1:
            actionPrey=[0,1,0,0]
        elif action==2:
            actionPrey=[0,0,1,0]
        elif action==3:
            actionPrey=[0,0,0,1]
        self.preyList[0].moveRLprey(actionPrey)
        target=self.targetGen(self.preyList[0])
        self.movePreyBoid(target)
        # preySelect=self.predatorList[0].findClosestPrey(self.preyList)
        # print("selected prey: "+str(preySelect[0]) +" predator position " +str((self.predatorList[0].x,self.predatorList[0].y)))
        collisionCheck=self.collision()
        boundary=self.boundaryCheck()
        self.updateScore(collisionCheck[0],boundary)
        # print(collisionCheck[0])
        self.predatorList[0].chase(self.preyList,collisionCheck)
        if (collisionCheck[0]==1):#collision with a predator
            if (collisionCheck[1]==0):
                reward-=5
            else:
                reward-=2
            self.preyList.pop(collisionCheck[1])
            self.numPrey-=1
        if (collisionCheck[0]==0):#prey-prey collision
            self.bounceBack(collisionCheck[1],collisionCheck[2])
            reward-=1
        if (boundary!=-1):#collision with the boundary
            if(boundary==0):#rl controlled agent hit the wall
                reward-=3
            else:
                reward-=2
            # try:
            #     self.preyList.pop(boundary)
            #     self.numPrey-=1
            # except IndexError:
            #     print("error at boundary check")
            #     print("prey list length:")
            #     print(len(self.preyList))
            #     print("try to pop:")
            #     print(boundary)
        self.update()
        self.clock.tick(CLOCK_TICK)
        self.frameIter+=1
        # w, h = pygame.display.get_surface().get_size()
        # print((w,h))
        # print(self.score)
        image=array3d(pygame.display.get_surface())
        info={'frameIter': self.frameIter}
        return state,reward,over,info
    def render(self):
        WIN.fill(BLACK)
        count=0
        for prey in self.preyList:
            prey.rect=pygame.Rect(prey.x,prey.y,AGENT_SIZE,AGENT_SIZE)
            pygame.draw.rect(WIN,BLUE1,prey.rect)
            WIN.fill(WHITE, ((prey.x, prey.y), (1, 1)))
            WIN.fill(WHITE, ((prey.x+AGENT_SIZE, prey.y+AGENT_SIZE), (1, 1)))
            text=font.render(str(count),True,WHITE)
            WIN.blit(text,[prey.x,prey.y])
            count=count+1
        for predator in self.predatorList:
            predator.rect=pygame.Rect(predator.x,predator.y,AGENT_SIZE,AGENT_SIZE)
            pygame.draw.rect(WIN,RED, predator.rect)
        # socreText=scoreFont.render("Frame Survived: "+str(self.frameIter),True,WHITE)
        # WIN.blit(socreText,[0,0])
        # pygame.display.flip()
    def close(self):
        pygame.quit()
    def getStateMLP(self):
        if len(self.preyList)==0:
            return np.zeros(16)
        ptRight = (self.preyList[0].rect.centerx  + 2*AGENT_SIZE, self.preyList[0].rect.centery)
        ptLeft = (self.preyList[0].rect.centerx - 2*AGENT_SIZE, self.preyList[0].rect.centery)
        ptUP = (self.preyList[0].rect.centerx, self.preyList[0].rect.centery - 2*AGENT_SIZE)
        pointDown = (self.preyList[0].rect.centerx, self.preyList[0].rect.centery + 2*AGENT_SIZE)

        boundaryDanger=[
        self.preCollisionCheck(ptRight)*1, #boundary right
        self.preCollisionCheck(ptLeft)*1, #boundary left
        self.preCollisionCheck(ptUP)*1, #boundary up
        self.preCollisionCheck(pointDown)*1, #boundary down
        ]
        predatorDanger=[ 
        (self.predatorList[0].rect.centerx > self.preyList[0].rect.centerx)*min((1/max((abs(self.predatorList[0].rect.centerx-self.preyList[0].rect.centerx)/WIDTH),0.01))/2,10)/10,  # predator right
        (self.predatorList[0].rect.centerx < self.preyList[0].rect.centerx)*min((1/max((abs(self.preyList[0].rect.centerx-self.predatorList[0].rect.centerx)/WIDTH),0.01))/2,10)/10,  # predator left
        (self.predatorList[0].rect.centery < self.preyList[0].rect.centery)*min((1/max((abs(self.predatorList[0].rect.centery-self.preyList[0].rect.centery)/HEIGHT),0.01))/2,10)/10, # predator up
        (self.predatorList[0].rect.centery > self.preyList[0].rect.centery)*min((1/max((abs(self.preyList[0].rect.centery-self.predatorList[0].rect.centery)/HEIGHT),0.01))/2,10)/10  # predator down
        ]
        agentDirection=[
            (self.preyList[0].direction=='RIGHT')*1,
            (self.preyList[0].direction=='LEFT')*1,
            (self.preyList[0].direction=='UP')*1,
            (self.preyList[0].direction=='DOWN')*1,
        ]
        rlAgentPos=np.array([self.preyList[0].x/WIDTH,self.preyList[0].y/HEIGHT])#2
        predatorPos=np.array([self.predatorList[0].x/WIDTH,self.predatorList[0].y/HEIGHT])#2
        
        boundaryDanger=np.asarray(boundaryDanger)#4
        predatorDanger=np.asarray(predatorDanger)#4
        agentDirection=np.asarray(agentDirection)#4
        state=np.concatenate((boundaryDanger,predatorDanger,agentDirection,rlAgentPos,predatorPos))
        return state

    def update(self):
        WIN.fill(BLACK)
        count=0
        for prey in self.preyList:
            prey.rect=pygame.Rect(prey.x,prey.y,AGENT_SIZE,AGENT_SIZE)
            pygame.draw.rect(WIN,BLUE1,prey.rect)
            WIN.fill(WHITE, ((prey.x, prey.y), (1, 1)))
            WIN.fill(WHITE, ((prey.x+AGENT_SIZE, prey.y+AGENT_SIZE), (1, 1)))
            text=font.render(str(count),True,WHITE)
            WIN.blit(text,[prey.x,prey.y])
            count=count+1
        for predator in self.predatorList:
            predator.rect=pygame.Rect(predator.x,predator.y,AGENT_SIZE,AGENT_SIZE)
            pygame.draw.rect(WIN,RED, predator.rect)
        # socreText=scoreFont.render("Frame Survived: "+str(self.frameIter),True,WHITE)
        # WIN.blit(socreText,[0,0])
        pygame.display.flip()
    def collision(self):
        # return a tuple (collisionNature,collisionIndex) or (collisionNature, collisionIndex1, collisionIndex2)
        # collisionNature==1 means collision with predator 
        # collisionNature==0 means collision with a prey
        # NOTE: prey id may not equal to the index 
        collision=False
        index=0
        for prey in self.preyList:
            collision=self.predatorList[0].rect.colliderect(prey.rect)
            if collision:
                # print((1,index))
                return (1,index)
            index+=1
        index=0
        while index<self.numPrey:
            for prey in self.preyList:
                #print("index: "+str(index))
                # print("id: "+str(prey.id))
                if self.preyList[index].id!=prey.id:
                    collision=self.preyList[index].rect.colliderect(prey.rect)
                # if prey.id!=self.preyList[9].id:
                #     collision=self.preyList[9].rect.colliderect(prey.rect)
                if collision:
                    #print((index,prey.id,0))
                    # print((0,index,self.preyList.index(prey)))#(collide with a non-predator, moving prey index, colliding with)
                    return (0,index,self.preyList.index(prey))
            index+=1
        return (-1,-1)
    def bounceBack(self,index1,index2):
        #only check prey collide with prey
        #other rect, moving rect
        #print((index1,index2))
        # print((self.preyList[index1].x,self.preyList[index1].y))
        tolerance=15
        if abs(self.preyList[index2].rect.top-self.preyList[index1].rect.bottom)<tolerance: #index1 prey bottom collison
            # self.preyList[index1].y-=10 #move up a bit
            self.preyList[index2].y+=20 #move down a bit
        if abs(self.preyList[index2].rect.bottom-self.preyList[index1].rect.top)<tolerance: #index1 prey top collison
            # self.preyList[index1].y+=10 #move down a bit
            self.preyList[index2].y-=20 #move up a bit
        if abs(self.preyList[index2].rect.right-self.preyList[index1].rect.left)<tolerance: #index1 prey left collison
            # self.preyList[index1].x+=10 #move right a bit
            self.preyList[index2].x-=20 #move left a bit
        if abs(self.preyList[index2].rect.left-self.preyList[index1].rect.right)<tolerance: #index1 prey right collison
            # self.preyList[index1].x-=10 #move left a bit
            self.preyList[index2].x+=20 #move right a bit
    def preCollisionCheck(self,pt):
        #helper function for checking pre-collision
        #pt: point represented by a tuple
        if pt[0]>WIDTH or pt[0]<0 or pt[1]>HEIGHT or pt[1]<0:#hits boundary
            return True
        return False
    def updateScore(self,collisionNature,boundaryCheck):#record the score
        # print(self.numPrey)
        self.score+=(self.numPrey/100)
        if collisionNature==1:#predator collision
            self.score-=10
        if collisionNature==0:#prey collision
            self.score-=2
        if boundaryCheck!=-1:
            self.score-=5
    def boundaryCheck(self):
        #check for boundary and bounce back physics if the agent hits the boundary
        for prey in self.preyList:
            if prey.rect.right>=WIDTH:#RIGHT
                prey.x-=10
                # print('right')
                return self.preyList.index(prey)
            if prey.rect.left<=0:#LEFT
                prey.x+=10
                # print('left')
                return self.preyList.index(prey)
            if prey.rect.top<=0:#TOP
                prey.y+=10
                # print('top')
                return self.preyList.index(prey)
            if prey.rect.bottom>=HEIGHT:#DOWN
                prey.y-=10
                # print('down')
                return self.preyList.index(prey)
        return -1
    def movePreyBoid(self, target):
        if self.numPrey>1:
            for prey in self.preyList[1:]:
                v1=self.centerOfMassRule(prey)
                v2=self.seperationRule(prey)
                v3=self.matchVelocity(prey)
                prey.velocityX+=v1[0]+v2[0]+v3[0]
                prey.velocityY+=v1[1]+v2[1]+v3[1]
                self.goal(prey,target[0],target[1])
                self.limitVel(prey)
                prey.x+= prey.velocityX
                prey.y+= prey.velocityY
            # print((prey.velocityX,prey.velocityY))

    def centerOfMassRule(self, boid):
        coMass=np.array([0,0])
        for prey in self.preyList[1:]:#not including the first element, first element is AI controlled
            if prey.id!=boid.id:
                coMass+=np.array([prey.x,prey.y],dtype=np.int32)
        if self.numPrey==2:
             coMass=coMass/(self.numPrey-1)
        else:
            coMass=coMass/(self.numPrey-2)
        return (coMass-np.array([boid.x,boid.y]))/100 #move each boid 1% toward the center of mass

    def seperationRule(self, boid, dist=50):
        c=np.array([0,0])
        for prey in self.preyList[1:]:
            if prey.id!=boid.id:
                if distance(prey.x,prey.y,boid.x,boid.y)<dist:
                    c=c-(np.array([prey.x,prey.y])-np.array([boid.x,boid.y]))
        return c
    
    def matchVelocity(self, boid):
        velocity= np.array([0.0,0.0])
        for prey in self.preyList[1:]:
            if prey.id!=boid.id:
                velocity+=np.array([prey.velocityX,prey.velocityY])
        if self.numPrey==2:
            velocity=velocity/(self.numPrey-1)
        else:
            velocity=velocity/(self.numPrey-2)
        return (velocity-np.array([boid.velocityX,boid.velocityY]))/8

    def goal(self,boid, x, y):
        boid.velocityX += (x-boid.rect.centerx)/100
        boid.velocityY += (y-boid.rect.centery)/100
    def limitVel(self,boid):
        vlim=3.5
        speed = math.sqrt(boid.velocityX**2 + boid.velocityY**2)
        if speed>vlim:
            boid.velocityX=(boid.velocityX/speed)*vlim
            boid.velocityY=(boid.velocityY/speed)*vlim
        pass
    def targetGen(self,leader):
        seperationFactor=2.5
        centerPos=np.array([0,0])
        for prey in self.preyList[1:]:
            if not math.isnan(prey.x):
                centerPos+=np.array([prey.x,prey.y],dtype=np.int32)
            else:
                return (0,0)
        if self.numPrey>1:
            centerPos=centerPos/(self.numPrey-1)  
        #4 possibilities 
        #1 leader is at the right down side of the herd
        if (centerPos[0]<leader.rect.centerx and centerPos[1]<leader.rect.centery):
            target=(leader.rect.centerx-seperationFactor*AGENT_SIZE,leader.rect.centery-seperationFactor*AGENT_SIZE)
        #2 leader is at the left down side of the herd
        elif(centerPos[0]>leader.rect.centerx and centerPos[1]<leader.rect.centery):
            target=(leader.rect.centerx+seperationFactor*AGENT_SIZE,leader.rect.centery-seperationFactor*AGENT_SIZE)
        #3 leader is at the left up side of the herd
        elif(centerPos[0]>leader.rect.centerx and centerPos[1]>leader.rect.centery):
            target=(leader.rect.centerx+seperationFactor*AGENT_SIZE,leader.rect.centery+seperationFactor*AGENT_SIZE)
        # leader is at the right up side of the herd
        else:
            target=(leader.rect.centerx-2*AGENT_SIZE,leader.rect.centery+3*AGENT_SIZE)
        return target
    def imageProcess(self,image):
        processedImage = cv2.cvtColor(cv2.resize(image, (160, 120)), cv2.COLOR_BGR2GRAY)
        processedImage=np.reshape(processedImage, (120,160,1))
        return processedImage

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)   