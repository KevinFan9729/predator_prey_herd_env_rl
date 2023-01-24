import pygame
import random
import math
import numpy as np

pygame.init()
# WIDTH, HEIGHT= 640,480
WIDTH, HEIGHT= 1024,640
# WIDTH, HIGHET= 2048,1080
WIN =pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Predator and Prey")

AGENT_SIZE=20
CLOCK_TICK=200#limit frame rate
PREY_SPEED=1.5
MANUAL_SPEED=10
RL_SPEED=3
# RL_SPEED=3
PREDATOR_SPEED=1
MAX_SPEED=5

#RGB COLOR 
WHITE=(255,255,255)
RED=(200,0,0)
BLUE1=(0,0,255)
BLUE2=(0,100,255)
BLACK=(0,0,0)

font=pygame.font.Font('arial.ttf',10)
scoreFont=pygame.font.Font('arial.ttf',25)
random.seed(15)


class preyAgent():
    __lastID = 0#class variable
    def __init__(self):
        #set initial position of the agent, random location at quadrant IV
        # self.x=random.randint(WIDTH/2,WIDTH-AGENT_SIZE)
        # self.y=random.randint(HEIGHT/2,HEIGHT-AGENT_SIZE)
        spawnChoice=random.choice([1,2])
        if spawnChoice==1:#spwan at quadrant I,II
            self.x=random.randint(AGENT_SIZE,WIDTH-AGENT_SIZE)
            self.y=random.randint(AGENT_SIZE,HEIGHT/2)
        else:#spwan at quadrant III, IV
            # self.x=random.randint(WIDTH/2,WIDTH-AGENT_SIZE)
            # self.y=random.randint(HEIGHT/2,HEIGHT-AGENT_SIZE) IV
            self.x=random.randint(AGENT_SIZE,WIDTH-AGENT_SIZE)
            self.y=random.randint(HEIGHT/2,HEIGHT-AGENT_SIZE)
        self.rect=pygame.Rect(self.x,self.y,AGENT_SIZE,AGENT_SIZE)
        self.velocityX=random.randint(1, 1)/10.0
        self.velocityY=random.randint(1, 1)/10.0
        self.id=preyAgent.__lastID
        self.direction=random.choice(('LEFT','RIGHT','UP','DOWN'))
        preyAgent.__lastID+=1
        
    def manualControl(self):
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                quit()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                self.x-=MANUAL_SPEED
            if keys[pygame.K_RIGHT]:
                self.x+=MANUAL_SPEED
            if keys[pygame.K_UP]:
                self.y-=MANUAL_SPEED
            if keys[pygame.K_DOWN]:
                self.y+=MANUAL_SPEED
    def autoRun(self):
        # self.x-=1.5# at this speed predator can catch the prey in a stright chase
        self.x-=2# at this speed predator very close to the predator speed
    def flee(self,predator):
        dist=distance(predator.x,predator.y,self.x,self.y)
        if dist<75:
            dx=self.rect.centerx-predator.rect.centerx
            dy=self.rect.centery-predator.rect.centery
            dx/=dist#normalize directional vector
            dy/=dist
            self.x+=PREY_SPEED*dx
            self.y+=PREY_SPEED*dy
        # else:
        #     self.x+=random.randint(-5, 5)
        #     self.y+=random.randint(-5, 5)
    def moveRLprey(self,action):
        if np.array_equal(action, [1,0,0,0]):#MOVE RIGHT
            self.x+=RL_SPEED
            self.direction='RIGHT'
        elif np.array_equal(action, [0,1,0,0]):#MOVE LEFT
            self.x-=RL_SPEED
            self.direction='LEFT'
        elif np.array_equal(action, [0,0,1,0]):#MOVE UP
            self.y-=RL_SPEED
            self.direction='UP'
        elif np.array_equal(action, [0,0,0,1]):#MOVE DOWN
            self.y+=RL_SPEED
            self.direction='DOWN'
    # def moveRLprey(self,action):
    #     # 0,1,2,3
    #     clockWise=['RIGHT','DOWN','LEFT','UP']
    #     idx=clockWise.index(self.direction)#get the current direction

    #     if np.array_equal(action, [1,0,0,0]):#go stright
    #         newDirect=clockWise[idx] #no change in direction
    #     elif np.array_equal(action, [0,1,0,0]):#right turn
    #         nextIdx=(idx+1)%4
    #         newDirect=clockWise[nextIdx]
    #     elif np.array_equal(action, [0,0,1,0]):#left turn
    #         nextIdx=(idx-1)%4
    #         newDirect=clockWise[nextIdx]
    #     else: #[0,0,0,1] trun back
    #         nextIdx=(idx+2)%4
    #         newDirect=clockWise[nextIdx]
        
    #     self.direction =newDirect

    #     if self.direction=='RIGHT':#MOVE RIGHT
    #         self.x+=RL_SPEED
    #     elif self.direction=='LEFT':#MOVE LEFT
    #         self.x-=RL_SPEED
    #     elif self.direction=='UP':#MOVE UP
    #         self.y-=RL_SPEED
    #     else: #MOVE DOWN
    #         self.y+=RL_SPEED



class predatorAgent():
    def __init__(self):
        #set initial position of the agent, random location at quadrant III
        # self.x=random.randint(AGENT_SIZE,WIDTH/2)
        # self.y=random.randint(HEIGHT/2,HEIGHT-AGENT_SIZE)
        self.x=random.randint(AGENT_SIZE,WIDTH-AGENT_SIZE)
        self.y=random.randint(AGENT_SIZE,HEIGHT-AGENT_SIZE)
        self.rect=pygame.Rect(self.x,self.y,AGENT_SIZE,AGENT_SIZE)
        self.killDelay=0
        self.pause=False
        self.isAvailable=True

    #logic to chase prey agent
    def findClosestPrey(self, visiablePreyList):
        d=[]
        for prey in visiablePreyList:
            d.append(distance(self.rect.centerx,self.rect.centery,prey.rect.centerx,prey.rect.centery))
        dist=-1
        cloestPreyIndex=-1
        if(len(d)>0):
            dist=min(d)
            cloestPreyIndex=d.index(dist)
        return (cloestPreyIndex,dist)
        
    def chase(self,visiablePreyList,capture=(-1,-1)):
        preySelect=self.findClosestPrey(visiablePreyList)
        preyIndex=preySelect[0]
        if capture[0]==-1 or capture[0]==0:#not capture the prey yet
            if preyIndex!=-1:
                prey=visiablePreyList[preyIndex]
                dx=prey.rect.centerx-self.rect.centerx
                dy=prey.rect.centery-self.rect.centery
                # dist=distance(self.rect.centerx,self.rect.centery,prey.rect.centerx,prey.rect.centery)
                dist=preySelect[1]
                dx/=dist#normalize directional vector
                dy/=dist
                if not(self.pause):#give chase
                    self.isAvailable=True
                    self.x+=PREDATOR_SPEED*dx
                    self.y+=PREDATOR_SPEED*dy
     
            if(self.killDelay>200):
                self.killDelay=0
                self.pause=False
            if(self.pause==True):
                self.killDelay+=1  
        elif capture[0]==1:# a prey is captured
            self.pause=True
            # print((PREDATOR_SPEED*dx,PREDATOR_SPEED*dy))
            
            
class simulator():
    def __init__(self):
        self.clock=pygame.time.Clock()#simulation clock
        self.reset()
       
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
                        # print("check "+str((prey.x,prey.y))+" with "+str((agent.x,agent.y)))
                        if((abs(agent.x-prey.x)<=AGENT_SIZE) and (abs(agent.y-prey.y)<=AGENT_SIZE)):
                            #print("remake"+str((prey.x,prey.y))+"becasue of "+str((agent.x,agent.y)))
                            prey=preyAgent() #remkae agent
                            noverlapFlag=False
                            break
                        noverlapFlag=True #no overlapping agents
            pygame.draw.rect(WIN,BLUE1,prey.rect)
            self.preyList.append(prey)
            # print((prey.x,prey.y, i))


    #render agents on screen
    def step(self,action):
        # if len(self.preyList)!=0:
        #     rlagentPredDist=distance(self.preyList[0].x,self.preyList[0].x,self.predatorList[0].x,self.predatorList[0].y)
        #     reward=self.numPrey*(self.frameIter/1000)*(rlagentPredDist/100)
        # else:
        #     reward=0
        # reward=self.frameIter/100
        reward=0.03
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        over=False
        if len(self.preyList)==0 or self.frameIter>6000:
            # if len(self.preyList)==0:
            #     reward-=20
            # if self.frameIter<800:
            #     reward-=10
            # if self.frameIter>1600:
            #     reward+=10

            reward+=self.frameIter/600

            # if self.frameIter>300:
            #     reward=5+self.frameIter/25
            # if self.frameIter>400:
            #     reward=10+self.frameIter/25
            # if self.frameIter>6000:
            #     reward+=20

            over=True
            return reward,over,self.frameIter,self.score
        # target = pygame.mouse.get_pos()
        # self.preyList[0].manualControl()
        self.preyList[0].moveRLprey(action)
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
            if(self.predatorList[0].isAvailable==True):
                if (collisionCheck[1]==0):
                    reward-=5
                else:
                    reward-=2
                self.preyList.pop(collisionCheck[1])
                self.numPrey-=1
                self.predatorList[0].isAvailable=False
            else:
                self.bounceBack(0,collisionCheck[1],1)

        if (collisionCheck[0]==0):#prey-prey collision
            self.bounceBack(collisionCheck[1],collisionCheck[2],0)
            reward-=1
        if (boundary!=-1):#collision with the boundary
            if(boundary==0):#rl controlled agent hit the wall
                reward-=5
            else:
                 reward-=2
            try:
                self.preyList.pop(boundary)
                self.numPrey-=1
            except IndexError:
                print("error at boundary check")
                print("prey list length:")
                print(len(self.preyList))
                print("try to pop:")
                print(boundary)
        self.updateUI()
        self.clock.tick(CLOCK_TICK)
        self.frameIter+=1
        # w, h = pygame.display.get_surface().get_size()
        # print((w,h))
        # print(self.score)
        return reward,over,self.frameIter,self.score

    def updateUI(self):
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
        socreText=scoreFont.render("Frame Survived: "+str(self.frameIter),True,WHITE)
        WIN.blit(socreText,[0,0])
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
    def bounceBack(self,index1,index2,mode=0):
        #mode 0: only check prey collide with prey
        #mode 1: check prey collide with a prey
        #other rect, moving rect
        #print((index1,index2))
        # print((self.preyList[index1].x,self.preyList[index1].y))
        tolerance=15
        if mode==0:
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
        else:#mode=1, predator bounceback physics
            if abs(self.preyList[index2].rect.top-self.predatorList[index1].rect.bottom)<tolerance: #index1 predator bottom collison
                self.preyList[index2].y+=20 #move down a bit
            if abs(self.preyList[index2].rect.bottom-self.predatorList[index1].rect.top)<tolerance: #index1 predator top collison
                self.preyList[index2].y-=20 #move up a bit
            if abs(self.preyList[index2].rect.right-self.predatorList[index1].rect.left)<tolerance: #index1 predator left collison
                self.preyList[index2].x-=20 #move left a bit
            if abs(self.preyList[index2].rect.left-self.predatorList[index1].rect.right)<tolerance: #index1 predator right collison
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
                

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)      


# def main():
#     over=False
#     sim=simulator()
#     while not over:
#         reward,over,score=sim.step()
#         # if pygame.event.get().type == pygame.QUIT:
#         #     run=False
#         # for event in pygame.event.get():
#         #         if event.type == pygame.QUIT:
#         #             run=False
        
#     pygame.quit()
#     print("Final Score: "+ str(int(score)))

# if __name__=="__main__":
#     main()