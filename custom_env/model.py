import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

varTest=[]

class DeepQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        # self.linear1 = nn.Linear(inputSize, hiddenSize)
        # self.linear2 = nn.Linear(hiddenSize, outputSize)
        
        self.model=nn.Sequential(
            nn.Linear(inputSize,hiddenSize),
            nn.ReLU(),
            nn.Linear(hiddenSize,int(hiddenSize)),
            nn.ReLU(),
            nn.Linear(hiddenSize,int(hiddenSize)),
            nn.ReLU(),
            nn.Linear(int(hiddenSize),outputSize),
            # nn.Softmax()
        )
        # self.inputLayer=nn.Linear(inputSize,hiddenSize)
        # self.activation=nn.ReLU()
        # self.h1=nn.Linear(hiddenSize,int(hiddenSize/2))
        # self.activation=nn.ReLU()
        # self.h2=nn.Linear(int(hiddenSize/2),int(hiddenSize/4))
        # self.activation=nn.ReLU()
        # self.outputLayer=nn.Linear(int(hiddenSize/4),outputSize)
        # self.softMax=nn.Softmax()

    def forward(self,x):
        # x = F.relu(self.linear1(x))
        # x = self.linear2(x)
        # return x
        # x=self.inputLayer(x)
        # x=self.activation(x)
        # x=self.h1(x)
        # x=self.activation(x)
        # x=self.h2(x)
        # x=self.activation(x)
        # x=self.outputLayer(x)
        compute=self.model(x)
        return compute
    
    def save(self, fileName='model.pth'):
        folderPath='./model'
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        fileName=os.path.join(folderPath,fileName)
        torch.save(self.state_dict(),fileName)

class trainer:
    def __init__(self,model,lr,gamma):
        if torch.cuda.is_available():
            dev="cuda:0"
        else:
            dev="cpu"
        self.device = torch.device(dev) 
        self.model=model.to(self.device)
        self.targetNet=model.to(self.device)#target network
        self.lr=lr
        self.gamma=gamma
        self.optimizer=optim.Adam(model.parameters(), lr=self.lr)
        self.criterion=nn.MSELoss().to(self.device)
        # self.criterion=nn.HuberLoss().to(self.device)
        # self.criterion=nn.CrossEntropyLoss().to(self.device)

    def trainStep(self,state,action,reward,nextState,over,numSim):
        #(n,x) shape
        state=torch.tensor(state,dtype=torch.float)
        action=torch.tensor(action,dtype=torch.long)
        reward=torch.tensor(reward,dtype=torch.float)
        nextState=torch.tensor(nextState,dtype=torch.float)

        
        if len(state.shape)==1:
            #add one dimension at dim 0 to make shape consistent
            state=torch.unsqueeze(state,0)
            action=torch.unsqueeze(action,0)
            reward=torch.unsqueeze(reward,0)
            nextState=torch.unsqueeze(nextState,0)
            over=(over,)
        
        # state=F.normalize(state)
        # nextState=F.normalize(nextState)
        # print(state)
        
        state=state.to(self.device)
        action=action.to(self.device)
        reward=reward.to(self.device)
        nextState=nextState.to(self.device)

        pred=self.model(state)#4 predictions for 1 index
        # target=pred.clone()
        target=self.targetNet(state)

        for idx in range(len(over)):
            qNew=reward[idx]
            if not over[idx]:
                qNew=reward[idx]+self.gamma*torch.max(self.model(nextState[idx]))
            # print(target[idx])
            target[idx][torch.argmax(action[idx]).item()]=qNew
            # print("new Q")
            # print(qNew)
            # print("target")
            # print(target[idx])
            # print("pred")
            # print(pred[idx])
        
        self.optimizer.zero_grad()
        loss=self.criterion(target,pred)
        loss.backward()
        self.optimizer.step()
        ResidualVariance=torch.var(target-pred)/(torch.var(target))
        varTest.append(ResidualVariance.item())
        if numSim%20==0:
            self.targetNet.load_state_dict(self.model.state_dict())
        # print(ResidualVariance.item())
        # figure(4)
        # plt.plot(varTest)
        # print(loss.item())
