import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(framePt,meanFramePt,scores, meanScores, reward, meanReward):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())

    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulation')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(meanScores)
    #plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(meanScores)-1, meanScores[-1], str(meanScores[-1]))

    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulation')
    plt.ylabel('Frame Survived')
    plt.plot(framePt)
    plt.plot(meanFramePt)
    plt.text(len(framePt)-1, framePt[-1], str(framePt[-1]))
    plt.text(len(meanFramePt)-1, meanFramePt[-1], str(meanFramePt[-1]))

    plt.figure(3)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Simulation')
    plt.ylabel('Reward')
    plt.plot(reward)
    plt.plot(meanReward)
    plt.text(len(reward)-1, reward[-1], str(reward[-1]))
    plt.text(len(meanReward)-1, meanReward[-1], str(meanReward[-1]))

    plt.show(block=False)
    plt.pause(.1)

def plotWindowed(windowFrame,windowFrameMean):
    plt.figure(4)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Window')
    plt.ylabel('Windowed Frame count')
    plt.plot(windowFrame)
    plt.plot(windowFrameMean)
    plt.text(len(windowFrame)-1, windowFrame[-1], str(windowFrame[-1]))
    plt.text(len(windowFrameMean)-1, windowFrameMean[-1], str(windowFrameMean[-1]))
