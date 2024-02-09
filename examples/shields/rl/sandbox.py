import minigrid
import gymnasium as gym
import random
import matplotlib.pyplot as plt

from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

def main():
    #samples = random.choices([0.0, 1.0], weights=(0.25, 0.75), k=100_000_000)
    # print(samples)

    # print(sum(samples))
    #print(sum(samples) / len(samples))
    
    
    
    names = [
              "MiniGrid-Adv-8x8-v0",
              "MiniGrid-AdvSimple-8x8-v0",
              "MiniGrid-AdvSlippery-8x8-v0",
              "MiniGrid-AdvLava-8x8-v0",
            # "MiniGrid-SingleDoor-7x6-v0",
            # "MiniGrid-DoubleDoor-10x8-v0",
            # "MiniGrid-DoubleDoor-12x12-v0",
            # "MiniGrid-DoubleDoor-16x16-v0",
            # "MiniGrid-LavaSlipperyS12-v0",
            # "MiniGrid-LavaSlipperyS12-v1",
            # "MiniGrid-LavaSlipperyS12-v2",
            # "MiniGrid-LavaSlipperyS12-v3",
            # "MiniGrid-LavaCrossingS9N3-v0",
            # "MiniGrid-SimpleCrossingS9N3-v0",
             
            # "MiniGrid-ObstructedMaze-1Dlh-v0",
            # "MiniGrid-BlockedUnlockPickup-v0",
            # "MiniGrid-KeyCorridorS6R3-v0", 
            # "MiniGrid-LockedRoom-v0",
            # "MiniGrid-KeyCorridorS3R1-v0",
            # "MiniGrid-LavaGapS7-v0",
            # "MiniGrid-DoorKey-8x8-v0", 
            # "MiniGrid-Dynamic-Obstacles-8x8-v0",
            # "MiniGrid-Empty-Random-6x6-v0",
            # "MiniGrid-Fetch-6x6-N2-v0", 
            # "MiniGrid-FourRooms-v0", 
            # "MiniGrid-LavaGapS7-v0",
            # "MiniGrid-RedBlueDoors-6x6-v0",
            ]
    
    for name in names:
        env = gym.make(name) 
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)
        
        env.reset()
        
        img = env.get_frame(highlight=False)
        plt.title(name)
        plt.imshow(img)
        f = open(F"{name}.txt", "w")
        f.write(env.printGrid(init=True))
        f.close()


        plt.show()

if __name__ == '__main__':
    main()