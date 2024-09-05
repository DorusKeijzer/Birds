import os

index2bird = {}
startdir = os.getcwd()
os.chdir("data/test")

for i, bird in enumerate([dir for dir in os.listdir() if os.path.isdir(dir)]):
    index2bird[i] = bird
os.chdir(startdir)
if __name__ == "__main__":
    print(index2bird[200])
            

