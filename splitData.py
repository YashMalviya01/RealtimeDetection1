import os
import random
import shutil
from itertools import islice

outputFolderPath = "Datasets/SplitData"
inputFolderPath = "Datasets/all"
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}
classes = ["fake", "real"]

# --------  Remove and recreate output folder -----------
if os.path.exists(outputFolderPath):
    try:
        shutil.rmtree(outputFolderPath)
    except OSError as e:
        print(f"Error: {e}")
os.makedirs(outputFolderPath, exist_ok=True)

# --------  Directories to Create -----------
os.makedirs(f"{outputFolderPath}/train/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images", exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels", exist_ok=True)

# --------  Get the Names of Images and Labels -----------
listNames = [name.split('.')[0] for name in os.listdir(inputFolderPath) if name.endswith(('.jpg', '.png'))]
uniqueNames = list(set(listNames))  # Remove duplicates if any
print(f"Total unique images found: {len(uniqueNames)}")

# --------  Shuffle the list to ensure randomness -----------
random.shuffle(uniqueNames)

# --------  Calculate the number of images for each split -----------
lenData = len(uniqueNames)
lenTrain = int(lenData * splitRatio['train'])
lenVal = int(lenData * splitRatio['val'])
lenTest = int(lenData * splitRatio['test'])

# Adjust if total doesn't add up to lenData due to rounding
if lenData != lenTrain + lenVal + lenTest:
    remaining = lenData - (lenTrain + lenVal + lenTest)
    lenTrain += remaining
print(f"Total Images: {lenData}\nSplit sizes: Train={lenTrain}, Val={lenVal}, Test={lenTest}")

# --------  Split the list into train, val, and test sets -----------
lengthToSplit = [lenTrain, lenVal, lenTest]
Input = iter(uniqueNames)
Output = [list(islice(Input, elem)) for elem in lengthToSplit]

print(f"Final Split: Train={len(Output[0])}, Val={len(Output[1])}, Test={len(Output[2])}")

# --------  Copy the files to respective directories -----------
sequence = ['train', 'val', 'test']
for i, split in enumerate(Output):
    for fileName in split:
        # Copy image file
        imageFilePath = f'{inputFolderPath}/{fileName}.jpg'
        if os.path.exists(imageFilePath):
            shutil.copy(imageFilePath, f'{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg')
        else:
            print(f"Warning: Image file {fileName}.jpg not found. Skipping...")

        # Check if the label (.txt) file exists before copying
        labelFilePath = f'{inputFolderPath}/{fileName}.txt'
        if os.path.exists(labelFilePath):
            shutil.copy(labelFilePath, f'{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt')
        else:
            print(f"Warning: Label file {fileName}.txt not found. Skipping...")

print("Split Process Completed...")

# -------- Creating Data.yaml file  -----------
dataYaml = f'''path: ../Data
train: ../train/images
val: ../val/images
test: ../test/images

nc: {len(classes)}
names: {classes}
'''

yamlFilePath = f"{outputFolderPath}/data.yaml"
with open(yamlFilePath, 'w') as f:
    f.write(dataYaml)

print("Data.yaml file Created at:", yamlFilePath)
