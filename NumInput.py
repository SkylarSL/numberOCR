
#simple algorithm to specifically get the number and no white space

from PIL import Image
import numpy as np
from mlxtend.data import loadlocal_mnist

def main():
    #input();
    print(getMNIST()[0])

'''
def input():
    img = Image.open("eight.png", "r")
    height, width = img.size
    nullLine = [];
    nullLineRGB = [];
    blackLine = [];
    blackLineRGB = [];
    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((j, i))
            total = r + g + b;
            nullLine.append(total)
            nullLineRGB.append([r, g, b])
        line = search(nullLine)
        if line == True:
            blackLine.append(nullLine);
            blackLineRGB.append(nullLineRGB);
        nullLine = []
        nullLineRGB = []
    height, width = len(blackLine[0]), len(blackLine)
    blackLine2 = [];
    finalImage = [];
    for i in range(height):
        for j in range(width):
            x = blackLine[j][i]
            y = blackLineRGB[j][i]
            nullLine.append(x)
            nullLineRGB.append(y)
        line = search(nullLine)
        if line == True:
            blackLine2.append(nullLine);
            finalImage = finalImage + nullLineRGB;
        nullLine = []
        nullLineRGB = []
    test = blackLine2
    test = toGreyScale(test)
    print(test)
    x, y = len(blackLine2), len(blackLine2[0]);
    finalImage = np.array(finalImage)
    data = np.zeros((y, x, 3), dtype=np.uint8)
    sum = 0
    for i in range(x):
        for j in range(y):
            data[j, i] = finalImage[sum]
            sum += 1;
    imgg = Image.fromarray(data).resize((30, 30), Image.ANTIALIAS);
    #imgg.show()
    print(finalImage[0]);
def search(arr):
    length = len(arr)
    j = 0;
    for i in range(length):
        if arr[i] < 700:
            return True;
        elif j == length:
            return False;
        else:
            j = j+1;
            continue;
def toGreyScale(matrix):
    height = len(matrix[0]);
    width = len(matrix);
    for i in range(height):
        for j in range(width):
            matrix[j][i] = round((765 - matrix[j][i])/765, 2)
    return matrix
'''

def getMNIST():
    image, label = loadlocal_mnist(
        "./train-images.idx3-ubyte",
        "./train-labels.idx1-ubyte")
    new_label = [];
    training_data = [];
    
    
    for i, j in zip(image, label):
        result = np.zeros((10,1))
        result[j]=1
        testd = np.matrix(i)/255
        training_data.append([testd.transpose(),result])
#    for i in range(label.shape[0]):
#        for j in range(label[i]):
#            z = np.zeros((10), dtype = int)
#            z[label[i]] = 1
#            training_data.append([[image[j]],[z]]);

    #train = [image, new_label];
    return (training_data)

if __name__ == "__main__":
    main();
