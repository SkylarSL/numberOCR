#simple algorithm to specifically get the number and no white space

from PIL import Image
import numpy as np

def resize():
    img = Image.open("eight.png", "r")
    height, width = img.size

    nullLine = [];
    nullLineRGB = [];
    blackLine = [];
    blackLineRGB = [];
    for i in range(width):
        for j in range(height):
            r, g, b = img.getpixel((j, i))
            total = r+g+b;
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

    finalImage = np.array(finalImage)
    data = np.zeros((y, x, 3), dtype=np.uint8)
    sum = 0
    for i in range(x):
        for j in range(y):
            data[j, i] = finalImage[sum]
            sum += 1;
    imgg = Image.fromarray(data)
    imgg.show()

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
