
#simple algorithm to specifically get the number and no white space

from PIL import Image
import PIL
import numpy as np
import math
from mlxtend.data import loadlocal_mnist

def main():
    getMNIST()
    resize()

def resize():

    #initialize the images
    img = Image.open("eight.png", "r").convert("L")
    converted_image = img.resize((28, 28), PIL.Image.ANTIALIAS)
    height, width = converted_image.size
    x = 0
    final_image = []
    height, width = converted_image.size

    #changes image matrix into an array with inverse greyscale
    for i in range(height):
        for j in range(width):
            h = math.ceil(1 - ((converted_image.getpixel((j, i)))/255));
            final_image.append([h]);
            x += 1

    #return final image and converts to numpy
    final_image = [(np.matrix(final_image)).reshape((784, 1))]
    return (final_image)

#this was to test the output, includes visible difference to see the number
'''
    for i in range(height):
        for j in range(width):
            if(final_image[x] == [1]):
                print("\033[1;37;40m" + str(final_image[x]), end=" ")
            else:

                print("\033[1;31;40m" + str(final_image[x]), end=" ")
            x+=1
        print("\n")
'''


def getMNIST():
    image, label = loadlocal_mnist(
        "./train-images.idx3-ubyte",
        "./train-labels.idx1-ubyte")
    training_data = []

    for x, y in zip(image, label):
        value = np.zeros((10, 1));
        value[y] = 1
        pixel_value = np.matrix(x)/255
        training_data.append([pixel_value.transpose(), value])
    #print(training_data[0])
    return training_data

'''
    for i in range(height):
        for j in range(width):
            print(converted_image.getpixel((j, i)))
        print("\n")

    for i in range(label.shape[0]):
        for j in range(label[i]):
            z = np.zeros((10), dtype = int)
            z[label[i]] = 1
            np.append(new_label, z, axis = 0);
'''

if __name__ == "__main__":
    main();
