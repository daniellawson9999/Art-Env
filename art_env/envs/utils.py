from PIL import Image
import os
import pathlib
import numpy as np
import time
import matplotlib.pyplot as plt

def load_images_abstract(n_images, size=100):
    main_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'images', 'Abstract_gallery','Abstract_image_')
    images = []
    arrays = []
    for i in range(n_images):
        image_path = main_path + str(i) + ".jpg"
        image = Image.open(image_path)
        width, height = image.size
        left = width / 2 - size / 2
        top = height / 2 - size / 2
        right = width / 2 + size / 2
        bottom = height / 2 + size / 2
        cropped_image = image.crop((left,top,right,bottom))
        images.append(cropped_image)
        arrays.append(np.array(cropped_image))
    return images, np.array(arrays)

if __name__ == "__main__":
    fig, ax = plt.subplots()
    plt.ion()
    plt.show()
    images,arrays = load_images_abstract(n_images=50, size=200)
    import pdb; pdb.set_trace()
    for i in range(len(images)):
        #plt.imshow(arrays[i])
        ax.imshow(arrays[i])
        plt.draw()
        plt.pause(0.001)
        #accept = raw_input('OK? ')
