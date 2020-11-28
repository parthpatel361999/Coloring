import os

import numpy as np
from PIL import Image


def getImagePixels(directory, fileName):
    filePath = os.path.join(directory, fileName)
    image = Image.open(filePath)
    return np.asarray(image)
