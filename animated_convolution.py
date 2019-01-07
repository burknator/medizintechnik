import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def visual_convolution(image, kernel, debug=False):
    fig = plt.figure()
    _image = np.pad(image, ((1, 1), (1, 1)), 'constant') # padd with zeros
    output_image = np.copy(image)
    frames = [output_image]
    convolve(image, kernel, frames=frames)
    img = plt.imshow(np.copy(output_image), animated=True, cmap="gray", norm=NoNorm())
    def update(f):
        img.set_array(f)
        return img
    
    ani = FuncAnimation(fig, update, frames=frames, interval=50)
    html = HTML(ani.to_html5_video())
    plt.close()
    return html

def convolve(image, kernel, debug=False, frames=None):
    _image = np.pad(image, ((1, 1), (1, 1)), 'constant') # padd with zeros
    output_image = np.copy(image)
    for (x, y), value in np.ndenumerate(image):
        if (debug and x == 0 and y < 3): # print first 3 iterations
            print(output_image)
        window = _image[x:x+3, y:y+3] # image[x-1:x+2, y-1:y+2] but x=x+1, y=y+1 because of pad
        output_image[x, y] = np.sum(np.multiply(window, kernel))
        if frames:
            frames.append(np.copy(output_image))
    return output_image