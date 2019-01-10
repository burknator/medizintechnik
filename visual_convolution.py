import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm

from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, Math, Latex

def show_matrix(data, title=None, mark=set()):
    # Helper-method to show a matrix
    return display(Latex(r'\[\scriptstyle ' + (title.replace('_', '\_') + ' = ' if title else '') + r' \def\g{\color{lightgray}}\left(\begin{array}{%s}' % (r'r'*data.shape[0]) + r'\\'.join([r' & '.join([ value if (x, y) in mark or not mark else r'\g{%s}' % value for (y, value) in enumerate(map(str, col))]) for (x, col) in enumerate(data)]) + r'\end{array}\right)\]'))

def show_matrix_with_kernel_and_result(data, kernel, kernel_x, kernel_y, result):
    # Helper-method to show a matrix
    mark = set([(i, j) for i in range(kernel_x-1, kernel_x+2) for j in range(kernel_y-1, kernel_y + 2) ])
    data = data.astype(str)

    for (x, y) in mark:
        data[x, y] = '%s\cdot %.2f' % (data[x, y], kernel[x - kernel_x + 1, y - kernel_y + 1])
    return display(Latex(r'\[\scriptstyle \def\g{\color{lightgray}}\left(\begin{array}{%s}' % (r'r'*data.shape[0]) + r'\\'.join([r' & '.join([ value if (x, y) in mark else r'\g{%s}' % value for (y, value) in enumerate(map(str, col))]) for (x, col) in enumerate(data)]) + r'\end{array}\right)=\left(\begin{array}{%s}' % (r'r'*data.shape[0]) + r'\\'.join([r' & '.join([ value if y == kernel_y-1 and x == kernel_x-1 else r'\g{%s}' % value for (y, value) in enumerate(map(str, col))]) for (x, col) in enumerate(result)]) + r'\end{array}\right)\]'))


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
    display(html)

def stepped_convolution(image, kernel, steps=3):
    convolve(image, kernel, stop_after_steps=steps, visualize=True)

def convolve(image, kernel, debug=False, frames=None, stop_after_steps=None, visualize=False):
    _image = np.pad(image, ((1, 1), (1, 1)), 'constant') # padd with zeros
    output_image = np.copy(image)
    count=0
    for (x, y), _ in np.ndenumerate(image):
        window = _image[x:x+3, y:y+3] # image[x-1:x+2, y-1:y+2] but x=x+1, y=y+1 because of pad
        output_image[x, y] = np.sum(np.multiply(window, kernel))
        if visualize:
            plt.imshow(output_image, cmap="gray", norm=NoNorm())
            #show_matrix(_image, set([(i, j) for i in range(x, x+3) for j in range(y, y + 3) ]))
            show_matrix_with_kernel_and_result(_image, kernel, x+1, y+1, output_image)
        if frames:
            frames.append(np.copy(output_image))
        count += 1
        if stop_after_steps and count >= stop_after_steps:
            break
    return output_image