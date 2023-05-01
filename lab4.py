from itertools import cycle
import cv2
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

__all__ = [
    'morphological_chan_vese',
    'morphological_geodesic_active_contour',
    'inverse_gaussian_gradient',
    'circle_level_set',
    'checkerboard_level_set'
]

__version__ = (2, 1, 1)
__version_str__ = ".".join(map(str, __version__))


class _fcycle(object):

    def __init__(self, iterable):
        """Call functions from the iterable each time it is called."""
        self.funcs = cycle(iterable)

    def __call__(self, *args, **kwargs):
        f = next(self.funcs)
        return f(*args, **kwargs)


# SI and IS operators for 2D and 3D.
_P2 = [np.eye(3),
       np.array([[0, 1, 0]] * 3),
       np.flipud(np.eye(3)),
       np.rot90([[0, 1, 0]] * 3)]
_P3 = [np.zeros((3, 3, 3)) for i in range(9)]

_P3[0][:, :, 1] = 1
_P3[1][:, 1, :] = 1
_P3[2][1, :, :] = 1
_P3[3][:, [0, 1, 2], [0, 1, 2]] = 1
_P3[4][:, [0, 1, 2], [2, 1, 0]] = 1
_P3[5][[0, 1, 2], :, [0, 1, 2]] = 1
_P3[6][[0, 1, 2], :, [2, 1, 0]] = 1
_P3[7][[0, 1, 2], [0, 1, 2], :] = 1
_P3[8][[0, 1, 2], [2, 1, 0], :] = 1


def sup_inf(u):
    """SI operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    erosions = []
    for P_i in P:
        erosions.append(ndi.binary_erosion(u, P_i))

    return np.array(erosions, dtype=np.int8).max(0)


def inf_sup(u):
    """IS operator."""

    if np.ndim(u) == 2:
        P = _P2
    elif np.ndim(u) == 3:
        P = _P3
    else:
        raise ValueError("u has an invalid number of dimensions "
                         "(should be 2 or 3)")

    dilations = []
    for P_i in P:
        dilations.append(ndi.binary_dilation(u, P_i))

    return np.array(dilations, dtype=np.int8).min(0)


_curvop = _fcycle([lambda u: sup_inf(inf_sup(u)),   # SIoIS
                   lambda u: inf_sup(sup_inf(u))])  # ISoSI


def _check_input(image, init_level_set):
    """Check that shapes of `image` and `init_level_set` match."""
    if image.ndim not in [2, 3]:
        raise ValueError("`image` must be a 2 or 3-dimensional array.")

    if len(image.shape) != len(init_level_set.shape):
        raise ValueError("The dimensions of the initial level set do not "
                         "match the dimensions of the image.")


def _init_level_set(init_level_set, image_shape):
    if isinstance(init_level_set, str):
        if init_level_set == 'checkerboard':
            res = checkerboard_level_set(image_shape)
        elif init_level_set == 'circle':
            res = circle_level_set(image_shape)
        elif init_level_set == 'ellipsoid':
            res = ellipsoid_level_set(image_shape)
        else:
            raise ValueError("`init_level_set` not in "
                             "['checkerboard', 'circle', 'ellipsoid']")
    else:
        res = init_level_set
    return res


def circle_level_set(image_shape, center=None, radius=None):
    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if radius is None:
        radius = min(image_shape) * 3.0 / 8.0

    grid = np.mgrid[[slice(i) for i in image_shape]]
    grid = (grid.T - center).T
    phi = radius - np.sqrt(np.sum((grid)**2, 0))
    res = np.int8(phi > 0)
    return res


def ellipsoid_level_set(image_shape, center=None, semi_axis=None):
    if center is None:
        center = tuple(i // 2 for i in image_shape)

    if semi_axis is None:
        semi_axis = tuple(i / 2 for i in image_shape)

    if len(center) != len(image_shape):
        raise ValueError("`center` and `image_shape` must have the same length.")

    if len(semi_axis) != len(image_shape):
        raise ValueError("`semi_axis` and `image_shape` must have the same length.")

    if len(image_shape) == 2:
        xc, yc = center
        rx, ry = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y: ((x - xc) / rx) ** 2 +
                         ((y - yc) / ry) ** 2,
            image_shape, dtype=float)
    elif len(image_shape) == 3:
        xc, yc, zc = center
        rx, ry, rz = semi_axis
        phi = 1 - np.fromfunction(
            lambda x, y, z: ((x - xc) / rx) ** 2 +
                            ((y - yc) / ry) ** 2 +
                            ((z - zc) / rz) ** 2,
            image_shape, dtype=float)
    else:
        raise ValueError("`image_shape` must be a 2- or 3-tuple.")

    res = np.int8(phi > 0)
    return res


def checkerboard_level_set(image_shape, square_size=5):
    grid = np.ogrid[[slice(i) for i in image_shape]]
    grid = [(grid_i // square_size) & 1 for grid_i in grid]

    checkerboard = np.bitwise_xor.reduce(grid, axis=0)
    res = np.int8(checkerboard)
    return res


def inverse_gaussian_gradient(image, alpha=100.0, sigma=5.0):
    gradnorm = ndi.gaussian_gradient_magnitude(image, sigma, mode='nearest')
    return 1.0 / np.sqrt(1.0 + alpha * gradnorm)


def morphological_chan_vese(image, iterations, init_level_set='checkerboard',
                            smoothing=1, lambda1=1, lambda2=1,
                            iter_callback=lambda x: None):


    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    for _ in range(iterations):

        # inside = u > 0
        # outside = u <= 0
        c0 = (image * (1 - u)).sum() / float((1 - u).sum() + 1e-8)
        c1 = (image * u).sum() / float(u.sum() + 1e-8)

        # Image attachment
        du = np.gradient(u)
        abs_du = np.abs(du).sum(0)
        aux = abs_du * (lambda1 * (image - c1)**2 - lambda2 * (image - c0)**2)

        u[aux < 0] = 1
        u[aux > 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

    return u


def morphological_geodesic_active_contour(gimage, iterations,
                                          init_level_set='circle', smoothing=1,
                                          threshold='auto', balloon=0,
                                          iter_callback=lambda x: None):


    image = gimage
    init_level_set = _init_level_set(init_level_set, image.shape)

    _check_input(image, init_level_set)

    if threshold == 'auto':
        threshold = np.percentile(image, 40)

    structure = np.ones((3,) * len(image.shape), dtype=np.int8)
    dimage = np.gradient(image)
    # threshold_mask = image > threshold
    if balloon != 0:
        threshold_mask_balloon = image > threshold / np.abs(balloon)

    u = np.int8(init_level_set > 0)

    iter_callback(u)

    for _ in range(iterations):

        # Balloon
        if balloon > 0:
            aux = ndi.binary_dilation(u, structure)
        elif balloon < 0:
            aux = ndi.binary_erosion(u, structure)
        if balloon != 0:
            u[threshold_mask_balloon] = aux[threshold_mask_balloon]

        # Image attachment
        aux = np.zeros_like(image)
        du = np.gradient(u)
        for el1, el2 in zip(dimage, du):
            aux += el1 * el2
        u[aux > 0] = 1
        u[aux < 0] = 0

        # Smoothing
        for _ in range(smoothing):
            u = _curvop(u)

        iter_callback(u)

    return u


import os
import logging

import numpy as np
from imageio import imread
import matplotlib
from matplotlib import pyplot as plt



# in case you are running on machine without display, e.g. server
if os.environ.get('DISPLAY', '') == '':
    logging.warning('No display found. Using non-interactive Agg backend.')
    matplotlib.use('Agg')

PATH_IMG_NODULE = 'images/mama07ORI.bmp'
PATH_IMG_STARFISH = './seastar2.png'
PATH_IMG_LAKES = 'images/lakes3.jpg'
PATH_IMG_CAMERA = '/Users/datle/Desktop/active-contour/camera.png'
PATH_IMG_COINS = 'images/coins.png'
PATH_ARRAY_CONFOCAL = 'images/confocal.npy'


def visual_callback_2d(background, fig=None):

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def visual_callback_3d(fig=None, plot_each=1):
    from mpl_toolkits.mplot3d import Axes3D
    # PyMCubes package is required for `visual_callback_3d`
    try:
        import mcubes
    except ImportError:
        raise ImportError("PyMCubes is required for 3D `visual_callback_3d`")

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax = fig.add_subplot(111, projection='3d')
    plt.pause(0.001)

    counter = [-1]

    def callback(levelset):

        counter[0] += 1
        if (counter[0] % plot_each) != 0:
            return

        if ax.collections:
            del ax.collections[0]

        coords, triangles = mcubes.marching_cubes(levelset, 0.5)
        ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                        triangles=triangles)
        plt.pause(0.1)

    return callback


def rgb2gray(img):
    """Convert a RGB image to gray scale."""
    return 0.2989 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def example_nodule():
    logging.info('Running: example_nodule (MorphGAC)...')

    # Load the image.
    img = imread(PATH_IMG_NODULE)[..., 0] / 255.0

    # g(I)
    gimg = inverse_gaussian_gradient(img, alpha=1000, sigma=5.48)

    # Initialization of the level-set.
    init_ls = circle_level_set(img.shape, (100, 126), 20)

    # Callback for visual plotting
    callback = visual_callback_2d(img)

    # MorphGAC.
    morphological_geodesic_active_contour(gimg, iterations=45,
                                             init_level_set=init_ls,
                                             smoothing=1, threshold=0.31,
                                             balloon=1, iter_callback=callback)


def example_starfish():
    logging.info('Running: example_starfish (MorphGAC)...')

    # Load the image.
    imgcolor = imread(PATH_IMG_STARFISH) / 255.0
    img = rgb2gray(imgcolor)

    # g(I)
    gimg = inverse_gaussian_gradient(img, alpha=1000, sigma=2)

    # Initialization of the level-set.
    init_ls = circle_level_set(img.shape, (163, 137), 135)

    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # MorphGAC.
    s= morphological_geodesic_active_contour(gimg, iterations=100,
                                             init_level_set=init_ls,
                                             smoothing=2, threshold=0.3,
                                             balloon=-1, iter_callback=callback)
    print(s)

def example_coins():
    logging.info('Running: example_coins (MorphGAC)...')

    # Load the image.
    img = imread(PATH_IMG_COINS) / 255.0

    # g(I)
    gimg = inverse_gaussian_gradient(img)

    # Manual initialization of the level set
    init_ls = np.zeros(img.shape, dtype=np.int8)
    init_ls[10:-10, 10:-10] = 1

    # Callback for visual plotting
    callback = visual_callback_2d(img)

    # MorphGAC.
    morphological_geodesic_active_contour(gimg, 230, init_ls,
                                             smoothing=1, threshold=0.69,
                                             balloon=-1, iter_callback=callback)


def example_lakes():
    logging.info('Running: example_lakes (MorphACWE)...')

    # Load the image.
    imgcolor = imread(PATH_IMG_LAKES)/255.0
    img = rgb2gray(imgcolor)

    # MorphACWE does not need g(I)

    # Initialization of the level-set.
    init_ls = circle_level_set(img.shape, (80, 170), 25)

    # Callback for visual plotting
    callback = visual_callback_2d(imgcolor)

    # Morphological Chan-Vese (or ACWE)
    morphological_chan_vese(img, iterations=200,
                               init_level_set=init_ls,
                               smoothing=3, lambda1=1, lambda2=1,
                               iter_callback=callback)


def example_camera():
    """
    Example with `morphological_chan_vese` with using the default
    initialization of the level-set.
    """

    logging.info('Running: example_camera (MorphACWE)...')

    # Load the image.
    img = imread(PATH_IMG_CAMERA)/255.0

    # Callback for visual plotting
    callback = visual_callback_2d(img)

    # Morphological Chan-Vese (or ACWE)
    morphological_chan_vese(img, 35,
                               smoothing=3, lambda1=1, lambda2=1,
                               iter_callback=callback)


def example_confocal3d():
    logging.info('Running: example_confocal3d (MorphACWE)...')

    # Load the image.
    img = np.load(PATH_ARRAY_CONFOCAL)

    # Initialization of the level-set.
    init_ls = circle_level_set(img.shape, (30, 50, 80), 25)

    # Callback for visual plotting
    callback = visual_callback_3d(plot_each=20)

    # Morphological Chan-Vese (or ACWE)
    morphological_chan_vese(img, iterations=150,
                               init_level_set=init_ls,
                               smoothing=1, lambda1=1, lambda2=2,
                               iter_callback=callback)

def water_shed(path):
    img= cv2.imread(path)
    # convert to gray
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying dilation for sure_bg detection
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Defining kernel for opening operation
    kernel= np.ones((3,3), np.uint8)
    # use open to remove any small white noises in the image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('Open', opening)
    # After opening, will perform dilation, Dilation increases object boundary to background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Sure background image
    cv2.imshow('dilated', sure_bg)
    # foreground extraction
    # there are two options for fg extract: one is distance transform, second is erosion
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Finding the Unknown Area (Neither sure Foreground Nor for Background)
    unknown = np.subtract(sure_bg, sure_fg)
    cv2.imshow('unknown', unknown)
    # apply watershed algorithm
    ret, markers = cv2.connectedComponents(sure_fg)
    print(markers)
    # Add one so that sure background is not 1
    markers = markers + 1
    # Making the unknown area as 0
    markers[unknown == 255] = 0
    # cv2.imshow('markers2', markers)
    cv2.waitKey(0)
    markers = cv2.watershed(img, markers)
    # boundary region is marked with -1
    img[markers == -1] = (255, 0, 0)
    cv2.imshow('watershed', img)
    cv2.waitKey(0)


def k_means_segment(path, K=3):
    img = cv2.imread(path)
    # conver to hsv color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # vectorized img
    vectorized = img_hsv.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    figure_size = 5
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()

def mean_shift(path):
    img= cv2.imread(path, cv2.IMREAD_COLOR)
    # filter to reduce noise
    img= cv2.medianBlur(img, 3)
    # flatten the image
    flate_image= img.reshape((-1,3))
    flate_image= np.float32(flate_image)
    #mean shift
    bandwidth= estimate_bandwidth(flate_image, quantile=.06, n_samples=3000)
    ms= MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flate_image)
    labeled= ms.labels_

    segments= np.unique(labeled)
    print(segments.shape[0])

    # get the average color of each segment
    total= np.zeros((segments.shape[0],3), dtype=float)
    count= np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label]= total[label]+ flate_image[i]
        count[label]+=1
    avg = total/count
    avg = np.uint8(avg)

    res= avg[labeled]
    result= res.reshape((img.shape))

    plt.imshow(result)
    plt.show()

import matplotlib.pyplot as plt

import sklearn
def menu():
    f= int(input('choose function:'))
    if f==1:
        print('active contour')
        img= cv2.imread("E:/CPV/messi5.jpg")
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
        callback = visual_callback_2d(img)

        # Morphological Chan-Vese (or ACWE)
        morphological_chan_vese(img, 35,
                                   smoothing=3, lambda1=1, lambda2=1,
                                   iter_callback=callback)
        plt.show()
    if f==2:
        print('watershed')
        water_shed("E:/CPV/messi5.jpg")
    if f==3:
        print('k-means-segmentation')
        k_means_segment("E:/CPV/messi5.jpg", K=4)
    if f==4:
        print("mean-shift segmentation")
        mean_shift("E:/CPV/messi5.jpg")

menu()