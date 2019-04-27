from skimage import io
from skimage import filters
from skimage import color
img = io.imread("../pants.jpg")
img = color.rgb2gray(img)
edge = filters.sobel(img)
io.imshow(edge)
io.show()