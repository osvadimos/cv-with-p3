from skimage import io
from skimage import feature
from skimage import color
img = io.imread("../pants.jpg")
img = color.rgb2gray(img)
edge = feature.canny(img,3)
io.imshow(edge)
io.show()