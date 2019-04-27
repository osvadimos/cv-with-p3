
from scipy import ndimage as ndi
from skimage.morphology import watershed, disk
from skimage import data
from skimage.io import imread
from skimage.filters import rank
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
img = data.astronaut()
img_gray = rgb2gray(img)
image = img_as_ubyte(img_gray)
#Calculate the local gradients of the image
#and only select the points that have a
#gradient value of less than 20
markers = rank.gradient(image, disk(5)) < 20
markers = ndi.label(markers)[0]
gradient = rank.gradient(image, disk(2))
#Watershed Algorithm
labels = watershed(gradient, markers)