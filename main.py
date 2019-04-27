from skimage import io, color

img = io.imread("pants.jpg")


gray = color.rgb2gray(img)
io.imshow(gray)
io.show()

exit()
