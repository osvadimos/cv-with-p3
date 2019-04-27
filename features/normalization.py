from skimage import io
from skimage import data, segmentation, color
from skimage.io import imread
from skimage import data
from skimage.future import graph
img = io.imread("../pants.jpg")
img_segments = segmentation.slic(img, compactness=30, n_segments=200)
out1 = color.label2rgb(img_segments, img, kind='avg')
segment_graph = graph.rag_mean_color(img, img_segments, mode='similarity')
img_cuts = graph.cut_normalized(img_segments, segment_graph)
normalized_cut_segments = color.label2rgb(img_cuts, img, kind='avg')
io.imshow(normalized_cut_segments)
io.show()