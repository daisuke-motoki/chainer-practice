from PIL import Image
from chainer.links import VGG16Layers
from backprop import GradCAM

if __name__ == "__main__":

    filename = "images/dog_cat.png"
    images = [Image.open(filename)]
    target_label = -1

    vgg = VGG16Layers()
    grad_cam = GradCAM(vgg, "conv5_3", "prob")
    L_gcam = grad_cam.feed(images, target_label)
