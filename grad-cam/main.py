from PIL import Image
from chainer.links import VGG16Layers
from utils import GuidedVGG16
from backprop import GradCAM, GradCAM_PP, GuidedBackprop
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":

    filename = "images/dog_cat.png"
    images = [Image.open(filename)]
    # target_label = 282
    target_label = -1

    # Grad-CAM
    vgg = VGG16Layers()
    grad_cam = GradCAM(vgg, "conv5_3", "prob")
    L_gcam = grad_cam.feed(images, target_label)

    # Grad-CAM++
    grad_cam_pp = GradCAM_PP(vgg, "conv5_3", "prob")
    L_gcampp = grad_cam_pp.feed(images, target_label)

    # Guided backprob
    gvgg = GuidedVGG16()
    guided_backprop = GuidedBackprop(gvgg, "input", "prob")
    R_0 = guided_backprop.feed(images, target_label)

    # Guided Grad-CAM
    ggrad_cam = R_0 * L_gcam[:, :, np.newaxis]
    ggrad_cam -= ggrad_cam.min()
    ggrad_cam = np.uint8(255 * ggrad_cam / ggrad_cam.max())

    R_0 -= R_0.min()
    R_0 = np.uint8(255 * R_0 / R_0.max())

    cmap = plt.get_cmap('jet')
    L_gcam = np.uint8(255. * cmap(L_gcam))
    L_gcampp = np.uint8(255. * cmap(L_gcampp))

    Image.fromarray(L_gcam).save("images/gcam.png")
    Image.fromarray(L_gcampp).save("images/gcampp.png")
    Image.fromarray(R_0).save("images/gbp.png")
    Image.fromarray(ggrad_cam).save("images/ggrad_cam.png")

    base = images[0].convert("RGBA")
    L_gcam[:, :, 3] = 100
    mix = Image.alpha_composite(base, Image.fromarray(L_gcam))
    mix.save("images/original_grad.png")
    L_gcampp[:, :, 3] = 100
    mix = Image.alpha_composite(base, Image.fromarray(L_gcampp))
    mix.save("images/original_gradpp.png")
