# pip install opencv-python scikit-image
import sys
import os
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

psnr = 0.0
ssim = 0.0
n = 0

def to_grey(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if len(sys.argv) != 3:
    print("[Usage]: calc_img_diff img1 img2")
else:
    sr_image = sys.argv[1] #"d1.jpg"
    hr_image = sys.argv[2] #"d2.jpg"
    compute_psnr = cv2.PSNR(cv2.imread(sr_image), cv2.imread(hr_image))
    compute_ssim = compare_ssim(to_grey(cv2.imread(sr_image)),
                                to_grey(cv2.imread(hr_image)))
    psnr += compute_psnr
    ssim += compute_ssim
    print("average psnr = ", psnr)
    print("average ssim = ", ssim)
