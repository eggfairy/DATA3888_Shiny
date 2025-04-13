import cv2
import os
from glob import glob

# === Input and output folder ===
input_folder = './B_Cells_test'             # the folder containing input images
output_folder = './B_Cells_test_denoise'    # the folder containing output images
os.makedirs(output_folder, exist_ok=True)


# === Parameters ===
# Kernel size for Gaussian
#gaussian_kernel = (3, 3)    # Light blur, preserves more details
#gaussian_kernel = (5, 5)    # Standard denoise for mild noise
gaussian_kernel = (7, 7)    # Stronger blur, removes more noise
#gaussian_kernel = (11, 11)  # Heavy blur, may distort tissue details

# Kernel size for Median (Must be odd: 3, 5, 7...)
# best for salt-and-pepper noise
#median_kernel = 3         # Minimal filtering
#median_kernel = 5         # Moderate noise reduction
median_kernel = 7         # Stronger, may smooth out fine features
#median_kernel = 11        # Very aggressive, possible artifacts


# === Process images ===
image_paths = glob(os.path.join(input_folder, '*.png'))

for path in image_paths:
    img = cv2.imread(path)

    # Apply Gaussian Blur
    img_gauss = cv2.GaussianBlur(img, gaussian_kernel, 0)
    out_gauss = os.path.join(output_folder, os.path.basename(path).replace('.png', '_gauss.png'))
    cv2.imwrite(out_gauss, img_gauss)

    # Apply Median Blur
    img_median = cv2.medianBlur(img, median_kernel)
    out_median = os.path.join(output_folder, os.path.basename(path).replace('.png', '_median.png'))
    cv2.imwrite(out_median, img_median)

    print(f"Saved: {out_gauss}, {out_median}")