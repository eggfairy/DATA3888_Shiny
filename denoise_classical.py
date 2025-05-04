import os
import sys
import cv2
from glob import glob

# === Command line arguments ===
if len(sys.argv) == 3 and sys.argv[1] in ["-G","-M", "-g", "-m"]:
    # python3 file.py -G/-M number
    if sys.argv[1] in ["-G", "-g"]:
        method = "G"
        level_num = int(sys.argv[2])
        level = (level_num, level_num)
        print(level)
    elif sys.argv[1] in ["-M", "-m"]:
        method = "M"
        level_num = int(sys.argv[2])
        level = level_num
    print(f"Method: '{method}', Level: '{level}'")
else:
    # default setting
    method = "G"
    level = (5, 5)
    print(f"Method: '{method}', Level: '{level}'")


# === PATH of input & output folders ===
# input folder labels
path_pre = './Images/100/'
image_labels = ['B_Cells', 'CD4+_T_Cells', 'DCIS_1', 'DCIS_2', 'Invasive_Tumor', 'Prolif_Invasive_Tumor']
input_paths = []
for label in image_labels:
    input_paths.append(path_pre+label)

# check folder exists
close_program = 0
for p in input_paths:
    if not os.path.isdir(p):
        close_program = 1
        print(f"Input folder '{p}'doesn't exists.")

if close_program == 1:
    print("Program is closing...")
    os._exit()

# output folder labels
paths_out_gaussian = []
paths_out_median = []

for p in input_paths:
    paths_out_gaussian.append(p+'_Gaussian_'+str(level_num))
    os.makedirs(p+'_Gaussian_'+str(level_num), exist_ok=True)
    paths_out_median.append(p+'_Median_'+str(level_num))
    os.makedirs(p+'_Median_'+str(level_num), exist_ok=True)

#print(paths_out_gaussian)
#print(paths_out_median)


# === Process images ===
for p in input_paths:
    image_paths = glob(os.path.join(p, '*.png'))
    num_images = len(image_paths)
    count = 0
    if method == "G":
        for path in image_paths:
            img = cv2.imread(path)

            # Apply Gaussian Blur
            img_gauss = cv2.GaussianBlur(img, level, 0)
            out_gauss = os.path.join(p+'_Gaussian_'+str(level_num), os.path.basename(path).replace('.png', '_gauss.png'))
            cv2.imwrite(out_gauss, img_gauss)
            count += 1
            print(f"Saved {count}/{num_images}: {out_gauss}")

    elif method == "M":
        for path in image_paths:
            img = cv2.imread(path)

            # Apply Median Blur
            img_median = cv2.medianBlur(img, level)
            out_median = os.path.join(p+'_Median_'+str(level_num), os.path.basename(path).replace('.png', '_median.png'))
            cv2.imwrite(out_median, img_median)
            count += 1
            print(f"Saved {count}/{num_images}: {out_median}")




'''
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
'''