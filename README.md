# DATA3888 Image 01

## how to run denoise_classical.py?

### command
Gaussian blur with (5, 5) kernel:
    python3 denoise_classical.py -G 5
Median blur with kernel size 5:
    python3 denoise_classical.py -M 5

Note: run "python3 denoise_classical.py", the argument is set to default (Gaussian blur with (5, 5) kernel)

### parameters
Kernel size for Gaussian
#gaussian_kernel = (3, 3)    # Light blur, preserves more details
#gaussian_kernel = (5, 5)    # Standard denoise for mild noise
gaussian_kernel = (7, 7)    # Stronger blur, removes more noise
#gaussian_kernel = (11, 11)  # Heavy blur, may distort tissue details

Kernel size for Median (Must be odd: 3, 5, 7...)
=== best for salt-and-pepper noise ===
#median_kernel = 3         # Minimal filtering
#median_kernel = 5         # Moderate noise reduction
median_kernel = 7         # Stronger, may smooth out fine features
#median_kernel = 11        # Very aggressive, possible artifacts

### output
This program will create the denoised image folder under the same directory of the input images folder, ./100/B_Cells (input image folder) => ./100/B_Cells_Gaussian_5 (output image folder using Gaussian blur with (5, 5) kernel)
