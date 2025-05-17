
Step 1: Upload Image
- Upload a .jpg, .jpeg, or .png file.

Step 2: Denoise Image
- Select a denoising model from the dropdown.
- View original and denoised images side by side.

Please store denoiser model in /denoiser

Step 3: Analyze Image
- Select a model such as CNN, SVM, or KNN.
- Run analysis to predict cancer likelihood (randomized for simulation).

Please store model in /model

Step 4: View Result
- Displays the result and processed image.
- Clicking 'Finish' will:
  - Save the result image and a text file with current Sydney time in `result/`
  - Delete temporary files in `image/`
  - Automatically close the app

Output:

After clicking **Finish**, the following will be saved in `result/` according to current time:

- 2025-05-18_15-08-32_final_image.jpg
- 2025-05-18_15-08-32_result.txt
