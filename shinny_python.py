import streamlit as st
import os
import shutil
import datetime
from PIL import Image
import torch
import joblib
import numpy as np
from torchvision import transforms
from shinnyapp_func import get_transform, get_heatmap, predict

os.environ["STREAMLIT_WATCH_INSTALL_DIR"] = "false"

# Directories
IMAGE_DIR = "image"
RESULT_DIR = "result"
DENOISER_DIR = "denoised_models"
MASK_DIR = "masking_models"
DENOISERS = ["None", "Restormer", "Gaussian_Blur", "Median_Blur", "DnCNN"]
MASKS = ["Centre", "Non-centre", "Random"]
MODELS = ["CNN", "SVM", "Random Forest", "KNN"]

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Step 1: Upload Image
st.title("Cancer Analysis App")

uploaded_file = st.file_uploader("Step 1: Upload an image", type=['jpg', 'jpeg', 'png'])
if uploaded_file:
    img_path = os.path.join(IMAGE_DIR, uploaded_file.name)
    with open(img_path, 'wb') as f:
        f.write(uploaded_file.read())
    image = Image.open(img_path)
    st.image(image, caption="Original Image", use_container_width=True)

# Step 2: Denoise Image
# denoiser_files = [f for f in os.listdir(DENOISER_DIR) if f.endswith('.pth') or f.endswith('.pkl')]
selected_denoiser = st.selectbox("Step 2: Select a denoising model", DENOISERS)

# def apply_denoiser(image, model_path):
#     # Dummy denoising logic (replace with your real denoising)
#     return image.filter(Image.Filter.SMOOTH)

if uploaded_file:
    denoised_image = get_transform(image, selected_denoiser)
    st.image([image, denoised_image], caption=["Original", "Denoised"], width=300)

# Step 3: Analyze Image
# model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth') or f.endswith('.pkl')]
selected_model = st.selectbox("Step 3: Select an analysis model", MODELS)

if uploaded_file and selected_model:
    if selected_model == "CNN":
        model_path = f"denoised_models/CNN_{selected_denoiser}.pth"
    else:
        model_path = f"denoised_models/{selected_model}_{selected_denoiser}.pkl"

    # def predict(image, model_path):
    #     image = image.resize((224, 224))
    #     x = transforms.ToTensor()(image).unsqueeze(0)

    #     if model_path.endswith('.pth'):
    #         model = torch.load(model_path, map_location=torch.device('cpu'))
    #         model.eval()
    #         with torch.no_grad():
    #             output = model(x)
    #             prob = torch.sigmoid(output).item()
    #     elif model_path.endswith('.pkl'):
    #         model = joblib.load(model_path)
    #         x_np = x.view(-1).numpy().reshape(1, -1)
    #         prob = model.predict_proba(x_np)[0][1]  # assuming binary
    #     else:
    #         prob = 0.0
    #     return prob

    predicted_label, confidence = predict(denoised_image, selected_denoiser, selected_model)
    st.success(f"Predicted type: {predicted_label}, Confidence: {confidence:.2f}%")

# Step 4: Save Result and Cleanup
if uploaded_file and st.button("Finish"):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_img_path = os.path.join(RESULT_DIR, f"{now}_final_image.jpg")
    result_txt_path = os.path.join(RESULT_DIR, f"{now}_result.txt")

    denoised_image.save(final_img_path)
    with open(result_txt_path, 'w') as f:
        f.write(f"Predicted type: {predicted_label}\n")
        f.write(f"Confidence: {confidence:.2f}\n")
        f.write(f"Processed at: {now} (Sydney Time)\n")

    # Cleanup
    shutil.rmtree(IMAGE_DIR)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    st.success("Result saved and temporary files cleaned up.")
    st.balloons()
