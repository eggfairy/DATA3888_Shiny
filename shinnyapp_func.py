from PIL import Image
from resotrmer import Restormer_Denoise
from models_DnCNN import DnCNN_Denoiser
from denoise_classical import GaussianBlur, MedianBlur
from masking import centre_mask, non_centre_mask, random_mask
import torch
from torchvision import models, transforms
from torchcam.methods import GradCAM
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
import requests

import warnings
warnings.filterwarnings("ignore")

restomer = Restormer_Denoise("blind")
dncnn = DnCNN_Denoiser()

TRANSFORMS = {
    "None": lambda x:x,
    "Restormer": restomer.denoise_image,
    "Gaussian_Blur": GaussianBlur,
    "Median_Blur": MedianBlur,
    "DnCNN": dncnn.denoise_image,
    "Centre": centre_mask,
    "Non-centre": non_centre_mask,
    "Random": random_mask 
}

DENOISERS = ["None", "Restormer", "Gaussian_Blur", "Median_Blur", "DnCNN"]
MASKS = ["Centre", "Non-centre", "Random"]
LABELS = ["Immune_Cells", "Non_Invasive_Tumor", "Invasive_Tumor_Set"]
OUTPUT_LABELS = {
    "Immune_Cells": "Immune Cell",
    "Non_Invasive_Tumor": "Non Invasive Tumor",
    "Invasive_Tumor_Set": "Invasive Tumor"
}
ML_MODELS_URLs = {
    "SVM": {
        "None": "1OSr4Npixf9Db22_YvYMjNOh2z1-bJzo7",
        "Restormer": "1Wz5Z6S5Tq1Jv3s41sU-eOH8-PiREKlJw",
        "Gaussian_Blur": "10RiSI1YuLrsAeN9KmUcq5pNfTwbTjk9J",
        "Median_Blur": "1Q0JmI9c5aCpjFxY6sU6TASRdLT2kgp2L",
        "DnCNN": "13TDJKK2yrgPsx_maZfZF5K670cCvI2TW",
        "Centre": "1maTbljndxz4d2nfdSEEDjmLUrREOG_Vw",
        "Non-centre": "1hXF4EsvhRa-1pG1f5BbavotAefdgdBIU",
        "Random": "1Sb9sRFwJe03TGZXyR9IDosn4rJ_3F-LK"

    },
    "RF": {
        "None": "1967KVeT8ERg3EqdSsAcqox4Cao3ZZpOk",
        "Restormer": "17tqG7P15lnFB6FqnP7jB1S9VENwFwOJx",
        "Gaussian_Blur": "1lV5zB-MTrzZZCBD6iSrleBvba2vBit_6",
        "Median_Blur": "1FG9S-WVVyqnIm462MAghOKY7RxBB9bU-",
        "DnCNN": "1xXEEG7p2Z1jq9OHtTWawyh_Tb4YmGPX8",
        "Centre": "1Q8qRWoy0n-O2jHsD3EwoZbJpJQYmaK-M",
        "Non-centre": "1tzEE7EFeVNbO6mb8T11ukgc2bF1b1BAA",
        "Random": "1F17I251SjVTawfvcKBiqscd15U5a0NhQ"
    },
    "KNN": {
        "None": "1gvvPLddZc9uguTm3mCQrvwra0FI4CSVh",
        "Restormer": "1ac9WhNUf91igkzRcE11GVXfHOq4JFSmV",
        "Gaussian_Blur": "1wb5pZmPXD1HHxRMGYmd-JrGmTK0IMU67",
        "Median_Blur": "16zE8guQEDfdbzzBdWGPj7xTbga4ZJya1",
        "DnCNN": "1TGLaCcP2rmiiUZnoNHyUApsiuHnHEjBk",
        "Centre": "1WD0DRDn6TUurQUyme7D3qrI9rRg8j99D",
        "Non-centre": "1Da1QlDeR0G5EssFk6HUXOq2-dXAEGk5N",
        "Random": "1IwHBREdNJROdkwv9t3quEe3OvHyKqcpj"
    }
}

IMG_SIZE = 224
torch_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
le = LabelEncoder()
numeric_labels = le.fit_transform(LABELS)

def load_cnn_model(model_path: str) -> torch.nn.Module:
    cnn_model = models.resnet18(pretrained=False)
    cnn_model.fc = torch.nn.Linear(cnn_model.fc.in_features, len(LABELS))
    cnn_model.load_state_dict(torch.load(model_path))
    cnn_model = cnn_model.to(device)
    return cnn_model

def feature_extraction(model: torch.nn.Module, img: Image.Image):
    """
    Extract features from the image using the specified model.

    Args:
        model (torch.nn.Module): The model to be used for feature extraction.
        img (Image.Image): The image to be transformed.

    Returns:
        The extracted features.
    """
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]) # remove the last layer
    feature_extractor.eval()
    feature_extractor.to(device)
    with torch.no_grad():
        input_tensor = torch_transform(img).unsqueeze(0).to(device)
        features = feature_extractor(input_tensor).squeeze().cpu().numpy().flatten()
    return [features]

def download_model(model: str, transform: str):
    file_id = ML_MODELS_URLs[model][transform]
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    if transform in DENOISERS:
        model_path = f"denoised_models/{model}_{transform}.pkl"
        if os.path.exists(model_path):
            return

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Failed to download model({model_path}): {response.status_code}")
        
    else:
        model_path = f"masking_models/{model}_{transform}.pkl"
        if os.path.exists(model_path):
            return 
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            raise Exception(f"Failed to download model({model_path}): {response.status_code}")
        
def load_ml_model(model: str, transform: str):
    model = "RF" if model == "Random Forest" else model
    if transform in DENOISERS:
        model_path = f"denoised_models/{model}_{transform}.pkl"
        if not os.path.exists(model_path):
            download_model(model, transform)

        with open(model_path, "rb") as f:
            pred_model = pickle.load(f)
        return pred_model
        
    else:
        model_path = f"masking_models/{model}_{transform}.pkl"
        if not os.path.exists(model_path):
            download_model(model, transform)
        
        with open(model_path, "rb") as f:
            pred_model = pickle.load(f)
        return pred_model


def get_transform(img: Image.Image, transform: str) -> Image.Image:
    """
    Transform the image using the specified transformation method(denoise or masking).

    Args:
        img (Image.Image): The image to be transformed.
        transform (str): The transformation method to be applied.
            Options include "None", "Restormer", "Gaussian_Blur", "Median_Blur", "DnCNN",
            "Centre", "Non-centre", and "Random".

    Returns:
        Image.Image: The transformed image.
    """
    return TRANSFORMS.get(transform, lambda x: x)(img)

def get_heatmap(img: Image.Image, transform: str) -> tuple[Image.Image, Image.Image]:
    """
    Generate a heatmap for the specified transformation method.
    Note: For CNN ONLY!!!

    Args:
        img (Image.Image): The image to be transformed.
        transform (str): The transformation method to be applied.
            Options include "None", "Restormer", "Gaussian_Blur", "Median_Blur", "DnCNN",
            "Centre", "Non-centre", and "Random".

    Returns:
        Image.Image: The heatmap image.
        Image.Image: The image circled by the important region.
    """
    transformed_img = get_transform(img, transform).convert("RGB")
    input_tensor = torch_transform(transformed_img).unsqueeze(0).to(device)
    cnn_model_path = f"denoised_models/CNN_{transform}.pth" if transform in DENOISERS else f"masking_models/{transform}.pth"
    cnn_model = load_cnn_model(cnn_model_path)
    cnn_model.eval()
    cam_extractor = GradCAM(cnn_model, target_layer="layer4") # the last conv layer
    output = cnn_model(input_tensor)
    predicted_class =output.argmax(dim=1).item()
    activation_map = cam_extractor(predicted_class, output)

    heatmap = activation_map[0].squeeze().cpu().numpy()
    heatmap_resized = cv2.resize(heatmap, transformed_img.size)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Threshold + contour
    _, thresh = cv2.threshold(heatmap_uint8, 220, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box
    img_cv = np.array(transformed_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)

    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img_cv, [approx], -1, (0, 0, 255), 1)

    return Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def predict(img: Image.Image, transform: str, model: str) -> tuple[str, int]:
    """
    Predict the class of the image and generate a heatmap.

    Args:
        img (Image.Image): The image to be transformed.
        transform (str): The transformation method to be applied.
            Options include "None", "Restormer", "Gaussian_Blur", "Median_Blur", "DnCNN",
            "Centre", "Non-centre", and "Random".
        model (str): The model to be used for prediction.
            Options include "CNN", "SVM", "Random Forest", and "KNN.

    Returns:
        str: The predicted class label.
        int: The confidition of the prediction(0-100%).
    """
    transformed_img = get_transform(img, transform)
    cnn_model_path = f"denoised_models/CNN_{transform}.pth" if transform in DENOISERS else f"masking_models/CNN_{transform}.pth"
    cnn_model = load_cnn_model(cnn_model_path)

    if model == "CNN":
        input_tensor = torch_transform(transformed_img).unsqueeze(0).to(device)
        cnn_model.eval()
        output = cnn_model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        predicted_label = le.inverse_transform([predicted_class])[0]
        predicted_label = OUTPUT_LABELS.get(predicted_label, "None")
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() * 100
        return predicted_label, confidence
    
    else: #traditional ML models
        features = feature_extraction(cnn_model, transformed_img)
        pred_model = load_ml_model(model, transform)
        # if transform in DENOISERS:
        #     if model == "Random Forest":
        #         with open(f"denoised_models/RF_{transform}.pkl", "rb") as f:
        #             pred_model = pickle.load(f)
        #     else:
        #         with open(f"denoised_models/{model}_{transform}.pkl", "rb") as f:
        #             pred_model = pickle.load(f)
        # else:
        #     if model == "Random Forest":
        #         with open(f"masking_models/RF_{transform}.pkl", "rb") as f:
        #             pred_model = pickle.load(f)
        #     else:
        #         with open(f"masking_models/{model}_{transform}.pkl", "rb") as f:
        #             pred_model = pickle.load(f)
        

        
        predicted_class = pred_model.predict(features)[0]
        predicted_label = le.inverse_transform([predicted_class])[0]
        predicted_label = OUTPUT_LABELS.get(predicted_label, "None")
        confidence = pred_model.predict_proba(features)[0][predicted_class] * 100
        return predicted_label, confidence # confidence not available for traditional ML models for now


def main():
    img = Image.open("cell_355_100.png")
    restomer_img = get_transform(img, "Restormer")
    dncnn_img = get_transform(img, "DnCNN")
    restomer_img.save("cell_355_100_restormer.png")
    dncnn_img.save("cell_355_100_dncnn.png")
    


if __name__ == "__main__":
    main()