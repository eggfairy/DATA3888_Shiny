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

def predict(img: Image.Image, transform: str) -> tuple[Image.Image, Image.Image, str, int]:
    """
    Predict the class of the image and generate a heatmap.
    Args:
        img (Image.Image): The image to be transformed.
        transform (str): The transformation method to be applied.
            Options include "None", "Restormer", "Gaussian_Blur", "Median_Blur", "DnCNN",
            "Centre", "Non-centre", and "Random".
    Returns:
        str: The predicted class label.
        int: The confidition of the prediction(0-100%).
    """
    transformed_img = get_transform(img, transform)
    input_tensor = torch_transform(transformed_img).unsqueeze(0).to(device)
    cnn_model_path = f"denoised_models/CNN_{transform}.pth" if transform in DENOISERS else f"masking_models/{transform}.pth"
    cnn_model = load_cnn_model(cnn_model_path)
    cnn_model.eval()
    output = cnn_model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    predicted_label = le.inverse_transform([predicted_class])[0]
    predicted_label = OUTPUT_LABELS.get(predicted_label, "None")
    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item() * 100
    return predicted_label, confidence

def main():
    img = Image.open("example.png")
    predicted_label, confidence = predict(img, "Restormer")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    ##### Do whatever you want for testing #####


if __name__ == "__main__":
    main()
