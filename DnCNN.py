import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import os

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, features, kernel_size=3, padding=1, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, kernel_size=3, padding=1, bias=True))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.dncnn(x) 

dncnn_model_paths = {
    "blind": "Restormer_models/gaussian_color_denoising_blind.pth",
    "sigma15": "Restormer_models/gaussian_color_denoising_sigma15.pth",
    "sigma25": "Restormer_models/gaussian_color_denoising_sigma25.pth",
    "sigma50": "Restormer_models/gaussian_color_denoising_sigma50.pth",
    "real": "Restormer_models/real_denoising.pth"}

class DnCNN_Denoiser():
    def __init__(self, model_name="real"):
        self.model = DnCNN()
        model_path = dncnn_model_paths[model_name]

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
            print(f"Loaded DnCNN weights for '{model_name}' from {model_path}")
        else:
            print(f"WARNING: DnCNN weights for '{model_name}' not found at {model_path}. Using random weights.")
        
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            T.Resize((248, 248)),
            T.ToTensor()
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def denoise_image(self, input_tensor):
        with torch.no_grad():
            predicted_noise = self.model(input_tensor.to(self.device))
            denoised_image = input_tensor - predicted_noise
            return torch.clamp(denoised_image, 0, 1).squeeze(0).cpu()

def save_image(tensor, path):
    image = T.ToPILImage()(tensor.cpu())
    image.save(path)

def main():
    denoiser_instances = {
        name: DnCNN_Denoiser(model_name=name) for name in dncnn_model_paths.keys()
    }

    input_image_path = 'noisy.png'
    if not os.path.exists(input_image_path):
        print(f"ERROR: Input image '{input_image_path}' not found. Please provide it.")
        return

    first_denoiser_key = list(denoiser_instances.keys())[0]
    loaded_image_tensor = denoiser_instances[first_denoiser_key].load_image(input_image_path)
    print(f"Loaded input image: {input_image_path}")

    for model_name, denoiser in denoiser_instances.items():
        print(f"Processing with DnCNN '{model_name}'...")
        denoised_output = denoiser.denoise_image(loaded_image_tensor.clone()) 
        save_image(denoised_output, f"dncnn_{model_name}_output.png")

    print("DnCNN denoising complete.")

if __name__ == "__main__":
    main()