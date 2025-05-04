import torch
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
import sys
import os

sys.path.append(os.path.abspath('Restormer'))
from Restormer.basicsr.models.archs.restormer_arch import Restormer

model_names = {"blind": "Restormer_models/gaussian_color_denoising_blind.pth",
               "sigma15": "Restormer_models/gaussian_color_denoising_sigma15.pth",
               "sigma25": "Restormer_models/gaussian_color_denoising_sigma25.pth",
               "sigma50": "Restormer_models/gaussian_color_denoising_sigma50.pth",
               "real": "Restormer_models/real_denoising.pth"}

class Restormer_Denoise():
    def __init__(self, model_name="real"):
        self.model = Restormer()
        self.model.load_state_dict(torch.load(f'Restormer_models/{model_names[model_name]}')['params'], strict=False)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            transforms.Resize((248, 248)),
            T.ToTensor()
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(self.device)

    def denoise_image(self, input_tensor):
        with torch.no_grad():
            restored = self.model(input_tensor)
            restored = torch.clamp(restored, 0, 1)
            return restored.squeeze(0).cpu()



def main():
    blind = Restormer()
    sigma15 = Restormer()
    sigma25 = Restormer()
    sigma50 = Restormer()
    real = Restormer()
    blind.load_state_dict(torch.load('Restormer_models/gaussian_color_denoising_blind.pth')['params'], strict=False)
    blind.eval()
    sigma15.load_state_dict(torch.load('Restormer_models/gaussian_color_denoising_sigma15.pth')['params'], strict=False)
    sigma15.eval()
    sigma25.load_state_dict(torch.load('Restormer_models/gaussian_color_denoising_sigma25.pth')['params'], strict=False)
    sigma25.eval()
    sigma50.load_state_dict(torch.load('Restormer_models/gaussian_color_denoising_sigma50.pth')['params'], strict=False)
    sigma50.eval()
    real.load_state_dict(torch.load('Restormer_models/real_denoising.pth')['params'], strict=False)
    real.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    blind.to(device)
    sigma15.to(device)
    sigma25.to(device)
    sigma50.to(device)
    real.to(device)

    # Image preprocessing (dynamic resizing handled here)
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([
            transforms.Resize((248, 248)),
            T.ToTensor()
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image.to(device)

    def denoise_image(input_tensor, model):
        with torch.no_grad():
            restored = model(input_tensor)
            restored = torch.clamp(restored, 0, 1)
            return restored.squeeze(0).cpu()

    # Save output
    def save_image(tensor, path):
        image = T.ToPILImage()(tensor)
        image.save(path)

    # Run
    input_image = load_image('noisy.png')
    save_image(denoise_image(input_image, blind), 'blind.png')
    save_image(denoise_image(input_image, sigma15), 'sigma15.png')
    save_image(denoise_image(input_image, sigma25), 'sigma25.png')
    save_image(denoise_image(input_image, sigma50), 'sigma50.png')
    save_image(denoise_image(input_image, real), 'real.png')


if __name__ == "__main__":
    main()