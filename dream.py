import torch

from torch.nn import Sequential
from torchvision.transforms import Normalize, ToTensor, ToPILImage, Resize
from torchvision.models import vgg16, VGG16_Weights

import argparse
import os

from PIL import Image
from tqdm import tqdm

from utils import *

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def gradient_ascent_step(
    image: torch.Tensor,
    model: torch.nn.Module,
    hook: Hook,
    learning_rate: float 
    ) -> torch.Tensor:
    
    image = image.to(DEVICE)
    image.requires_grad_()
    
    model(image)
    activations = hook.output

    loss = (activations**2).sum()

    loss.backward()
    normalized_grad = (image.grad - image.grad.mean()) / image.grad.std()

    image.grad.zero_()
    image = image.detach()
    image += learning_rate * normalized_grad

    return image

def get_initial_image_size(original_image_size: tuple[int], pyramid_levels: int, growth_rate: float):
    height, width = original_image_size
    return int(height / growth_rate ** (pyramid_levels-1)), int(width / growth_rate ** (pyramid_levels-1))

def get_new_image_size(original_image_size: tuple[int], pyramid_level: int, growth_rate: float):
    height, width = original_image_size
    pyramid_level += 1
    return int(height * growth_rate**pyramid_level), int(width * growth_rate**pyramid_level)


def deepdream(
        image: torch.Tensor,
        model: torch.nn.Module,
        model_layer: torch.nn.Module,
        pyramid_levels=4,
        growth_rate=1.8,
        steps=10,
        learning_rate=9e-2
    ) -> torch.Tensor:
    
    normalizer = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    denormalizer = DeNormalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    original_image_size = image.size()[-2:]
    initial_image_size = current_image_size = get_initial_image_size(original_image_size, pyramid_levels, growth_rate)
    hook = Hook(model_layer, backward=False)
    
    image = normalizer(image)
    for pyramid_level in tqdm(range(pyramid_levels)):
        resizer = Resize(current_image_size, antialias=True)
        image = resizer(image)

        for _ in tqdm(range(steps), leave=False):
            image = gradient_ascent_step(image, model, hook, learning_rate)
        
        image = image.detach()

        current_image_size = get_new_image_size(initial_image_size, pyramid_level, growth_rate)
    
    image = denormalizer(image)
    image = image.clamp(0, 1)
    return image

def generate_random_image(image_size: tuple[int] = (1, 3, 448, 448)):
    return torch.rand(image_size)

def load_image(img_path: str) -> torch.Tensor:
    return ToTensor()(Image.open(img_path).convert("RGB")).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="DeepDream in PyTorch using VGG16")

    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--layer-index", type=int, default=22, help="Layer index")
    parser.add_argument("--pyramid-levels", type=int, default=4, help="Number of pyramid levels")
    parser.add_argument("--growth-rate", type=float, default=1.8, help="Growth rate")
    parser.add_argument("--steps", type=int, default=10, help="Number of gradient ascent steps")
    parser.add_argument("--learning-rate", type=float, default=9e-2, help="Learning rate")
    parser.add_argument("--save-image", action='store_true', help="Save image")
    
    args = parser.parse_args()

    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(DEVICE)
    model.eval()
    
    if args.image is None:
        image = generate_random_image()
    else:
        image = load_image(args.image)

    image.to(DEVICE)

    image = deepdream(
        image, 
        model, 
        model.features[args.layer_index], 
        args.pyramid_levels, 
        args.growth_rate, 
        args.steps, 
        args.learning_rate
        )

    image_pil = ToPILImage()(image.squeeze(0))
    image_pil.show()
    
    if args.save_image:
        if args.image is None:
            image_path = "random_dream.png"
        else:
            image_path = os.path.splitext(args.image)[0] + "_dream.png"
        image_pil.save(image_path)

if __name__ == "__main__":
    main()