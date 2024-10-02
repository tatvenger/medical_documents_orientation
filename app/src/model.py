import PIL
import torch
import torchvision

import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

from PIL import Image

import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3

DEVICE = "cpu"

CLASSES = {
    "rotation_0":0,
    "rotation_90":1,
    "rotation_180":2,
    "rotation_270":3
}
LABELS_MAP = dict(zip(CLASSES.values(), CLASSES.keys()))
N_CLASSES = 4

current_file = Path(__file__)
current_file_dir = current_file.parent

# Фунция для загрузки предобученной модели и весов
def pretrained_model(params_path:str, device:str):

    model = inception_v3(weights=None)
    model.AuxLogits.fc = nn.Linear(in_features=768, out_features=N_CLASSES)
    model.fc = nn.Linear(in_features=2048, out_features=N_CLASSES)
    model.classifier = nn.Linear(in_features=2048, out_features=N_CLASSES)
    model = model.to(device)
    model.load_state_dict(torch.load(params_path, map_location=torch.device(device)))
    return model

# Функция для предсказания поворота изображения
def predict(path:str, inp_size: int):

    try:
        image = Image.open(path).convert('RGB')
    except PIL.UnidentifiedImageError:
        status = 'Fail'
        return status, path

    transformer = transforms.Compose([
                transforms.Resize((inp_size,inp_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    image = transformer(image)
    model = pretrained_model(os.path.join(current_file_dir,'Inception_img_orientation.pth'), DEVICE)
    with torch.no_grad():
        x = image.to(DEVICE).unsqueeze(0)
        predictions = model.eval()(x)
    result = int(torch.argmax(predictions, 1).cpu())
    status = 'OK'
    return status, LABELS_MAP[result]

# Функция для поворота изображения
def rotate_image(image: Image.Image, rotation_class: str):
    if rotation_class == "rotation_90":
        return image.rotate(90, expand=True)
    elif rotation_class == "rotation_180":
        return image.rotate(180, expand=True)
    elif rotation_class == "rotation_270":
        return image.rotate(270, expand=True)
    else:
        return image
