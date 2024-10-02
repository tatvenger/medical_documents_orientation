from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import uvicorn
import argparse

import os
from PIL import Image
from pathlib import Path

from model import predict, rotate_image

app = FastAPI()

current_file = Path(__file__)
current_file_dir = current_file.parent
static_root_absolute = current_file_dir / "tmp"
templates_root = current_file_dir/"templates"

# Подключаем темплейты
templates = Jinja2Templates(directory=templates_root)

# Создаем папку с медиа для темплейтов
media_root = current_file_dir / "media"
media_root.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=media_root), name="media")

# Создаем временную папку для изображений, если ее нет
static_root_absolute.mkdir(exist_ok=True)
# Подключение статической папки для временного хранения изображений
app.mount("/tmp", StaticFiles(directory=static_root_absolute),name="tmp")


INP_SIZE = 299
DEVICE="cpu"

@app.get("/health")
def health():
    return {"status": "OK"}

@app.get("/", response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse("starting_page.html", {"request": request})

@app.post("/predict-detect")
async def process_request(file: UploadFile, request: Request):
    """Save image to tmp folder"""
    save_pth = f"{static_root_absolute}/{file.filename}"
    with open(save_pth, "wb") as fid:
        fid.write(await file.read())

    status, result = predict(save_pth, INP_SIZE)

    if status == 'OK':
        # Открываем оригинальное изображение
        image = Image.open(save_pth)
        # Поворачиваем изображение в зависимости от предсказанного класса
        rotated_image = rotate_image(image, result)
        rotated_image_path = f"{static_root_absolute}/rotated_{file.filename}"
        rotated_image.save(rotated_image_path)
        # Переходим на страницу с результатами
        return templates.TemplateResponse("result.html", {
            "request": request,
            "rotation_class": result,
            "original_image": file.filename,
            "rotated_image": f"rotated_{file.filename}"
        })
    else:
        # Выдаем страницу с ошибкой
         return templates.TemplateResponse("error_page.html",{
             "request": request,
             "rotation_class": status,
             "original_image": file.filename,
         })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, host=args["host"], port=args["port"])
