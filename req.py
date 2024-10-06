import time
import requests
import base64
import random
import glob
import io
from PIL import Image
import matplotlib.pyplot as plt

def request_path_from_backend(img_path, start=None, goal=None):
    try:
        with open(img_path, "rb") as img_file:
            img_data = img_file.read()

        img_base64 = base64.b64encode(img_data).decode("utf-8")

        request_data = {
            "start": start,
            "goal": goal,
            "image": img_base64,
        }

        t_ = time.time()

        response = requests.post("http://localhost:8000/api", json=request_data)

        if response.status_code == 200:
            result_data = response.json()
            image_base64 = result_data["image"]
            start = result_data["start"]
            goal = result_data["goal"]

            img_data = base64.b64decode(image_base64)
            img_pil = Image.open(io.BytesIO(img_data))

            return img_pil
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None

paths = glob.glob('trail\\WhatsApp\\*.jpeg')
ind = random.randint(0, len(paths) - 1)
img_path = paths[ind]

start = (500, 500)
goal = (50, 50)

img_with_path = request_path_from_backend(img_path, start, goal)

if img_with_path:
    plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    img_pil = Image.open(img_path)
    plt.imshow(img_pil)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(img_with_path)
    plt.title("Image with Path")
    plt.axis('off')

    plt.show()
else:
    print("No valid path received from the backend.")
