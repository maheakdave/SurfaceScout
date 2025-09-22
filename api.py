import uvicorn
import torch
import random
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form
from io import BytesIO
from PIL import Image
import torchvision.transforms.functional as F
from ultralytics import YOLO
from neural_astar.planner import NeuralAstar
from neural_astar.utils.training import load_from_ptl_checkpoint

app = FastAPI()

device = "cpu"
yolo_model = YOLO("best.pt")
neural_astar = NeuralAstar(encoder_arch='CNN').to(device)
neural_astar.load_state_dict(load_from_ptl_checkpoint("model\\mazes_032_moore_c8\\lightning_logs\\version_0\\checkpoints\\"))
neural_astar.eval()

def process_image_and_find_path(img, start=None, goal=None):
    img_resized = F.resize(torch.tensor(img).permute(2, 0, 1), (640, 640)).permute(1, 2, 0).numpy()
    img_resized = np.ascontiguousarray(img_resized)

    results = yolo_model.predict(img_resized, save=False, imgsz=640, conf=0.1, iou=0.5)

    binary_map = [[1 for _ in range(img_resized.shape[1])] for _ in range(img_resized.shape[0])]
    for result in results:
        boxes = result.boxes.xyxy
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            for i in range(y1, y2):
                for j in range(x1, x2):
                    binary_map[i][j] = 0

    dilated_map = F.resize(torch.tensor(binary_map, dtype=torch.uint8).unsqueeze(0).unsqueeze(0), size=(32, 32)).squeeze(0)

    def is_valid_point(point, dilated_map):
        return dilated_map[0, point[0], point[1]] == 1

    if start is None or not is_valid_point(start, dilated_map):
        while True:
            start = (random.randint(0, 31), random.randint(0, 31))
            if is_valid_point(start, dilated_map):
                break

    if goal is None or not is_valid_point(goal, dilated_map) or goal == start:
        while True:
            goal = (random.randint(0, 31), random.randint(0, 31))
            if is_valid_point(goal, dilated_map):
                break

    start_map = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    start_map[0, 0, start[0], start[1]] = 1

    goal_map = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    goal_map[0, 0, goal[0], goal[1]] = 1

    na_outputs = neural_astar(dilated_map.unsqueeze(0).to(device), start_map.to(device), goal_map.to(device), store_intermediate_results=True)
    na_outputs[-1][-1]['paths'][0] = (1 - na_outputs[-1][-1]['paths'][0])
    path = F.resize(na_outputs[-1][-1]['paths'][0], (640, 640))
    path_np = path.permute(1, 2, 0).cpu().detach().numpy()
    path_np = np.clip(path_np * 255, 0, 255).astype(np.uint8)
    alpha = 0.6 
    blended_img = (img_resized*alpha) + (path_np*(1-alpha))
    return blended_img, start, goal

@app.post("/pathfinding/")
async def pathfinding(file: UploadFile = File(...), x_start: int = Form(None), y_start: int = Form(None), x_goal: int = Form(None), y_goal: int = Form(None)):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    img = np.array(img)

    start = (x_start, y_start) if x_start is not None and y_start is not None else None
    goal = (x_goal, y_goal) if x_goal is not None and y_goal is not None else None

    blended_img, valid_start, valid_goal = process_image_and_find_path(img, start, goal)

    result_image = blended_img.tolist()

    return {"start": valid_start, "goal": valid_goal, "path": result_image}

if __name__ == '__main__':
    uvicorn.run(app, port=8000)
