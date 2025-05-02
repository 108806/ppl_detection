import os
from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Folder roboczy
work_dir = os.path.dirname(os.path.abspath(__file__))

# Lista plików do przetworzenia
fragments = [f for f in os.listdir(work_dir) if f.startswith('fragment_') and f.endswith('.mp4') or f.endswith('.avi')  and '_detected' not in f]

# Wymuś użycie GPU
if not torch.cuda.is_available():
    raise RuntimeError('CUDA (GPU) nie jest dostępna! Upewnij się, że sterowniki i CUDA są poprawnie zainstalowane.')
device = 'cuda'

# Załaduj największy model YOLOv8 (yolov8x.pt)
model = YOLO('yolov8x.pt')  # automatycznie pobierze model jeśli nie ma
model.to(device)

# Lista kolorów (BGR)
COLORS = [
    (255, 0, 0),    # niebieski
    (0, 255, 0),    # zielony
    (0, 0, 255),    # czerwony
    (255, 255, 0),  # cyjan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # żółty
    (128, 0, 128),  # fiolet
    (0, 128, 255),  # pomarańcz
    (128, 128, 0),  # oliwkowy
    (0, 128, 128),  # morski
]

for frag in fragments:
    input_path = os.path.join(work_dir, frag)
    output_path = os.path.join(work_dir, frag.replace('.mp4', '_colored.mp4'))
    print(f'Przetwarzam {frag}...')
    # Detekcja tylko ludzi (klasa 0), niższy próg confidence i większa rozdzielczość
    results = model(
        input_path,
        stream=True,
        classes=[0],
        device=device,
        conf=0.15,
        imgsz=1280
    )
    # Otwórz wejściowy film
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    prev_gray = None
    for i, (r, _) in enumerate(zip(results, iter(int, 1))):
        ret, frame = cap.read()
        if not ret:
            break
        # Wzmocnienie kontrastu (CLAHE)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        frame_contrast = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        # Wyostrzenie
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        frame_sharp = cv2.filter2D(frame_contrast, -1, kernel)
        # Motion mask (opcjonalnie)
        gray = cv2.cvtColor(frame_sharp, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            motion = cv2.absdiff(gray, prev_gray)
            _, motion_mask = cv2.threshold(motion, 15, 255, cv2.THRESH_BINARY)
            frame_sharp[motion_mask > 0] = cv2.addWeighted(frame_sharp, 0.7, np.full_like(frame_sharp, 255), 0.3, 0)[motion_mask > 0]
        prev_gray = gray
        # Detekcja na przetworzonej klatce
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None and r.boxes.conf is not None else []
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            # Wyznacz jasny kolor na podstawie tła (odwrócony kolor w lewym górnym rogu ramki)
            patch = frame[max(y1,0):min(y1+5,height), max(x1,0):min(x1+5,width)]
            if patch.size > 0:
                mean_color = patch.mean(axis=(0,1))
                inv_color = tuple([int(255-c) for c in mean_color])
                color = tuple([max(180, c) for c in inv_color])  # wymuś jasność
            else:
                color = (255,255,255)
            # Rysuj subtelną, półprzezroczystą ramkę
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            # Dodaj napis z numerem osoby i confidence
            conf = float(confs[idx]) if idx < len(confs) else 0.0
            label = f"person {idx+1} : {conf:.2f}"
            font_scale = 0.5
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_x = x1
            text_y = max(y1 - 5, th + 2)
            cv2.rectangle(frame, (text_x, text_y-th-2), (text_x+tw, text_y+2), color, -1)  # tło pod tekst
            cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness, cv2.LINE_AA)
        out.write(frame)
    cap.release()
    out.release()
    print(f'Zapisano: {output_path}')

print('Gotowe!')
