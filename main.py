import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from ultralytics import YOLO  # YOLOv8 o YOLO11

# --- CARGAR MODELOS ---
# Detección de rostros
face_net = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

# Clasificador de mascarillas
mask_model = load_model("mask_detector.h5")

# Modelo YOLOv8 entrenado para detectar vest, glasses, goggles y labcoat
epp_model = YOLO("C:/Users/Narut/OneDrive/Escritorio/Prueba EPP/runs/detect/train2/weights/best.pt")

# --- CONFIGURAR CÁMARA ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cv2.namedWindow("Detección EPP", cv2.WINDOW_NORMAL)

# Colores por clase (puedes personalizar)
epp_colors = {
    "Bata-Correcta": (255, 255, 0),      # Amarillo
 #   "glasses": (0, 255, 255),   # Cyan
  #  "goggles": (255, 0, 255),   # Magenta
    "Bata-Incorrecta": (0, 165, 255)    # Naranja oscuro
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    (h, w) = frame.shape[:2]

    # --- DETECCIÓN DE ROSTROS ---
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    locs = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            if face.size == 0:
                continue
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # --- PREDICCIÓN DE MASCARILLAS ---
    if len(faces) > 0:
        preds = mask_model.predict(np.array(faces), batch_size=32)
    else:
        preds = []

    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        mask_prob = pred[0]

        label = "Con Mascarilla" if mask_prob > 0.5 else "Sin Mascarilla"
        color = (0, 255, 0) if mask_prob > 0.5 else (0, 0, 255)
        label_text = f"{label}: {mask_prob*100:.2f}%"

        cv2.putText(frame, label_text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # --- DETECCIÓN DE OTROS EPP CON YOLO ---
    results = epp_model(orig, verbose=False)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = epp_model.names[class_id]

        if conf > 0.5:
            color = epp_colors.get(class_name, (200, 200, 200))  # gris si no definido
            label = f"{class_name}: {conf*100:.1f}%"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- MOSTRAR RESULTADO ---
    cv2.imshow("Detección EPP", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
