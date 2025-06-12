import cv2
import os
import time
from ultralytics import YOLO

# Cargar modelos YOLO
person_model = YOLO("ruta/a/person_model.pt")
labcoat_model = YOLO("ruta/a/labcoat_detector.pt")
labcoat_quality_model = YOLO("ruta/a/labcoat_quality.pt")

# Crear carpeta para capturas si no existe
output_dir = "capturas_incorrectas"
os.makedirs(output_dir, exist_ok=True)

# Configurar cámara
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cv2.namedWindow("Detección EPP", cv2.WINDOW_NORMAL)

# Variables para control de tiempo
incorrect_detected_time = None
captured = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    orig = frame.copy()
    current_time = time.time()

    # Paso 1: Detección de personas
    results_person = person_model(orig, verbose=False)[0]
    persons = [box for box in results_person.boxes if person_model.names[int(box.cls[0])] == "person" and float(box.conf[0]) > 0.5]

    bata_incorrecta_detectada = False

    for person_box in persons:
        x1, y1, x2, y2 = map(int, person_box.xyxy[0])
        person_roi = orig[y1:y2, x1:x2]

        # Paso 2: Detección de delantal
        results_labcoat = labcoat_model(person_roi, verbose=False)[0]
        has_labcoat = any(labcoat_model.names[int(box.cls[0])] == "labcoat" and float(box.conf[0]) > 0.5 for box in results_labcoat.boxes)

        if has_labcoat:
            # Paso 3: Validación del uso correcto
            results_quality = labcoat_quality_model(person_roi, verbose=False)[0]
            for box in results_quality.boxes:
                class_name = labcoat_quality_model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                color = (0, 255, 0) if "Correcta" in class_name else (0, 165, 255)
                label = f"{class_name}: {conf*100:.1f}%"
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1+bx1, y1+by1), (x1+bx2, y1+by2), color, 2)
                cv2.putText(frame, label, (x1+bx1, y1+by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Detectar bata incorrecta
                if "Incorrecta" in class_name:
                    bata_incorrecta_detectada = True

    # Manejo de tiempo para captura
    if bata_incorrecta_detectada:
        if incorrect_detected_time is None:
            incorrect_detected_time = current_time
        elif current_time - incorrect_detected_time >= 3 and not captured:
            # Guardar captura
            filename = f"{output_dir}/bata_incorrecta_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Captura guardada: {filename}")
            captured = True
    else:
        incorrect_detected_time = None
        captured = False

    cv2.imshow("Detección EPP", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
