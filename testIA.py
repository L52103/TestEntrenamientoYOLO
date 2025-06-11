import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


def main():
    # --- PASO 1: Cargar los modelos ---
    prototxt_path = "deploy.prototxt"
    weights_path = "res10_300x300_ssd_iter_140000.caffemodel"
    face_net = cv2.dnn.readNet(prototxt_path, weights_path)

    mask_model_path = "mask_detector.h5"
    print("[INFO] Cargando modelo de mascarillas en formato H5...")
    mask_net = load_model(mask_model_path)

    # --- PASO 2: Iniciar la captura de video ---
    print("[INFO] Iniciando la cámara...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    cv2.namedWindow("Deteccion de Mascarillas", cv2.WINDOW_NORMAL)
    print("Presiona 'q' para salir.")

    # --- PASO 3: Bucle principal ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                     (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        faces = []
        locs = []
        preds = []

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

                # Usa tamaño 150x150 si tu modelo fue entrenado así
                face = cv2.resize(face, (224, 224))
                face = preprocess_input(face)
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            preds = mask_net.predict(faces, batch_size=32)

        # print para debug (puedes comentar luego)
        # print(preds)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box

            mask_prob = pred[0]  # Probabilidad de usar mascarilla

            if mask_prob > 0.5:
                label = "Con Mascarilla"
                color = (0, 255, 0)
            else:
                label = "Sin Mascarilla"
                color = (0, 0, 255)

            label = f"{label}: {mask_prob * 100:.2f}%"
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Deteccion de Mascarillas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Limpiando...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
