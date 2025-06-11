from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')
    model.train(data='C:/Users/Narut/OneDrive/Escritorio/Prueba EPP/entrenamiento/data.yaml', epochs=100, imgsz=512 , batch=8, device=0)
