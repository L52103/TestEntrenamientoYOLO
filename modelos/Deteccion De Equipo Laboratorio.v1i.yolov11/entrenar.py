from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11s.pt')
    model.train(data='C:/Users/Narut/TestEntrenamientoYOLO/modelos/Deteccion De Equipo Laboratorio.v1i.yolov11/data.yaml', epochs=800, batch=24, imgsz=640 , device=0)
