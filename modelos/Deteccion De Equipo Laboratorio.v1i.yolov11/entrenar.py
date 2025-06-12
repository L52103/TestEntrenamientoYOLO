from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(data='C:/Users/Narut/TestEntrenamientoYOLO/modelos/Deteccion De Equipo Laboratorio.v1i.yolov11/data.yaml', epochs=50, batch=16, imgsz=640 , device=0)
