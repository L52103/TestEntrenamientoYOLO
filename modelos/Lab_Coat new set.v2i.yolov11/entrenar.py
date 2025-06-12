from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11n.pt')
    model.train(data='C:/Users/Narut/TestEntrenamientoYOLO/modelos/Lab_Coat new set.v2i.yolov11/data.yaml', epochs=100, batch=12, imgsz=640 , device=0)
