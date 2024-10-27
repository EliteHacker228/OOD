from ultralytics import YOLO

# Замените 'yolov11.pt' на путь к предобученной модели YOLOv11, если необходимо
model = YOLO('../result/ood_model_yolo.pt')

results = model.predict(source='./testdata', save=True)