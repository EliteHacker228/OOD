from ultralytics import YOLO

# Замените 'yolov11.pt' на путь к предобученной модели YOLOv11, если необходимо
model = YOLO('../models/yolo11n.pt')

# Запустите обучение
model.train(data='../dataset/data.yaml', epochs=50, imgsz=640)

# Оценка модели на валидационном наборе данных
results = model.val()

# Сохранение модели
model.save('../result/ood_model_yolo.pt')

# results = model.predict(source='path/to/test/images', save=True)