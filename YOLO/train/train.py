from ultralytics import YOLO

model = YOLO('../models/yolo11n.pt')
model.train(data='../dataset/data.yaml', epochs=50, imgsz=640)
model.save('../result/ood_model_yolo.pt')
