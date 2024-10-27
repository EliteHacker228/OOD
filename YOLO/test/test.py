from ultralytics import YOLO

model = YOLO('../result/ood_model_yolo.pt')
results = model.predict(source='./testdata', save=True)
