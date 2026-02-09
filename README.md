# Blind Detektor (v1)

Android aplikace pro nevidomé: detekce objektů v reálném čase (YOLOv8n ONNX), overlay a hlasové čtení v češtině.

## Požadavky
- Android Studio (JDK 17)
- Model: `app/src/main/assets/yolov8n.onnx`
- Labels: `app/src/main/assets/labels_cs.json`

## Přidání modelu
Do `app/src/main/assets/` vlož:
- `yolov8n.onnx` (není součástí ZIPu)
- případně uprav `labels_cs.json`

## Export modelu (pokud máte jen .pt)
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=12 simplify=True
```

## Build
Otevři projekt v Android Studio a spusť na telefonu (minSdk 26).

## Ovládání
- **Přečíst relevantní**: řekne 3–4 nejdůležitější objekty + pozici
- **Průběžné hlášení**: čte jen při změně scény a s cooldownem
