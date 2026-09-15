from ultralytics import YOLO
from pathlib import Path

project_dir = Path(__file__).resolve().parent
model = YOLO(project_dir / 'runs' / 'detect' / 'train-4' / 'weights' / 'best.pt')
output_dir = project_dir / 'runs' / 'detect' / 'football_result'
results = model.predict(
    project_dir / 'input_videos' / '08fd33_4.mp4',
    save=True,
    project=str(output_dir.parent),
    name=output_dir.name,
    exist_ok=True,
)
print(results)
print('------------------')
for box in results[0].boxes:
    print(box)
print('Video saved in: runs/detect/football_result/08fd33_4.mp4')