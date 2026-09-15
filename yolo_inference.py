from ultralytics import YOLO
from pathlib import Path

model = YOLO('/Users/thomas/Documents/projects/football analysis/runs/detect/train-4/weights/best.pt')

project_dir = Path(__file__).resolve().parent
output_dir = project_dir / 'runs' / 'detect' / 'football_result'
results = model.predict(
    'input_videos/08fd33_4.mp4',
    save=True,
    project=str(output_dir.parent),
    name=output_dir.name,
    exist_ok=True,
)
print(results)
print('------------------')
for box in results[0].boxes:
    print(box)
print(f'Video saved in: {output_dir / "08fd33_4.mp4"}')