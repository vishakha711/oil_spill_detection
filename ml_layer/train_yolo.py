from pathlib import Path
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent.parent

def train_stage_one():
    model=YOLO(str(ROOT/"runs"/"stage1_yolo"/"weights"/"last.pt"))
    model.train(resume=True)
    best=ROOT/"runs"/"stage1_yolo"/"weights"/"best.pt"
    print(f"Training complete. Best model: {best}")

if __name__=="__main__":
    train_stage_one()