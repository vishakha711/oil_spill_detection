from pathlib import Path
from ultralytics import YOLO

def run_stage1_inference(weights_path,image_path,conf_threshold=0.25):
    model=YOLO(weights_path)
    results=model.predict(source=image_path,conf=conf_threshold,save=True,project="inference_results",name="stage1_preds",exist_ok=True)
    detections=[]
    for r in results:
        for box,conf in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy()):
            detections.append({"bbox_pixels":[float(x) for x in box],"confidence":float(conf)})
    print(f"Detected {len(detections)} potential slick(s) in {image_path}")
    return detections

if __name__=="__main__":
    ROOT=Path(__file__).resolve().parent.parent
    weights=ROOT/"ml_layer/weights/best.pt"
    test_image=ROOT/"data/yolo_dataset/images/val/ow-0009.jpg"
    if weights.exists() and test_image.exists(): print("Detections:",run_stage1_inference(weights,test_image))
    else: print("Please verify best.pt and test image paths.")