from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import cv2
from dataset import OilSpillDataset

def convert_to_yolo_format(box,w,h):
    xmin,ymin,xmax,ymax=box
    return [((xmin+xmax)/2)/w,((ymin+ymax)/2)/h,(xmax-xmin)/w,(ymax-ymin)/h]

def build_yolo_dataset(raw_dataset_dir,metadata_path,output_dir):
    out=Path(output_dir)
    for split in ["train","val"]:
        (out/"images"/split).mkdir(parents=True,exist_ok=True)
        (out/"labels"/split).mkdir(parents=True,exist_ok=True)

    df=pd.read_excel(metadata_path,engine="openpyxl")
    img_col=next((c for c in df.columns if "jpg" in str(c).lower() or "image" in str(c).lower()),None)
    id_col=next((c for c in df.columns if "sentinel_id" in str(c).lower() or "sentinel id" in str(c).lower()),None)

    if img_col is None or id_col is None:
        raise ValueError(f"Required columns not found. Available: {df.columns.tolist()}")

    splitter=GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    train_idx,val_idx=next(splitter.split(df,groups=df[id_col]))
    train_patches=set(df.iloc[train_idx][img_col].astype(str))
    val_patches=set(df.iloc[val_idx][img_col].astype(str))

    dataset=OilSpillDataset(root_dir=raw_dataset_dir)
    train_count=val_count=0

    for idx in range(len(dataset)):
        img_path=dataset.image_paths[idx]
        filename=img_path.name

        if filename in val_patches:
            split="val"
            val_count+=1
        elif filename in train_patches:
            split="train"
            train_count+=1
        else:
            print(f"Skipping {filename}: not found in metadata")
            continue

        img_tensor,target=dataset[idx]
        img_np=(img_tensor.permute(1,2,0).numpy()*255).astype("uint8")
        h,w=img_np.shape[:2]

        cv2.imwrite(str(out/"images"/split/filename),img_np)

        label_path=out/"labels"/split/f"{img_path.stem}.txt"
        with open(label_path,"w") as f:
            for box in target["boxes"].tolist():
                x,y,bw,bh=convert_to_yolo_format(box,w,h)
                f.write(f"0 {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"YOLO dataset generated! Train: {train_count}, Val: {val_count}")

if __name__=="__main__":
    ROOT=Path(__file__).resolve().parent.parent
    build_yolo_dataset(
        raw_dataset_dir=ROOT/"data",
        metadata_path=ROOT/"data"/"data_table.xlsx",
        output_dir=ROOT/"data"/"yolo_dataset"
    )