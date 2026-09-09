from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class OilSpillDataset(Dataset):
    def __init__(self,root_dir="C:/oilspill_project/data",transforms=None):
        self.root_dir=Path(root_dir)
        self.transforms=transforms
        self.image_paths=sorted([*self.root_dir.joinpath("oil").rglob("*.jpg"),*self.root_dir.joinpath("no_oil").rglob("*.jpg")])

    def apply_sigmoid_normalization(self,image):
        img=image.astype(np.float32)
        beta,alpha=np.median(img),3*np.std(img)
        if alpha==0:return image
        normalized=255/(1+np.exp(-(img-beta)/alpha))
        return np.clip(normalized,0,255).astype(np.uint8)

    def __getitem__(self,idx):
        img_path=self.image_paths[idx]
        filename=img_path.name
        img=cv2.imread(str(img_path),cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read image: {img_path}")

        img=self.apply_sigmoid_normalization(img)
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)

        boxes,labels=[],[]
        is_oil=filename.startswith(("oc-","ow-"))

        if is_oil:
            xml_path=img_path.with_suffix(".xml")

            if xml_path.exists():
                root=ET.parse(xml_path).getroot()

                for obj in root.findall("object"):
                    bbox=obj.find("bndbox")
                    if bbox is None:continue

                    xmin=float(bbox.find("xmin").text)
                    ymin=float(bbox.find("ymin").text)
                    xmax=float(bbox.find("xmax").text)
                    ymax=float(bbox.find("ymax").text)

                    boxes.append([xmin,ymin,xmax,ymax])
                    labels.append(0)

        target={
            "boxes":torch.tensor(boxes,dtype=torch.float32).reshape(-1,4),
            "labels":torch.tensor(labels,dtype=torch.int64)
        }

        if self.transforms:
            transformed=self.transforms(image=img,bboxes=boxes,class_labels=labels)
            img=transformed["image"]
            target["boxes"]=torch.tensor(transformed["bboxes"],dtype=torch.float32).reshape(-1,4)
            target["labels"]=torch.tensor(transformed["class_labels"],dtype=torch.int64)

        img=torch.tensor(img,dtype=torch.float32).permute(2,0,1)/255.0
        return img,target

    def __len__(self):
        return len(self.image_paths)

if __name__=="__main__":
    dataset=OilSpillDataset()
    print("Total images:",len(dataset))