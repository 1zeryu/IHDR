import sys
sys.path.insert(0, "/mnt/pfs/users/yukun.zhou/codes/zyk-giga/giga-train/gt_projects/pipelines/GroundedSAM2")
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import decord
import grounding_dino.groundingdino.datasets.transforms as T
from PIL import Image
import numpy as np
from typing import Tuple
import torch


GROUNDING_DINO_CONFIG = "/mnt/pfs/users/yukun.zhou/codes/zyk-giga/giga-train/gt_projects/pipelines/GroundedSAM2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/mnt/pfs/users/yukun.zhou/codes/tools/GroundedSAM2/checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
PROMPT_TYPE_FOR_VIDEO = "box"  # ["box", "mask"]
TEXT_PROMPT = "human"


def load_image(image: Image) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(image, Image.Image):
        image = np.array(image)
    elif isinstance(image, bytes):
        image = BytesIO(image)
        image = Image.open(image).convert("RGB")
    elif isinstance(image, str):
        image = Image.open(image).convert("RGB")
    image_source = image
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed

class Detector():
    def __init__(self, device) -> None:
        self.device = device
        self.grounding_model = load_model(model_config_path=GROUNDING_DINO_CONFIG, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT, device=self.device)

    def detect(self, images):
        box_sequence = []
        if not isinstance(images, list):
            images = [images]
        for image in images:
            image_source, image_transformed = load_image(image)

            boxes, confidences, labels = predict(
                model=self.grounding_model,
                image=image_transformed,
                caption=TEXT_PROMPT,
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )
            box_sequence.append({"boxes": boxes, "confidences": confidences, "labels": labels})
        return box_sequence
    


    def convert_box(self, box):
        cx, cy, w, h = box
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
        return torch.tensor([x1, y1, x2, y2])
    
    def crop_image(self, image, boxes):
        cropped_images = []
        for i, box in enumerate(boxes[0]["boxes"]):
            box = box.tolist()
            # cx cy w h -> x1 y1 x2 y2
            box = self.convert_box(box)
            box = box.tolist()
            w, h = image.size
            box[0] = max(0, box[0] * w)
            box[1] = max(0, box[1] * h)
            box[2] = min(w, box[2] * w)
            box[3] = min(h, box[3] * h)
            cropped_image = image.crop(box)
            cropped_images.append(cropped_image)

        return cropped_images

if __name__ == "__main__":
    detector = Detector(device="cuda:1")
    boxes = detector.detect("/mnt/pfs/users/yukun.zhou/asserts/mimo效果测试/8.png")
    print(boxes)

    # 裁剪
    image = Image.open("/mnt/pfs/users/yukun.zhou/asserts/mimo效果测试/8.png")
    boxes = detector.crop_image(image, boxes)

