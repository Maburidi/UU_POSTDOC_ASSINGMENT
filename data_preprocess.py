import os
from PIL import Image
import shutil
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Random split (80/20)
def split_and_copy(img_list, label, train_ratio=0.8):
    random.shuffle(img_list)
    split_idx = int(len(img_list) * train_ratio)
    train_imgs = img_list[:split_idx]
    test_imgs = img_list[split_idx:]

    for img_name in train_imgs:
        shutil.copy(
            os.path.join(source_folder, img_name),
            os.path.join(target_folder, "training", label, img_name)
        )

    for img_name in test_imgs:
        shutil.copy(
            os.path.join(source_folder, img_name),
            os.path.join(target_folder, "testing", label, img_name)
        )


folder_path = ["/content/UU_POSTDOC_ASSINGMENT/Data"]

for p in range(len(folder_path)):
    image_names = [f for f in os.listdir(folder_path[p]) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    x1, y1, x2, y2 = 140, 60, 240, 160
    
    for image_name in image_names:
        image_path = os.path.join(folder_path[p], image_name)
        image = Image.open(image_path).convert('RGB')
        cropped_image = image.crop((x1, y1, x2, y2))
        #image_path0 = '/content/data_1/' + image_name
        cropped_image.save(image_path)
        #print(f"Cropped and saved: {image_name}")


# Your source folder with all 64 images
source_folder = folder_path[0]
target_folder = "/content/data0"
os.makedirs(target_folder, exist_ok=True)

# Create class-wise folders inside training and testing
for split in ["training", "testing"]:
    for cls in ["1", "0"]:
        os.makedirs(os.path.join(target_folder, split, cls), exist_ok=True)

# Image ID groups
sintered_imgs = [f"{i}.jpg" for i in range(1, 33)]     # 1–32
unsintered_imgs = [f"{i}.jpg" for i in range(33, 65)]  # 33–64


# Split and copy both classes
split_and_copy(sintered_imgs, "1")
split_and_copy(unsintered_imgs, "0")

logging.info("✅ Dataset split completed and copied to training/testing folders.")


