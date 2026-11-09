import os
import shutil
import zipfile

dataset_dir = r"c:\Program-Bas\esp32cam\project\codes\dataset"
output_zip = r"c:\Program-Bas\esp32cam\project\codes\dataset_for_colab.zip"
staging_dir = r"c:\Program-Bas\esp32cam\project\codes\staging_colab"

if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)

os.makedirs(os.path.join(staging_dir, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(staging_dir, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(staging_dir, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(staging_dir, "val", "labels"), exist_ok=True)

# Copy train
shutil.copytree(os.path.join(dataset_dir, "train", "images"), os.path.join(staging_dir, "train", "images"), dirs_exist_ok=True)
shutil.copytree(os.path.join(dataset_dir, "train", "labels"), os.path.join(staging_dir, "train", "labels"), dirs_exist_ok=True)

# Copy val
shutil.copytree(os.path.join(dataset_dir, "val", "images"), os.path.join(staging_dir, "val", "images"), dirs_exist_ok=True)
shutil.copytree(os.path.join(dataset_dir, "val", "labels"), os.path.join(staging_dir, "val", "labels"), dirs_exist_ok=True)

# Write modified data.yaml for Google Colab
yaml_content = """path: /content/dataset
train: train/images
val: val/images

nc: 3
names: ['Sachet_20gr_LunchBox', 'Variant_YB77', 'Variant_BB725']
"""

with open(os.path.join(staging_dir, "data.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content)

# Zip
if os.path.exists(output_zip):
    os.remove(output_zip)

shutil.make_archive(r"c:\Program-Bas\esp32cam\project\codes\dataset_for_colab", 'zip', staging_dir)

shutil.rmtree(staging_dir)

size_mb = os.path.getsize(output_zip) / (1024 * 1024)
print(f"ZIP Berhasil Dibuat: dataset_for_colab.zip ({size_mb:.2f} MB)")
