import os

data_dir ="C:/Users/mihika/OneDrive/Desktop/data/raw_images"
for style in os.listdir(data_dir):
    count = len(os.listdir(os.path.join(data_dir, style)))
    print(f"{style}: {count} images")