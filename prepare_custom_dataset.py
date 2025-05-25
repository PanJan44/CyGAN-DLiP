import cv2
import os
import random
import shutil

anime_input_dir = "./Japanese-Anime-Scenes"
anime_output_dir = "./anime-resized/"
landscape_input_dir = "./landscape-pictures/"
landscape_output_dir = "./landscape-resized/"

anime_val_dir = "./anime-val/"
landscape_val_dir = "./landscape-val/"

resize_dim = (256, 256)

def resize(input_path: str, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filenames = os.listdir(input_path)
    exts = (".jpg", ".jpeg", ".png")
    for filename in filenames:
        if filename.endswith(exts):
          img = cv2.imread(os.path.join(os.path.abspath(input_path), filename))
          resized_img = cv2.resize(img, resize_dim)
          cv2.imwrite(os.path.join(output_dir, filename), resized_img)

def squeeze_landscape_dataset():
    print(f"anime dataset size : {len(os.listdir(anime_input_dir))}")
    os.makedirs(landscape_output_dir, exist_ok=True)
    selected_files_names = random.sample(os.listdir(landscape_input_dir), len(os.listdir(anime_input_dir)))
    for filename in selected_files_names:
      img = cv2.imread(os.path.join(os.path.abspath(landscape_input_dir), filename))
      resized_img = cv2.resize(img, resize_dim)
      cv2.imwrite(os.path.join(landscape_output_dir, filename), resized_img)

def divide_dataset(input_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    imgs_number = len(os.listdir(input_path))
    selected_imgs = random.sample(os.listdir(input_path), int(0.1*imgs_number))
    for img in selected_imgs:
        shutil.move(os.path.join(input_path, img), output_dir)

resize(anime_input_dir, anime_output_dir)
squeeze_landscape_dataset()
divide_dataset(anime_output_dir, anime_val_dir)
divide_dataset(landscape_output_dir, landscape_val_dir)
