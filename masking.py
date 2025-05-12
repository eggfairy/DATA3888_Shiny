from PIL import Image, ImageDraw
import cv2
import numpy as np
import random

def centre_mask(img: Image.Image)-> Image.Image:
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = img.size[0] // 2, img.size[1] // 2
    radius = 80
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)

def non_centre_mask(img: Image.Image)-> Image.Image:
    mask = Image.new("L", img.size, 255)
    draw = ImageDraw.Draw(mask)
    cx, cy = img.size[0] // 2, img.size[1] // 2
    radius = 80
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=0)
    return Image.composite(img, Image.new("RGB", img.size, (0, 0, 0)), mask)

def random_mask(img: Image.Image)-> Image.Image:
    random.seed(1)
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    grid_rows, grid_cols = 10, 10
    cell_h, cell_w = h // grid_rows, w // grid_cols

    # Select 50 random grid cells to mask
    total_cells = [(i, j) for i in range(grid_rows) for j in range(grid_cols)]
    to_mask = random.sample(total_cells, k=len(total_cells) // 2)

    for i, j in to_mask:
        y_start, y_end = i * cell_h, (i + 1) * cell_h
        x_start, x_end = j * cell_w, (j + 1) * cell_w
        img_np[y_start:y_end, x_start:x_end] = 0  # black out

    return Image.fromarray(img_np)


def main():
    img = Image.open("example.png")
    c = centre_mask(img)
    n = non_centre_mask(img)
    r = random_mask(img)
    c.save("c.png")
    n.save("n.png")
    r.save("r.png")

if __name__ == "__main__":
    main()