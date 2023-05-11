from typing import Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

Pathy = Path | str

BAYER_PATTERN = [
    [(255, 0, 0), (0, 255, 0)],  # Red, Green
    [(0, 255, 0), (0, 0, 255)]   # Green, Blue
]

def generate_bayer_pattern(width: int, height: int) -> np.ndarray:
    img = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            img[y, x] = BAYER_PATTERN[y % 2][x % 2]

    return img


def generate_text_image(text: str, font_path: Pathy, font_size: int,
                        image_size: Tuple[int,int], offset_x: int, offset_y: int) -> np.ndarray:
    # Create a new image with transparent background
    img = Image.new('RGBA', image_size)
    d = ImageDraw.Draw(img)

    # Load the font
    font = ImageFont.truetype(str(font_path), font_size)

    # Draw the text onto the image
    width, height = image_size
    d.text((width / 2 + offset_x, height / 2 + offset_y), text,
           fill=(0, 0, 0, 255), font=font, anchor='mm')

    return np.ndarray(img)


def generate_bayer_text_image(out_path: Pathy, text: str, font_path: Pathy, font_size: int,
                              image_size: Tuple[int,int], offset_x: int=0, offset_y: int=0):
    # Generate a Bayer pattern image of the desired size
    bayer_img = generate_bayer_pattern(*image_size)

    # Generate a text image of the same size
    text_img = generate_text_image(text, font_path, font_size, image_size, offset_x, offset_y)

    # Modulate the Bayer pattern by the alpha channel of the text image
    low_opacity = 0.3
    alpha = np.clip(text_img[..., 3] / 255.0, low_opacity, None)  # Normalize alpha to [0, 1]
    result_img = (bayer_img * alpha[..., None]).astype(np.uint8)

    Image.fromarray(result_img).save(out_path)


def render_as_isometric_voxels(in_path: Pathy, out_path: Pathy, dpi: int):
    # Load image as numpy array
    img = np.array(Image.open(in_path))

    # Convert to the 0-1 range for matplotlib
    img_3d = img / 255

    # Create a new figure
    fig = plt.figure(figsize=(10, 10))

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection='3d', proj_type='ortho')

    # Get the dimensions of the image
    height, width, _ = img_3d.shape

    # Create empty 4D arrays for the voxel colors
    colors = np.empty((height, width, 1, 4))

    # Go through each pixel in the image
    for i in range(height):
        for j in range(width):
            # Get the color of the pixel
            color = img_3d[i, j]
            # Set the voxel color
            colors[i, j, 0, :3] = color  # set color
            colors[i, j, 0, 3] = 1  # set opacity

    # Draw the voxels
    filled = np.ones((height, width, 1), dtype=bool)
    ax.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.5, shade=True, lightsource=LightSource(azdeg=315, altdeg=25))

    # Set the viewing angle for an isometric projection
    ax.view_init(azim=40, elev=35.264)  # 35.264 is approximately arctan(1/sqrt(2))

    # Set the axes limits
    ax.set_aspect('equal')

    # Remove grid lines
    ax.grid(False)

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Remove axis lines
    ax.set_axis_off()

    # Save the plot
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=dpi)

    # Close the plot
    plt.close(fig)


def rotate_image(in_path: Pathy, out_path: Pathy, angle: float):
    img = Image.open(in_path)

    img_rotated = img.rotate(angle,
                             fillcolor=(255, 255, 255, 0), 
                             resample=Image.Resampling.BICUBIC) 

    img_rotated.save(out_path)


def crop_image_to_content(in_path: Pathy, out_path: Pathy):
    img = Image.open(in_path)

    bbox = img.getbbox()
    img_cropped = img.crop(bbox)

    img_cropped.save(out_path)


def resize_image(in_path: Pathy, out_path: Pathy, width: int):
    img = Image.open(in_path)

    height = int(width * img.height / img.width)
    img_resized = img.resize((width, height), resample=Image.LANCZOS)

    img_resized.save(out_path)


generate_bayer_text_image(out_path='logo_a.png', 
                          text='RAWPY', font_path='devin-chavez-font.ttf', font_size=14,
                          image_size=(58, 14), offset_x=1)
render_as_isometric_voxels(in_path='logo_a.png', out_path='logo_b.png', dpi=600)
rotate_image(in_path='logo_b.png', out_path='logo_c.png', angle=25.85)
crop_image_to_content(in_path='logo_c.png', out_path='logo_d.png')
resize_image(in_path='logo_d.png', out_path='logo.png', width=1280)
