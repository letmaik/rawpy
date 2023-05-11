import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

BAYER_PATTERN = [
    [(255, 0, 0), (0, 255, 0)],  # Red, Green
    [(0, 255, 0), (0, 0, 255)]   # Green, Blue
]

def generate_bayer_pattern(width, height):
    img = np.empty((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            img[y, x] = BAYER_PATTERN[y % 2][x % 2]

    return img

def generate_text_image(text, font_path, font_size, image_size):
    # Create a new image with transparent background
    img = Image.new('RGBA', image_size)
    d = ImageDraw.Draw(img)

    # Load the font
    font = ImageFont.truetype(font_path, font_size)

    # Draw the text onto the image
    width, height = image_size
    d.text((width / 2, height / 2), text,
           fill=(0, 0, 0, 255), font=font, anchor='mm')

    return img

def generate_bayer_text_image(text, font_path, font_size, image_size):
    # Generate a Bayer pattern image of the desired size
    bayer_img = generate_bayer_pattern(*image_size)

    # Generate a text image of the same size
    text_img = generate_text_image(text, font_path, font_size, image_size)
    text_pixels = np.array(text_img)

    # Modulate the Bayer pattern by the alpha channel of the text image
    low_opacity = 0.3
    alpha = np.clip(text_pixels[..., 3] / 255.0, low_opacity, None)  # Normalize alpha to [0, 1]
    result_img = (bayer_img * alpha[..., None]).astype(np.uint8)

    return result_img


def render_as_isometric_voxels(img, path):
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
    plt.savefig(path, bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)

    # Close the plot
    plt.close(fig)


def rotate_image(in_path, out_path, angle):
    img = Image.open(in_path)

    img_rotated = img.rotate(angle,
                             fillcolor=(255, 255, 255, 0), 
                             resample=Image.Resampling.BICUBIC) 

    img_rotated.save(out_path)


def crop_image(in_path, out_path):
    img = Image.open(in_path)

    bbox = img.getbbox()
    img_cropped = img.crop(bbox)

    img_cropped.save(out_path)


result_img = generate_bayer_text_image('RAWPY', 'Inconsolata_ExtraExpanded-Black.ttf', 18, (66, 12))
Image.fromarray(result_img).save('logo_a.png')
render_as_isometric_voxels(result_img, 'logo_b.png')
rotate_image('logo_b.png', 'logo_c.png', 25.85)
crop_image('logo_c.png', 'logo_d.png')
