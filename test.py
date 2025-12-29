import random
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import helper

# use Euclidean distance-based sampling (area-uniform radius)
helper.RANDOM_FLAG = 1

# Grid dimensions and render settings
width, height = 8, 6
scale_factor = 0.8  # Perspective scaling factor
fig_size = (8, 6)
output_dpi = 100

NUM_IMAGES = 100  # Number of non-black images to generate

# Allowed colors for box and ball
COLOR_CHOICES = ['green', 'red', 'blue', 'white']

# Prepare output folder and frame counter so filenames stay ordered
output_dir = os.path.join(os.path.dirname(__file__), 'rnd_img')
os.makedirs(output_dir, exist_ok=True)
frame_idx = 1

# Generate images according to NUM_IMAGES
for iteration in range(NUM_IMAGES):
    # Generate random positions for box and ball
    box_pos, ball_pos = helper.generate_positions(width, height)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # Draw grid and objects
    helper.draw_grid(ax, width, height, scale_factor)
    # Choose distinct colors for this frame
    box_color = COLOR_CHOICES[1]    # box_color = random.choice(COLOR_CHOICES)
    ball_color = COLOR_CHOICES[2]   #ball_color = random.choice(COLOR_CHOICES)
    # while ball_color == box_color:
        # ball_color = random.choice(COLOR_CHOICES)

    helper.draw_box(ax, box_pos, width, scale_factor, height, color=box_color)
    helper.draw_ball(ax, ball_pos, width, scale_factor, height, color=ball_color)
    
    # Setup axes appearance
    helper.setup_axes(ax, height)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the regular image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    output_file = os.path.join(output_dir, f'episode_{frame_idx:05d}_{timestamp}.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=output_dpi)
    print(f"[{iteration + 1}/{NUM_IMAGES}] Image saved to: {output_file}")
    
    # Close figure to free memory
    plt.close(fig)

    # Insert black image right after the regular one, using the same naming scheme for ordering
    black_image = np.zeros(
        (int(fig_size[1] * output_dpi), int(fig_size[0] * output_dpi), 3),
        dtype=np.uint8,
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    black_file = os.path.join(output_dir, f'episode_{frame_idx:05d}_{timestamp}.png')
    plt.imsave(black_file, black_image)
    print(f"    Inserted black image: {black_file}")
    frame_idx += 1

print(f"\nAll {NUM_IMAGES} images (with interleaved black frames) generated and saved!")
