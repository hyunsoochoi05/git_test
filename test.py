import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import helper

# --- Configuration ---
# Set sampling strategy: 0=stratified, 1=distance-based, 2=uniform, 3=sobol, 4=_sobol_relative_polar_sampling, 5=_stratified_polar_sampling
# We set it to 3 to use the new Sobol sequence sampler.
helper.RANDOM_FLAG = 3

# Grid dimensions and render settings
width, height = 8, 6
scale_factor = 0.8  # Perspective scaling factor
fig_size = (8, 6)
output_dpi = 100
NUM_IMAGES = 100  # Number of images to generate

# --- Main Execution ---

# Prepare output folder
output_dir = os.path.join(os.path.dirname(__file__), 'rnd_img')
os.makedirs(output_dir, exist_ok=True)
frame_idx = 1

print(f"Generating {NUM_IMAGES} images using Sobol sampling...")

# Generate images according to NUM_IMAGES
for iteration in range(NUM_IMAGES):
    # Generate a new pair of positions for box and ball using the selected sampler
    box_pos, ball_pos = helper.generate_positions(width, height)

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=fig_size)

    # Draw grid and objects
    helper.draw_grid(ax, width, height, scale_factor)
    helper.draw_box(ax, box_pos, width, scale_factor, height, color='red')
    helper.draw_ball(ax, ball_pos, width, scale_factor, height, color='blue')
    
    # Setup axes appearance
    helper.setup_axes(ax, height)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save the image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    output_file = os.path.join(output_dir, f'episode_{frame_idx:05d}_{timestamp}.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0, dpi=output_dpi)
    print(f"[{iteration + 1}/{NUM_IMAGES}] Image saved to: {output_file}")
    
    # Close figure to free memory
    plt.close(fig)
    frame_idx += 1

print(f"\nAll {NUM_IMAGES} images generated and saved in '{output_dir}'!")
