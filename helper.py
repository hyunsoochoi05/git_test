import random
import numpy as np
import matplotlib.patches as patches
import math

# Sampling strategy flag: 0=stratified, 1=distance-based, 2=uniform
RANDOM_FLAG = 0

def generate_positions(width, height):
    """Generate positions for box and ball based on RANDOM_FLAG.
    
    Args:
        width, height: Grid dimensions
    
    Returns:
        Tuple of (box_pos, ball_pos) where each pos is (x, y)
    """
    if RANDOM_FLAG == 0:
        return _stratified_sampling(width, height)
    elif RANDOM_FLAG == 1:
        return _distance_based_sampling(width, height)
    else:  # RANDOM_FLAG == 2
        return _uniform_sampling(width, height)


def _uniform_sampling(width, height):
    """Uniform random sampling (original method)."""
    box_pos = (random.randint(0, width - 3), random.randint(0, height - 1))
    ball_pos = (random.randint(0, width - 3), random.randint(0, height - 1))
    
    # Ensure box and ball are at different positions
    while ball_pos == box_pos:
        ball_pos = (random.randint(0, width - 3), random.randint(0, height - 1))
    
    return box_pos, ball_pos

def _distance_based_sampling(width, height):
    """Polar distance+direction sampling:
    - sample one point uniformly (box)
    - sample distance and direction uniformly
    - compute second point (ball) from polar coords and clamp to grid
    """
    min_distance = 1.0
    # maximum possible Euclidean distance given the grid bounds
    max_distance = math.hypot(max(0, width - 3), max(0, height - 1))
    max_attempts = 200

    for _ in range(max_attempts):
        # sample base point (one object) uniformly as before
        x0 = random.randint(0, width - 3)
        y0 = random.randint(0, height - 1)

        # sample distance and direction uniformly
        dist = random.uniform(min_distance, max_distance)
        angle = random.uniform(-math.pi, math.pi)

        # compute target using polar -> cartesian, then round to nearest grid cell
        x1 = int(round(x0 + dist * math.cos(angle)))
        y1 = int(round(y0 + dist * math.sin(angle)))

        # clamp to valid grid ranges
        x1 = max(0, min(width - 3, x1))
        y1 = max(0, min(height - 1, y1))

        box_pos = (x0, y0)
        ball_pos = (x1, y1)

        if ball_pos != box_pos:
            return box_pos, ball_pos
    # fallback to uniform sampling if unable to place after attempts
        return _uniform_sampling(width, height)
    # ...existing code...


def _stratified_sampling(width, height):
    """Stratified sampling - divide grid into regions and sample uniformly from each."""
    # Divide grid into 2x2 regions
    regions_x, regions_y = 2, 2
    region_width = width / regions_x
    region_height = height / regions_y
    
    # Randomly select two different regions
    region_indices = list(range(regions_x * regions_y))
    random.shuffle(region_indices)
    box_region = region_indices[0]
    ball_region = region_indices[1]
    
    def get_position_in_region(region_idx):
        rx = region_idx % regions_x
        ry = region_idx // regions_x
        x_min = int(rx * region_width)
        x_max = int((rx + 1) * region_width) - 1
        y_min = int(ry * region_height)
        y_max = int((ry + 1) * region_height) - 1
        
        x = random.randint(max(0, x_min), min(width - 3, x_max))
        y = random.randint(max(0, y_min), min(height - 1, y_max))
        return (x, y)
    
    box_pos = get_position_in_region(box_region)
    ball_pos = get_position_in_region(ball_region)
    
    # Ensure they're different
    while ball_pos == box_pos:
        ball_region = random.randint(0, regions_x * regions_y - 1)
        ball_pos = get_position_in_region(ball_region)
    
    return box_pos, ball_pos


def _distance_based_sampling(width, height):
    """Distance-based sampling - ensure minimum distance between box and ball."""
    min_distance = 2
    max_distance = 10
    max_attempts = 100
    
    for _ in range(max_attempts):
        box_pos = (random.randint(0, width - 3), random.randint(0, height - 1))
        ball_pos = (random.randint(0, width - 3), random.randint(0, height - 1))
        
        # Calculate Manhattan distance
        distance = abs(box_pos[0] - ball_pos[0]) + abs(box_pos[1] - ball_pos[1])
        
        if min_distance <= distance <= max_distance:
            return box_pos, ball_pos
    
    # Fallback to uniform sampling if max attempts exceeded
    return _uniform_sampling(width, height)


def trapezoid_transform(x, y, width, height, scale_factor):
    """Apply trapezoid (perspective) transformation to coordinates.
    
    Args:
        x, y: Original coordinates
        width, height: Dimensions of the grid
        scale_factor: Right side vertical scale relative to left (>1 makes right longer)
    
    Returns:
        Tuple of transformed (x, y) coordinates
    """
    # Scale vertical coordinate depending on x so left columns are taller.
    scale_y = 1 - (1 - scale_factor) * (x / width)
    y_center = height / 2
    y_offset = (y - y_center) * scale_y + y_center
    
    # Compress horizontal spacing progressively
    x_compressed = x * (1 - 0.17 * (x / width))
    
    return x_compressed, y_offset


def _color_styles(color):
    """Return edge and text colors suitable for the given fill color."""
    if color == 'white':
        edge = 'black'
        text_color = 'black'
    else:
        edge = color
        text_color = 'white'
    return edge, text_color


def draw_grid(ax, width, height, scale_factor):
    """Draw perspective-transformed grid on axes.
    
    Args:
        ax: Matplotlib axis object
        width, height: Grid dimensions
        scale_factor: Perspective transformation factor
    """
    # Draw vertical grid lines (columns)
    for i in range(width + 1):
        x_vals = np.ones(200) * i
        y_vals = np.linspace(0, height, 200)
        x_transformed = []
        y_transformed = []
        for x, y in zip(x_vals, y_vals):
            x_t, y_t = trapezoid_transform(x, y, width, height, scale_factor)
            x_transformed.append(x_t)
            y_transformed.append(y_t)
        ax.plot(x_transformed, y_transformed, 'k-', linewidth=2)

    # Draw horizontal grid lines (rows)
    for j in range(height + 1):
        x_vals = np.linspace(0, width, 200)
        y_vals = np.ones(200) * j
        x_transformed = []
        y_transformed = []
        for x, y in zip(x_vals, y_vals):
            x_t, y_t = trapezoid_transform(x, y, width, height, scale_factor)
            x_transformed.append(x_t)
            y_transformed.append(y_t)
        ax.plot(x_transformed, y_transformed, 'k-', linewidth=2)


def draw_box(ax, pos, width, scale_factor, height, color='red'):
    """Draw a box (red rectangle) at the given position.
    
    Args:
        ax: Matplotlib axis object
        pos: (x, y) position tuple
        width: Grid width
        scale_factor: Perspective transformation factor
        height: Grid height
    """
    x_t, y_t = trapezoid_transform(pos[0], pos[1], width, height, scale_factor)
    
    edge_color, text_color = _color_styles(color)
    box_rect = patches.Rectangle(
        (x_t + 0.1, y_t + 0.1),
        width=0.8, height=0.8,
        linewidth=2, edgecolor=edge_color, facecolor=color, alpha=0.9
    )
    ax.add_patch(box_rect)
    ax.text(x_t + 0.5, y_t + 0.5, 'Box', ha='center', va='center',
            fontsize=10, fontweight='bold', color=text_color)


def draw_ball(ax, pos, width, scale_factor, height, color='blue'):
    """Draw a ball (blue circle) at the given position.
    
    Args:
        ax: Matplotlib axis object
        pos: (x, y) position tuple
        width: Grid width
        scale_factor: Perspective transformation factor
        height: Grid height
    """
    x_t, y_t = trapezoid_transform(pos[0], pos[1], width, height, scale_factor)
    
    edge_color, text_color = _color_styles(color)
    ball_circle = patches.Circle(
        (x_t + 0.5, y_t + 0.5),
        radius=0.35,
        linewidth=2, edgecolor=edge_color, facecolor=color, alpha=0.9
    )
    ax.add_patch(ball_circle)
    ax.text(x_t + 0.5, y_t + 0.5, 'Ball', ha='center', va='center',
            fontsize=8, fontweight='bold', color=text_color)


def setup_axes(ax, height):
    """Configure axes appearance."""
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
