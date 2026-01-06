import random
import numpy as np
import matplotlib.patches as patches
import math
import matplotlib.pyplot as plt
from scipy.stats import qmc
import itertools

# Sampling strategy flag: 0=stratified, 1=distance-based, 2=uniform, 3=sobol_cartesian, 4=sobol_polar, 5=stratified_polar, 6=stratified_difficulty, 7=stratified_hand_difficulty, 8=simple_random, 9=numpy_random, 10=redundancy_aware, 11=sobol_ball_diversity, 12=balanced_spatial, 13=edge_biased_spatial, 14=edge_biased_spatial_v2, 15=deterministic_spatial, 16=fully_deterministic_spatial, 17=fully_deterministic_spatial_v3
RANDOM_FLAG = 17

# Global counter to track fallback events in distance-based sampling
FALLBACK_COUNT = 0

# --- Sampler Definitions ---

import collections



# --- Original Sampling Functions (for comparison) ---
cartesian_sampler_4d = qmc.Sobol(d=4, scramble=True)
polar_sampler_4d = qmc.Sobol(d=4, scramble=True)

# (2)
def _uniform_sampling(width, height):
    """Uniform random sampling (With Replacement). May produce duplicates."""
    box_pos = (random.randint(0, width - 1), random.randint(0, height - 1))
    ball_pos = (random.randint(0, width - 1), random.randint(0, height - 1))
    while ball_pos == box_pos:
        ball_pos = (random.randint(0, width - 1), random.randint(0, height - 1))
    return box_pos, ball_pos

# (0)
def _stratified_sampling(width, height):
    """Stratified sampling on a 2x2 grid."""
    regions_x, regions_y = 2, 2
    region_width = width / regions_x
    region_height = height / regions_y
    region_indices = list(range(regions_x * regions_y))
    random.shuffle(region_indices)
    box_region_idx, ball_region_idx = region_indices[0], region_indices[1]
    
    def get_pos(region_idx):
        rx, ry = region_idx % regions_x, region_idx // regions_x
        x_min, x_max = int(rx * region_width), int((rx + 1) * region_width) - 1
        y_min, y_max = int(ry * region_height), int((ry + 1) * region_height) - 1
        return (random.randint(x_min, x_max), random.randint(y_min, y_max))
    
    return get_pos(box_region_idx), get_pos(ball_region_idx)

# (1)
def _distance_based_sampling(width, height, r_min=1.0, r_max=None, max_attempts=200):
    """Area-uniform radius sampling."""
    global FALLBACK_COUNT
    if r_max is None: r_max = math.hypot(width - 1, height - 1)
    for _ in range(max_attempts):
        x0, y0 = random.randint(0, width - 1), random.randint(0, height - 1)
        theta = random.uniform(0, 2 * math.pi)
        # Sample r such that points are uniformly distributed over the area (annulus).
        # If we sampled r uniformly (linear), points would cluster near the center.
        # PDF(r) ~ r, so CDF(r) ~ r^2. We sample from r^2 space and take sqrt.
        r = math.sqrt(random.uniform(r_min ** 2, r_max ** 2))
        x1, y1 = int(round(x0 + r * math.cos(theta))), int(round(y0 + r * math.sin(theta)))
        if 0 <= x1 < width and 0 <= y1 < height and (x1, y1) != (x0, y0):
            return (x0, y0), (x1, y1)
    FALLBACK_COUNT += 1
    return _uniform_sampling(width, height)

# (3)
def _sobol_cartesian_sampling(width, height):
    """Quasi-random sampling of Cartesian coordinates (x1, y1, x2, y2)."""
    while True:
        p = cartesian_sampler_4d.random(n=1)[0]
        x1, y1 = int(p[0] * width), int(p[1] * height)
        x2, y2 = int(p[2] * width), int(p[3] * height)
        p1 = (max(0, min(width - 1, x1)), max(0, min(height - 1, y1)))
        p2 = (max(0, min(width - 1, x2)), max(0, min(height - 1, y2)))
        if p1 != p2: return p1, p2

# (4)
def _sobol_relative_polar_sampling(width, height):
    """Uses rejection sampling on (box_x, box_y, r, theta) space."""
    # Calculate diagonal length to ensure coverage of the entire grid from any point
    r_max = math.hypot(width, height)
    
    while True:
        # Sample 4D point from Sobol sequence: [x_ratio, y_ratio, r_ratio, theta_ratio]
        p = polar_sampler_4d.random(n=1)[0]
        
        # 1. Determine Box position
        box_x_f, box_y_f = p[0] * width, p[1] * height
        
        # 2. Determine Ball position relative to Box using Polar coordinates
        r, theta = p[2] * r_max, p[3] * 2 * math.pi
        ball_x_f, ball_y_f = box_x_f + r * math.cos(theta), box_y_f + r * math.sin(theta)
        
        # 3. Rejection Sampling: Check if Ball is within grid bounds
        if not (0.0 <= ball_x_f < width and 0.0 <= ball_y_f < height):
            continue
            
        p1, p2 = (int(box_x_f), int(box_y_f)), (int(ball_x_f), int(ball_y_f))
        if p1 != p2: return p1, p2

# (5)
# --- Global Sampler Instances ---

# This global cache will store the expensive bin computation
STRATIFIED_CACHE = {}
# This global iterator will hold the generated samples for the main execution
_stratified_polar_sampler_iterator = None

# --- Correct Stratified Polar Sampling Implementation ---

def _initialize_stratified_polar_sampler(n_samples, width, height, num_r_bins, num_theta_bins):
    """
    Generates a uniformly distributed set of samples and initializes a global
    iterator to serve them one by one. This is the correct way to ensure a
    flat histogram.
    """
    global _stratified_polar_sampler_iterator
    
    grid_dims = (width, height)
    cache_key = (width, height, num_r_bins, num_theta_bins)

    if cache_key in STRATIFIED_CACHE:
        binned_pairs = STRATIFIED_CACHE[cache_key]
    else:
        print(f"Stratified sampling: Pre-computing all pairs for grid {grid_dims}...")
        binned_pairs = collections.defaultdict(list)
        all_points = list(itertools.product(range(width), range(height)))
        all_pairs = list(itertools.permutations(all_points, 2))

        # Use a slightly larger r_max to ensure coverage and avoid boundary issues with discrete distances
        r_max = math.hypot(width - 1, height - 1) + 1e-5
        r_bins = np.linspace(0, r_max, num_r_bins + 1)
        
        # Shift theta bins slightly to ensure cardinal directions (0, pi/2, etc.) fall clearly within bins
        theta_bins = np.linspace(-1e-5, 2 * np.pi - 1e-5, num_theta_bins + 1)
        
        for p1, p2 in all_pairs:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            r = math.hypot(dx, dy)
            if r == 0: continue
            
            theta = (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
            
            r_idx = np.digitize(r, r_bins) - 1
            theta_idx = np.digitize(theta, theta_bins) - 1
            
            if r_idx == num_r_bins: r_idx -= 1
            if theta_idx == num_theta_bins: theta_idx -= 1
            
            if 0 <= r_idx < num_r_bins and 0 <= theta_idx < num_theta_bins:
                 binned_pairs[(r_idx, theta_idx)].append((p1, p2))

        STRATIFIED_CACHE[cache_key] = binned_pairs
        print(f"Pre-computation complete. Found {len(binned_pairs)} non-empty bins.")

    # --- New Sampling Logic ---
    # Optimization: Iterative Proportional Fitting (IPF)
    # Goal: Keep R distribution perfectly uniform (Hard Constraint),
    #       while making Theta distribution as uniform as possible (Soft Constraint).
    
    # 1. Identify valid indices
    valid_r_indices = sorted(list(set(k[0] for k in binned_pairs.keys())))
    valid_theta_indices = sorted(list(set(k[1] for k in binned_pairs.keys())))
    
    # 2. Set targets
    target_r_count = n_samples / len(valid_r_indices)
    target_theta_count = n_samples / len(valid_theta_indices)

    # 3. Initialize weights
    weights = {k: 1.0 for k in binned_pairs.keys()}
    
    # 4. Run IPF iterations
    for _ in range(20):
        # Step A: Adjust for Theta uniformity (Soft)
        curr_theta_sums = collections.defaultdict(float)
        for (r, t), w in weights.items(): curr_theta_sums[t] += w
        for (r, t) in weights:
            if curr_theta_sums[t] > 0:
                weights[(r, t)] *= (target_theta_count / curr_theta_sums[t])

        # Step B: Enforce R uniformity (Hard - applied last)
        curr_r_sums = collections.defaultdict(float)
        for (r, t), w in weights.items(): curr_r_sums[r] += w
        for (r, t) in weights:
            if curr_r_sums[r] > 0:
                weights[(r, t)] *= (target_r_count / curr_r_sums[r])

    # 5. Convert weights to integer counts (Largest Remainder Method)
    final_counts = {k: int(w) for k, w in weights.items()}
    remainders = {k: w - final_counts[k] for k, w in weights.items()}
    needed = n_samples - sum(final_counts.values())
    
    for k, _ in sorted(remainders.items(), key=lambda x: x[1], reverse=True)[:needed]:
        final_counts[k] += 1
    
    # 6. Generate samples
    sampled_pairs = []
    for key, count in final_counts.items():
        if count > 0:
            pairs = binned_pairs[key]
            # Sample with replacement/cycling if needed
            chosen = [pairs[i % len(pairs)] for i in range(count)]
            random.shuffle(chosen) # Mix duplicates
            sampled_pairs.extend(chosen)
            
    random.shuffle(sampled_pairs)
            
    _stratified_polar_sampler_iterator = iter(sampled_pairs)

# (6)
def _initialize_stratified_difficulty_sampler(n_samples, width, height, num_r_bins=10, num_theta_bins=12):
    """
    Stratified Distance & Direction Sampling with Difficulty Sorting.
    1. Divides space into (r, theta) bins.
    2. Calculates a 'difficulty score' for every possible pair.
    3. Selects 'n_samples' distributed evenly across bins.
    4. Within each bin, picks the easiest tasks first.
    """
    global _stratified_polar_sampler_iterator
    
    # Cache key could be extended if needed, but reusing the structure for simplicity
    # or just re-computing since n_samples might change difficulty cutoff.
    # For this specific logic, we re-compute to ensure 'n_samples' affects the difficulty cut.
    
    print(f"Stratified Difficulty Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Generate all pairs and bins
    all_points = list(itertools.product(range(width), range(height)))
    all_pairs = list(itertools.permutations(all_points, 2))
    
    r_max = math.hypot(width - 1, height - 1) + 1e-5
    r_bins = np.linspace(0, r_max, num_r_bins + 1)
    theta_bins = np.linspace(-1e-5, 2 * np.pi - 1e-5, num_theta_bins + 1)
    
    binned_pairs = collections.defaultdict(list)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0

    for p1, p2 in all_pairs:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        r = math.hypot(dx, dy)
        if r == 0: continue
        
        theta = (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
        
        r_idx = np.digitize(r, r_bins) - 1
        theta_idx = np.digitize(theta, theta_bins) - 1
        
        if r_idx == num_r_bins: r_idx -= 1
        if theta_idx == num_theta_bins: theta_idx -= 1
        
        if 0 <= r_idx < num_r_bins and 0 <= theta_idx < num_theta_bins:
            # Calculate Difficulty Score
            # 1. Distance: Closer is easier (weight 1.0)
            # 2. Angle: Horizontal (0, 180) is easier than Vertical (90, 270).
            #    abs(sin(theta)) is 0 for horizontal, 1 for vertical. (weight 2.0)
            # 3. Centrality: Closer to center is easier. (weight 0.5)
            d1 = math.hypot(p1[0] - cx, p1[1] - cy)
            d2 = math.hypot(p2[0] - cx, p2[1] - cy)
            centrality = (d1 + d2) / 2.0
            
            difficulty = r + 2.0 * abs(math.sin(theta)) + 0.5 * centrality
            
            binned_pairs[(r_idx, theta_idx)].append({'pair': (p1, p2), 'score': difficulty})

    # 2. Sort pairs within each bin by difficulty (Ascending: Easy -> Hard)
    for key in binned_pairs:
        binned_pairs[key].sort(key=lambda x: x['score'])

    # 3. Determine target count per bin
    valid_bins = list(binned_pairs.keys())
    if not valid_bins:
        print("Error: No valid bins found.")
        _stratified_polar_sampler_iterator = iter([])
        return

    # Distribute n_samples as evenly as possible
    base_count = n_samples // len(valid_bins)
    remainder = n_samples % len(valid_bins)
    
    final_samples = []
    
    # To distribute remainder fairly, we can just give it to the first few bins 
    # or random bins. Here we just iterate.
    for i, bin_key in enumerate(valid_bins):
        count = base_count + (1 if i < remainder else 0)
        
        candidates = binned_pairs[bin_key]
        if not candidates: continue
        
        # Select top 'count' easiest tasks
        # If we need more than available, we cycle (reuse easiest ones)
        selected = []
        for k in range(count):
            item = candidates[k % len(candidates)]
            selected.append(item['pair'])
        
        final_samples.extend(selected)
    
    # Shuffle the final list so we don't get all Bin 0 tasks, then all Bin 1 tasks...
    random.shuffle(final_samples)
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Stratified Difficulty Sampling: Prepared {len(final_samples)} samples across {len(valid_bins)} bins.")

def _stratified_polar_sampling(width, height):
    """
    Pulls a single sample from the pre-generated global sample pool.
    This function's signature matches the other samplers.
    """
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Stratified sampler not initialized. Auto-initializing with default settings (N=1000)...")
        # Auto-initialize with reasonable defaults if forgotten
        _initialize_stratified_polar_sampler(
            n_samples=1000, 
            width=width, 
            height=height, 
            num_r_bins=10, 
            num_theta_bins=12
        )
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of stratified samples. This shouldn't happen if initialized correctly for the analysis script.")
        # Fallback to prevent crashing if used elsewhere
        return _uniform_sampling(width, height)

def _stratified_difficulty_sampling(width, height):
    """
    Wrapper for Flag 6. Uses the same iterator mechanism as Flag 5.
    """
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Stratified Difficulty sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_stratified_difficulty_sampler(
            n_samples=1000, 
            width=width, 
            height=height
        )
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (7)
def _initialize_stratified_hand_difficulty_sampler(n_samples, width, height, num_r_bins=10, num_theta_bins=12):
    """
    Stratified Distance & Direction Sampling with Hand-Proximity Difficulty Sorting.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Stratified Hand Difficulty Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Generate all pairs and bins
    all_points = list(itertools.product(range(width), range(height)))
    all_pairs = list(itertools.permutations(all_points, 2))
    
    r_max = math.hypot(width - 1, height - 1) + 1e-5
    r_bins = np.linspace(0, r_max, num_r_bins + 1)
    theta_bins = np.linspace(-1e-5, 2 * np.pi - 1e-5, num_theta_bins + 1)
    
    binned_pairs = collections.defaultdict(list)
    # cx, cy = (width - 1) / 2.0, (height - 1) / 2.0 # Used in (6)

    for p1, p2 in all_pairs:
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        r = math.hypot(dx, dy)
        if r == 0: continue
        
        theta = (math.atan2(dy, dx) + 2 * math.pi) % (2 * math.pi)
        
        r_idx = np.digitize(r, r_bins) - 1
        theta_idx = np.digitize(theta, theta_bins) - 1
        
        if r_idx == num_r_bins: r_idx -= 1
        if theta_idx == num_theta_bins: theta_idx -= 1
        
        if 0 <= r_idx < num_r_bins and 0 <= theta_idx < num_theta_bins:
            # Calculate Difficulty Score
            
            # --- Original (6) Code ---
            # d1 = math.hypot(p1[0] - cx, p1[1] - cy)
            # d2 = math.hypot(p2[0] - cx, p2[1] - cy)
            # centrality = (d1 + d2) / 2.0
            # difficulty = r + 2.0 * abs(math.sin(theta)) + 0.5 * centrality
            
            # --- New (7) Code ---
            # Hands at bottom edge, approx x=1 and x=6 (indices)
            hand_l = (1, height - 1)
            hand_r = (width - 2, height - 1)
            
            d_box = min(math.hypot(p1[0] - hand_l[0], p1[1] - hand_l[1]), 
                        math.hypot(p1[0] - hand_r[0], p1[1] - hand_r[1]))
            d_ball = min(math.hypot(p2[0] - hand_l[0], p2[1] - hand_l[1]), 
                         math.hypot(p2[0] - hand_r[0], p2[1] - hand_r[1]))
            
            # Closer to hands is easier (smaller d_box/d_ball)
            difficulty = r + 2.0 * abs(math.sin(theta)) + 0.5 * (d_box + d_ball)
            
            binned_pairs[(r_idx, theta_idx)].append({'pair': (p1, p2), 'score': difficulty})

    # 2. Sort pairs within each bin by difficulty (Ascending: Easy -> Hard)
    for key in binned_pairs:
        binned_pairs[key].sort(key=lambda x: x['score'])

    # 3. Determine target count per bin
    valid_bins = list(binned_pairs.keys())
    if not valid_bins:
        print("Error: No valid bins found.")
        _stratified_polar_sampler_iterator = iter([])
        return

    # Distribute n_samples as evenly as possible
    base_count = n_samples // len(valid_bins)
    remainder = n_samples % len(valid_bins)
    
    final_samples = []
    
    for i, bin_key in enumerate(valid_bins):
        count = base_count + (1 if i < remainder else 0)
        
        candidates = binned_pairs[bin_key]
        if not candidates: continue
        
        # Select top 'count' easiest tasks
        selected = []
        for k in range(count):
            item = candidates[k % len(candidates)]
            selected.append(item['pair'])
        
        final_samples.extend(selected)
    
    random.shuffle(final_samples)
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Stratified Hand Difficulty Sampling: Prepared {len(final_samples)} samples across {len(valid_bins)} bins.")

def _stratified_hand_difficulty_sampling(width, height):
    """
    Wrapper for Flag 7.
    """
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Stratified Hand Difficulty sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_stratified_hand_difficulty_sampler(
            n_samples=1000, 
            width=width, 
            height=height
        )
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (8)
def _initialize_simple_random_sampler(n_samples, width, height):
    """
    Simple Random Sampling (Flag 8).
    Selects 'n_samples' randomly from all possible pairs without replacement.
    Does not use stratification or difficulty sorting.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Simple Random Sampling: Computing pairs for grid ({width}x{height})...")
    
    all_points = list(itertools.product(range(width), range(height)))
    all_pairs = list(itertools.permutations(all_points, 2))
    
    if len(all_pairs) <= n_samples:
        selected = all_pairs
        random.shuffle(selected)
    else:
        selected = random.sample(all_pairs, n_samples)
        
    _stratified_polar_sampler_iterator = iter(selected)
    print(f"Simple Random Sampling: Prepared {len(selected)} samples.")

def _simple_random_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Simple Random sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_simple_random_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (9)
def _initialize_numpy_random_sampler(n_samples, width, height):
    """
    Numpy Random Sampling (Flag 9).
    Uses NumPy to shuffle indices for sampling.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Numpy Random Sampling: Computing pairs for grid ({width}x{height})...")
    
    all_points = list(itertools.product(range(width), range(height)))
    all_pairs = list(itertools.permutations(all_points, 2))
    
    if len(all_pairs) <= n_samples:
        selected = all_pairs
        indices = np.arange(len(selected))
        np.random.shuffle(indices)
        selected = [selected[i] for i in indices]
    else:
        indices = np.arange(len(all_pairs))
        np.random.shuffle(indices)
        selected_indices = indices[:n_samples]
        selected = [all_pairs[i] for i in selected_indices]
        
    _stratified_polar_sampler_iterator = iter(selected)
    print(f"Numpy Random Sampling: Prepared {len(selected)} samples.")

def _numpy_random_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Numpy Random sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_numpy_random_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (10)
def _initialize_redundancy_aware_sampler(n_samples, width, height):
    """
    Redundancy-Aware Sampling (Flag 10).
    Implements the 'Optimization Prompt' logic:
    1. Prioritizes Anchor Points (9 key spots) for Box.
    2. Prioritizes Long Trajectories.
    3. Filters out samples with high Positional & Vector similarity.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Redundancy-Aware Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Define Anchor Points (9 points: Corners, Mid-edges, Center)
    anchors = set()
    xs = [0, width // 2, width - 1]
    ys = [0, height // 2, height - 1]
    for x in xs:
        for y in ys:
            anchors.add((x, y))
            
    # 2. Generate all candidates with Scores
    all_points = list(itertools.product(range(width), range(height)))
    all_pairs = list(itertools.permutations(all_points, 2))
    
    candidates = []
    max_dist = math.hypot(width, height)
    
    for p1, p2 in all_pairs:
        # p1: Box, p2: Ball
        vec = (p2[0] - p1[0], p2[1] - p1[1])
        traj_len = math.hypot(vec[0], vec[1])
        angle = math.atan2(vec[1], vec[0])
        
        # Score Calculation
        # A. Anchor Priority
        is_anchor = 1.0 if p1 in anchors else 0.0
        
        # B. Trajectory Length (Longer is better)
        score_len = traj_len / max_dist
        
        # Total Score (Weights: Anchor > Length)
        final_score = (is_anchor * 2.0) + (score_len * 1.5)
        # Add random noise to make the priority 'soft' rather than 'hard'.
        noise = random.uniform(0, 2.0)
        final_score = (is_anchor * 2.0) + (score_len * 1.5) + noise
        
        candidates.append({
            'pair': (p1, p2),
            'box': p1,
            'vec': vec,
            'angle': angle,
            'score': final_score
        })
        
    # Sort by score descending (Best candidates first)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    # 3. Greedy Selection with Redundancy Filtering
    selected_items = []
    rejected_items = []
    
    # Thresholds for redundancy (Tunable)
    thresh_pos = 1.5  # Box position similarity threshold
    thresh_vec = 0.9  # Relative vector similarity threshold (Lowered to distinguish dist 1 from 2)
    thresh_angle = 0.1 # Angle similarity threshold (radians)
    thresh_pos_parallel = 2.5 # Stricter position threshold for parallel vectors
    
    for cand in candidates:
        is_redundant = False
        # Check against currently selected items
        for s in selected_items:
            d_pos = math.hypot(cand['box'][0] - s['box'][0], cand['box'][1] - s['box'][1])
            d_vec = math.hypot(cand['vec'][0] - s['vec'][0], cand['vec'][1] - s['vec'][1])
            
            if d_pos < thresh_pos and d_vec < thresh_vec:
                is_redundant = True
                break
            
            # Check for parallel vectors (Same Direction)
            angle_diff = abs(cand['angle'] - s['angle'])
            if angle_diff > math.pi: angle_diff = 2*math.pi - angle_diff
            
            if angle_diff < thresh_angle and d_pos < thresh_pos_parallel:
                is_redundant = True
                break
        
        if not is_redundant:
            selected_items.append(cand)
        else:
            rejected_items.append(cand)
            
        if len(selected_items) >= n_samples:
            break
            
    # Fill from rejected pool if not enough non-redundant samples found
    if len(selected_items) < n_samples:
        needed = n_samples - len(selected_items)
        print(f"Notice: Only found {len(selected_items)} unique samples. Filling {needed} from rejected pool.")
        selected_items.extend(rejected_items[:needed])
        
    final_samples = [x['pair'] for x in selected_items]
    random.shuffle(final_samples) # Shuffle for IID training
    
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Redundancy-Aware Sampling: Prepared {len(final_samples)} samples.")

def _redundancy_aware_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Redundancy-Aware sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_redundancy_aware_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (11)
def _initialize_sobol_ball_diversity_sampler(n_samples, width, height):
    """
    Sobol Ball + Diversity Box Sampling (Flag 11).
    1. Ball positions: Generated via Sobol sequence (Uniform distribution).
    2. Box positions: Selected to maximize vector diversity (Least-used vector first).
    3. No long-path priority.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Sobol Ball Diversity Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Generate Ball positions using Sobol (d=2)
    sampler = qmc.Sobol(d=2, scramble=True)
    # Generate samples. Sobol guarantees uniform stratification.
    sobol_points = sampler.random(n_samples)
    
    ball_positions = []
    for p in sobol_points:
        bx = int(p[0] * width)
        by = int(p[1] * height)
        # Clamp to ensure within bounds
        bx = min(max(bx, 0), width - 1)
        by = min(max(by, 0), height - 1)
        ball_positions.append((bx, by))
        
    # 2. Select Box for each Ball to maximize vector diversity
    # Track vector usage counts: (dx, dy) -> count
    vector_counts = collections.defaultdict(int)
    final_samples = []
    all_grid_points = list(itertools.product(range(width), range(height)))
    
    for ball_pos in ball_positions:
        # Shuffle candidates to break ties randomly
        current_candidates = list(all_grid_points)
        random.shuffle(current_candidates)
        
        best_box = None
        min_count = float('inf')
        
        for box_pos in current_candidates:
            if box_pos == ball_pos: continue
            
            vec = (ball_pos[0] - box_pos[0], ball_pos[1] - box_pos[1])
            count = vector_counts[vec]
            
            if count < min_count:
                min_count = count
                best_box = box_pos
                if min_count == 0: break # Found unused vector, take it
        
        if best_box is not None:
            final_samples.append((best_box, ball_pos))
            vec = (ball_pos[0] - best_box[0], ball_pos[1] - best_box[1])
            vector_counts[vec] += 1
            
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Sobol Ball Diversity Sampling: Prepared {len(final_samples)} samples.")

def _sobol_ball_diversity_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Sobol Ball Diversity sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_sobol_ball_diversity_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (12)
def _initialize_balanced_spatial_sampler(n_samples, width, height):
    """
    Balanced Spatial Sampling (Flag 12).
    Balances between Vector Diversity and Box Spatial Uniformity.
    Method 11 tends to cluster boxes at edges to satisfy rare vectors.
    Method 12 adds a penalty for repeated box positions to encourage spatial uniformity.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Balanced Spatial Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Generate Ball positions using Sobol (d=2)
    sampler = qmc.Sobol(d=2, scramble=True)
    sobol_points = sampler.random(n_samples)
    
    ball_positions = []
    for p in sobol_points:
        bx = int(p[0] * width)
        by = int(p[1] * height)
        bx = min(max(bx, 0), width - 1)
        by = min(max(by, 0), height - 1)
        ball_positions.append((bx, by))
        
    # 2. Select Box for each Ball
    vector_counts = collections.defaultdict(int)
    box_counts = collections.defaultdict(int)
    final_samples = []
    all_grid_points = list(itertools.product(range(width), range(height)))
    
    for ball_pos in ball_positions:
        current_candidates = list(all_grid_points)
        random.shuffle(current_candidates)
        
        best_box = None
        min_score = float('inf')
        
        for box_pos in current_candidates:
            if box_pos == ball_pos: continue
            
            vec = (ball_pos[0] - box_pos[0], ball_pos[1] - box_pos[1])
            dist = math.hypot(vec[0], vec[1])
            
            # Score minimizes both vector usage and box usage.
            # (v_count + b_count) ensures we pick rare vectors AND rare box positions.
            # dist is added as a tie-breaker. Since short vectors (e.g. dist=1) are geometrically rare,
            # we must prioritize them when counts are tied to ensure they are included (preventing starvation).
            score = (vector_counts[vec] + box_counts[box_pos]) #* 1000 + dist
            
            if score < min_score:
                min_score = score
                best_box = box_pos
        
        if best_box is not None:
            final_samples.append((best_box, ball_pos))
            vec = (ball_pos[0] - best_box[0], ball_pos[1] - best_box[1])
            vector_counts[vec] += 1
            box_counts[best_box] += 1
            
    random.shuffle(final_samples)
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Balanced Spatial Sampling: Prepared {len(final_samples)} samples.")

def _balanced_spatial_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Balanced Spatial sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_balanced_spatial_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (13)
def _initialize_edge_biased_spatial_sampler(n_samples, width, height):
    """
    Edge-Biased Spatial Sampling (Flag 13) - Original Version.
    1. Samples Ball positions using a Beta(0.5, 0.5) distribution to favor edges/corners.
       This ensures we have candidates capable of forming long/rare vectors.
    2. Selects Box positions greedily to minimize vector count variance.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Edge-Biased Spatial Sampling: Computing pairs for grid ({width}x{height})...")
    
    ball_positions = []
    # Beta(0.5, 0.5) is U-shaped, favoring 0 and 1 (edges).
    alpha, beta = 0.5, 0.5
    
    for _ in range(n_samples):
        # Beta distribution returns [0, 1]. Map to [0, width-1].
        bx = int(random.betavariate(alpha, beta) * width)
        by = int(random.betavariate(alpha, beta) * height)
        bx = min(max(bx, 0), width - 1)
        by = min(max(by, 0), height - 1)
        ball_positions.append((bx, by))
        
    vector_counts = collections.defaultdict(int)
    box_counts = collections.defaultdict(int)
    final_samples = []
    all_grid_points = list(itertools.product(range(width), range(height)))
    
    for ball_pos in ball_positions:
        # Shuffle to ensure random selection among equal scores
        current_candidates = list(all_grid_points)
        random.shuffle(current_candidates)
        
        best_box = None
        min_score = float('inf')
        
        for box_pos in current_candidates:
            if box_pos == ball_pos: continue
            
            vec_x = ball_pos[0] - box_pos[0]
            vec_y = ball_pos[1] - box_pos[1]
            vec = (vec_x, vec_y)
            
            # Hierarchical Score:
            # 1. Vector Count (Primary): We want flat vector histogram.
            # 2. Box Count (Secondary): We want to use different box positions if possible.
            # 3. Random (Tertiary): Break ties randomly (crucial to avoid geometric bias).
            score = (vector_counts[vec] * 10000) + (box_counts[box_pos] * 10) + random.random()
            
            if score < min_score:
                min_score = score
                best_box = box_pos
        
        if best_box is not None:
            final_samples.append((best_box, ball_pos))
            vec = (ball_pos[0] - best_box[0], ball_pos[1] - best_box[1])
            vector_counts[vec] += 1
            box_counts[best_box] += 1
            
    random.shuffle(final_samples)
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Edge-Biased Spatial Sampling: Prepared {len(final_samples)} samples.")

def _edge_biased_spatial_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Edge-Biased Spatial sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_edge_biased_spatial_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (14)
def _initialize_edge_biased_spatial_sampler_v2(n_samples, width, height):
    """
    Edge-Biased Spatial Sampling V2 (Flag 14) - Improved Version.
    Includes 'availability' weighting to prioritize geometrically rare vectors (long distances)
    when ball is at the edge.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Edge-Biased Spatial Sampling V2: Computing pairs for grid ({width}x{height})...")
    
    ball_positions = []
    # Relaxed Beta parameters from 0.5 to 0.75 to mitigate excessive corner clustering
    alpha, beta = 0.85, 0.85
    
    for _ in range(n_samples):
        bx = int(random.betavariate(alpha, beta) * width)
        by = int(random.betavariate(alpha, beta) * height)
        bx = min(max(bx, 0), width - 1)
        by = min(max(by, 0), height - 1)
        ball_positions.append((bx, by))
        
    vector_counts = collections.defaultdict(int)
    box_counts = collections.defaultdict(int)
    final_samples = []
    all_grid_points = list(itertools.product(range(width), range(height)))
    
    for ball_pos in ball_positions:
        current_candidates = list(all_grid_points)
        random.shuffle(current_candidates)
        
        best_box = None
        min_score = float('inf')
        
        for box_pos in current_candidates:
            if box_pos == ball_pos: continue
            
            vec_x = ball_pos[0] - box_pos[0]
            vec_y = ball_pos[1] - box_pos[1]
            vec = (vec_x, vec_y)
            
            availability = (width - abs(vec_x)) * (height - abs(vec_y))
            
            # Increased vector_counts weight (100 -> 1000) to ensure we fill empty bins (count 0)
            # before repeating rare vectors. Max availability is approx W*H (e.g., 48),
            # so weight must be > 48*3 to prevent availability from overriding count.
            score = (vector_counts[vec] * 1000) + (availability * 3) + (box_counts[box_pos] * 1) + random.random()
            
            if score < min_score:
                min_score = score
                best_box = box_pos
        
        if best_box is not None:
            final_samples.append((best_box, ball_pos))
            vec = (ball_pos[0] - best_box[0], ball_pos[1] - best_box[1])
            vector_counts[vec] += 1
            box_counts[best_box] += 1
            
    random.shuffle(final_samples)
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Edge-Biased Spatial Sampling V2: Prepared {len(final_samples)} samples.")

def _edge_biased_spatial_sampling_v2(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Edge-Biased Spatial V2 sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_edge_biased_spatial_sampler_v2(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (15)
def _initialize_deterministic_spatial_sampler(n_samples, width, height):
    """
    Deterministic Spatial Sampling (Flag 15).
    Completely removes randomness to achieve maximum uniformity for both
    Relative Positions (Vectors) and Absolute Positions (Box/Ball).
    
    Strategy:
    1. Enumerate all possible vectors and assign target counts (N / |V|).
    2. Sort tasks by 'availability' (rare vectors first).
    3. Greedily select positions to minimize total occupancy of grid cells.
    4. Use deterministic tie-breaking (round-robin offset) to spread samples.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Deterministic Spatial Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Identify all valid vectors and their availability
    vectors = []
    for dy in range(-(height - 1), height):
        for dx in range(-(width - 1), width):
            if dx == 0 and dy == 0: continue
            # Availability: Number of valid positions for this vector
            avail = (width - abs(dx)) * (height - abs(dy))
            vectors.append({'vec': (dx, dy), 'avail': avail})
            
    # 2. Determine counts per vector
    # Sort by availability (ascending) to handle remainder distribution logic
    vectors.sort(key=lambda x: x['avail'])
    
    num_vectors = len(vectors)
    base_count = n_samples // num_vectors
    remainder = n_samples % num_vectors
    
    tasks = []
    for i, item in enumerate(vectors):
        count = base_count
        # Distribute remainder to vectors with HIGHEST availability (end of list).
        # These are short vectors that are flexible and easy to place without clustering.
        if i >= (num_vectors - remainder):
            count += 1
        for _ in range(count):
            tasks.append(item)
            
    # 3. Sort tasks by availability (Rare First) for processing
    # Processing rare vectors first forces them to take edges.
    # Then common vectors will naturally fill the center to balance absolute usage.
    tasks.sort(key=lambda x: x['avail'])
    
    # 4. Greedy Filling
    # Tracks how many times a grid cell has been used (as either box or ball)
    position_counts = collections.defaultdict(int)
    final_samples = []
    
    for i, task in enumerate(tasks):
        dx, dy = task['vec']
        
        # Find all valid boxes for this vector
        min_bx, max_bx = max(0, -dx), min(width, width - dx)
        min_by, max_by = max(0, -dy), min(height, height - dy)
        
        valid_boxes = []
        for by in range(min_by, max_by):
            for bx in range(min_bx, max_bx):
                valid_boxes.append((bx, by))
        
        best_pair = None
        min_score = float('inf')
        
        # Deterministic Tie-Breaker:
        # Start search from a different index each time to avoid clustering.
        # Using a prime multiplier ensures we cycle through candidates.
        offset = (i * 7919) % len(valid_boxes)
        
        for k in range(len(valid_boxes)):
            idx = (k + offset) % len(valid_boxes)
            bx, by = valid_boxes[idx]
            lx, ly = bx + dx, by + dy
            
            # Score: Minimize total usage of these two cells
            score = position_counts[(bx, by)] + position_counts[(lx, ly)]
            
            if score < min_score:
                min_score = score
                best_pair = ((bx, by), (lx, ly))
        
        if best_pair:
            box, ball = best_pair
            final_samples.append((box, ball))
            position_counts[box] += 1
            position_counts[ball] += 1
            
    # 5. Deterministic Shuffle
    # The list is currently sorted by vector length (rare first).
    # We shuffle it deterministically to mix them up for the iterator.
    rng = random.Random(42)
    rng.shuffle(final_samples)
    
    _stratified_polar_sampler_iterator = iter(final_samples)
    print(f"Deterministic Spatial Sampling: Prepared {len(final_samples)} samples.")

def _deterministic_spatial_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Deterministic Spatial sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_deterministic_spatial_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (16)
def _initialize_fully_deterministic_spatial_sampler(n_samples, width, height):
    """
    Fully Deterministic Spatial Sampling (Flag 16).
    Improvements over Flag 15:
    1. Score function minimizes sum of squares (L2) instead of sum (L1) to enforce
       individual uniformity for both Box and Ball positions.
    2. Removes 'random.shuffle' entirely. Uses a deterministic prime stride
       to mix the samples (which are generated in rare-first order).
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Fully Deterministic Spatial Sampling: Computing pairs for grid ({width}x{height})...")
    
    # 1. Identify all valid vectors and their availability
    vectors = []
    for dy in range(-(height - 1), height):
        for dx in range(-(width - 1), width):
            if dx == 0 and dy == 0: continue
            avail = (width - abs(dx)) * (height - abs(dy))
            vectors.append({'vec': (dx, dy), 'avail': avail})
            
    # 2. Determine counts per vector
    vectors.sort(key=lambda x: x['avail'])
    
    num_vectors = len(vectors)
    base_count = n_samples // num_vectors
    remainder = n_samples % num_vectors
    
    tasks = []
    for i, item in enumerate(vectors):
        count = base_count
        if i >= (num_vectors - remainder):
            count += 1
        for _ in range(count):
            tasks.append(item)
            
    # 3. Sort tasks by availability (Rare First)
    tasks.sort(key=lambda x: x['avail'])
    
    # 4. Greedy Filling with L2 Score
    position_counts = collections.defaultdict(int)
    final_samples = []
    
    for i, task in enumerate(tasks):
        dx, dy = task['vec']
        
        min_bx, max_bx = max(0, -dx), min(width, width - dx)
        min_by, max_by = max(0, -dy), min(height, height - dy)
        
        valid_boxes = []
        for by in range(min_by, max_by):
            for bx in range(min_bx, max_bx):
                valid_boxes.append((bx, by))
        
        best_pair = None
        min_score = float('inf')
        
        # Deterministic Tie-Breaker Offset
        offset = (i * 7919) % len(valid_boxes)
        
        for k in range(len(valid_boxes)):
            idx = (k + offset) % len(valid_boxes)
            bx, by = valid_boxes[idx]
            lx, ly = bx + dx, by + dy
            
            # Score: Sum of squares penalizes peaks more than simple sum.
            # This ensures both box and ball distributions remain flat.
            c_box = position_counts[(bx, by)]
            c_ball = position_counts[(lx, ly)]
            score = (c_box * c_box) + (c_ball * c_ball)
            
            if score < min_score:
                min_score = score
                best_pair = ((bx, by), (lx, ly))
        
        if best_pair:
            box, ball = best_pair
            final_samples.append((box, ball))
            position_counts[box] += 1
            position_counts[ball] += 1
            
    # 5. Deterministic Mixing (No Random Shuffle)
    # The list 'final_samples' is sorted by vector rarity (Rare -> Common).
    # We want to disperse these rare vectors uniformly throughout the sequence.
    # We use a prime stride to map the sorted index 'i' to a destination index.
    stride = 997
    while math.gcd(stride, n_samples) != 1:
        stride += 1
        
    mixed_samples = [None] * n_samples
    for i in range(n_samples):
        # Scatter the i-th sorted sample to a strided position
        dest_idx = (i * stride) % n_samples
        mixed_samples[dest_idx] = final_samples[i]
    
    _stratified_polar_sampler_iterator = iter(mixed_samples)
    print(f"Fully Deterministic Spatial Sampling: Prepared {len(mixed_samples)} samples.")

def _fully_deterministic_spatial_sampling(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Fully Deterministic Spatial sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_fully_deterministic_spatial_sampler(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)

# (17)
def _initialize_fully_deterministic_spatial_sampler_v3(n_samples, width, height):
    """
    Fully Deterministic Spatial Sampling V3 (Flag 17).
    Optimizes for absolute position uniformity using Cubic Scoring and Center-Preference.
    
    Changes from Flag 16:
    1. Score = (box_count^3 + ball_count^3). Penalizes peaks more aggressively.
    2. Adds a 'Center Preference' tie-breaker. When counts are similar, we prefer
       placing flexible vectors near the center. This reserves the edges for
       the rare vectors that strictly require them, preventing edge overcrowding.
    """
    global _stratified_polar_sampler_iterator
    
    print(f"Fully Deterministic Spatial Sampling V3: Computing pairs for grid ({width}x{height})...")
    
    # 1. Vectors & Availability
    vectors = []
    for dy in range(-(height - 1), height):
        for dx in range(-(width - 1), width):
            if dx == 0 and dy == 0: continue
            avail = (width - abs(dx)) * (height - abs(dy))
            vectors.append({'vec': (dx, dy), 'avail': avail})
            
    vectors.sort(key=lambda x: x['avail'])
    
    # 2. Counts
    num_vectors = len(vectors)
    base_count = n_samples // num_vectors
    remainder = n_samples % num_vectors
    
    tasks = []
    for i, item in enumerate(vectors):
        count = base_count
        if i >= (num_vectors - remainder):
            count += 1
        for _ in range(count):
            tasks.append(item)
            
    # 3. Sort Rare First
    tasks.sort(key=lambda x: x['avail'])
    
    # 4. Greedy with Cubic Score + Center Pref
    position_counts = collections.defaultdict(int)
    final_samples = []
    
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    
    for i, task in enumerate(tasks):
        dx, dy = task['vec']
        min_bx, max_bx = max(0, -dx), min(width, width - dx)
        min_by, max_by = max(0, -dy), min(height, height - dy)
        
        valid_boxes = []
        for by in range(min_by, max_by):
            for bx in range(min_bx, max_bx):
                valid_boxes.append((bx, by))
        
        best_pair = None
        min_score = float('inf')
        
        offset = (i * 7919) % len(valid_boxes)
        
        for k in range(len(valid_boxes)):
            idx = (k + offset) % len(valid_boxes)
            bx, by = valid_boxes[idx]
            lx, ly = bx + dx, by + dy
            
            c_box = position_counts[(bx, by)]
            c_ball = position_counts[(lx, ly)]
            
            # Cubic Score: Stronger penalty for peaks than Square
            score = (c_box ** 3) + (c_ball ** 3)
            
            # Center Preference Tie-breaker:
            # If scores are roughly equal, prefer positions closer to center.
            # This saves the edges for the rare vectors that MUST use them.
            d_box = abs(bx - cx) + abs(by - cy)
            d_ball = abs(lx - cx) + abs(ly - cy)
            score += (d_box + d_ball) * 0.01
            
            if score < min_score:
                min_score = score
                best_pair = ((bx, by), (lx, ly))
        
        if best_pair:
            box, ball = best_pair
            final_samples.append((box, ball))
            position_counts[box] += 1
            position_counts[ball] += 1
            
    # 5. Deterministic Mixing
    stride = 997
    while math.gcd(stride, n_samples) != 1:
        stride += 1
        
    mixed_samples = [None] * n_samples
    for i in range(n_samples):
        dest_idx = (i * stride) % n_samples
        mixed_samples[dest_idx] = final_samples[i]
        
    _stratified_polar_sampler_iterator = iter(mixed_samples)
    print(f"Fully Deterministic Spatial Sampling V3: Prepared {len(mixed_samples)} samples.")

def _fully_deterministic_spatial_sampling_v3(width, height):
    global _stratified_polar_sampler_iterator
    if _stratified_polar_sampler_iterator is None:
        print("Notice: Fully Deterministic Spatial V3 sampler not initialized. Auto-initializing (N=1000)...")
        _initialize_fully_deterministic_spatial_sampler_v3(1000, width, height)
    
    try:
        return next(_stratified_polar_sampler_iterator)
    except StopIteration:
        print("Error: Ran out of samples.")
        return _uniform_sampling(width, height)


# --- Drawing and Plotting ---

def trapezoid_transform(x, y, width, height, scale_factor):
    scale_y = 1 - (1 - scale_factor) * (x / width)
    y_center = height / 2
    y_offset = (y - y_center) * scale_y + y_center
    x_compressed = x * (1 - 0.17 * (x / width))
    return x_compressed, y_offset

def _color_styles(color):
    return ('black', 'black') if color == 'white' else (color, 'white')

def draw_grid(ax, width, height, scale_factor):
    for i in range(width + 1):
        x_vals, y_vals = np.ones(200) * i, np.linspace(0, height, 200)
        x_t, y_t = np.vectorize(trapezoid_transform)(x_vals, y_vals, width, height, scale_factor)
        ax.plot(x_t, y_t, 'k-', linewidth=2)
    for j in range(height + 1):
        x_vals, y_vals = np.linspace(0, width, 200), np.ones(200) * j
        x_t, y_t = np.vectorize(trapezoid_transform)(x_vals, y_vals, width, height, scale_factor)
        ax.plot(x_t, y_t, 'k-', linewidth=2)

def draw_box(ax, pos, width, scale_factor, height, color='red'):
    x_t, y_t = trapezoid_transform(pos[0], pos[1], width, height, scale_factor)
    edge_color, text_color = _color_styles(color)
    box_rect = patches.Rectangle((x_t + 0.1, y_t + 0.1), width=0.8, height=0.8, linewidth=2, edgecolor=edge_color, facecolor=color, alpha=0.9)
    ax.add_patch(box_rect)
    ax.text(x_t + 0.5, y_t + 0.5, 'Box', ha='center', va='center', fontsize=10, fontweight='bold', color=text_color)

def draw_ball(ax, pos, width, scale_factor, height, color='blue'):
    x_t, y_t = trapezoid_transform(pos[0], pos[1], width, height, scale_factor)
    edge_color, text_color = _color_styles(color)
    ball_circle = patches.Circle((x_t + 0.5, y_t + 0.5), radius=0.35, linewidth=2, edgecolor=edge_color, facecolor=color, alpha=0.9)
    ax.add_patch(ball_circle)
    ax.text(x_t + 0.5, y_t + 0.5, 'Ball', ha='center', va='center', fontsize=8, fontweight='bold', color=text_color)

def setup_axes(ax, height):
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

def show_sampling_distribution_hist(sampling_func, width, height, n_samples=500, hist_r_bins=10, hist_theta_bins=12):
    global FALLBACK_COUNT
    FALLBACK_COUNT = 0  # Reset counter before generation
    print(f"Generating {n_samples} samples using {sampling_func.__name__} to analyze distribution...")
    
    samples = [sampling_func(width, height) for _ in range(n_samples)]
    dists, thetas = [], []
    dx_list, dy_list = [], []
    ball_x_list, ball_y_list = [], []
    box_x_list, box_y_list = [], []
    for box_pos, ball_pos in samples:
        dx, dy = ball_pos[0] - box_pos[0], ball_pos[1] - box_pos[1]
        dists.append(math.sqrt(dx**2 + dy**2))
        thetas.append(math.atan2(dy, dx))
        dx_list.append(dx)
        dy_list.append(dy)
        ball_x_list.append(ball_pos[0])
        ball_y_list.append(ball_pos[1])
        box_x_list.append(box_pos[0])
        box_y_list.append(box_pos[1])

    # Report fallback statistics if applicable
    if sampling_func.__name__ == '_distance_based_sampling' or (sampling_func.__name__ == 'generate_positions' and RANDOM_FLAG == 1):
        print(f"Fallback to Uniform Sampling: {FALLBACK_COUNT}/{n_samples} ({FALLBACK_COUNT/n_samples*100:.2f}%)")

    fig, axes = plt.subplots(1, 6, figsize=(38, 6))
    func_name = sampling_func.__name__.replace('_', ' ').title()
    fig.suptitle(f'Distribution for {n_samples} Samples ({func_name})', fontsize=16)

    # Distance Histogram
    # Match the sampler's logic: use width-1, height-1 and add epsilon
    r_max = math.hypot(width - 1, height - 1) + 1e-5
    r_edges = np.linspace(0, r_max, hist_r_bins + 1)
    r_counts, _ = np.histogram(dists, bins=r_edges)
    axes[0].bar((r_edges[:-1] + r_edges[1:]) / 2.0, r_counts, width=np.diff(r_edges), align='center', edgecolor='black')
    axes[0].set_xlabel('Distance (r)'); axes[0].set_ylabel('Count'); axes[0].set_title('Relative Distance Histogram')
    axes[0].set_xticks(r_edges); axes[0].tick_params(axis='x', rotation=45)

    print('\nDistance histogram (bin_range -> count):')
    for i in range(len(r_counts)): print(f'  [{r_edges[i]:.2f}, {r_edges[i+1]:.2f}) -> {r_counts[i]}')
    
    if len(dists) > 0:
        print(f"  [Stats] Min Dist: {min(dists):.4f} (Excludes 0.0), Max Dist: {max(dists):.4f}")

    # Theta Histogram
    thetas_positive = [(t + 2 * math.pi) % (2 * math.pi) for t in thetas]
    # Match the sampler's logic for theta bins
    theta_edges = np.linspace(-1e-5, 2 * math.pi - 1e-5, hist_theta_bins + 1)
    th_counts, _ = np.histogram(thetas_positive, bins=theta_edges)
    axes[1].bar((theta_edges[:-1] + theta_edges[1:]) / 2.0, th_counts, width=np.diff(theta_edges), align='center', edgecolor='black')
    axes[1].set_xlabel('Theta (rad)'); axes[1].set_title('Relative Angle Histogram')
    axes[1].set_xticks(theta_edges); axes[1].tick_params(axis='x', rotation=45)

    # Relative Position Density (2D Histogram)
    # Create bins centered on integer coordinates for dx, dy
    bins_x = np.arange(-(width - 1) - 0.5, width + 0.5, 1)
    bins_y = np.arange(-(height - 1) - 0.5, height + 0.5, 1)
    
    h = axes[2].hist2d(dx_list, dy_list, bins=[bins_x, bins_y], cmap='Purples', vmin=0)
    axes[2].set_title(f'Relative Position Density (N={n_samples})')
    axes[2].set_xlabel('dx')
    axes[2].set_ylabel('dy')
    axes[2].axhline(0, color='black', linewidth=0.5)
    axes[2].axvline(0, color='black', linewidth=0.5)
    axes[2].grid(True, linestyle='--', alpha=0.3)
    axes[2].set_aspect('equal')
    fig.colorbar(h[3], ax=axes[2], label='Count')

    # Ball Position Density (2D Histogram)
    bins_ball_x = np.arange(-0.5, width + 0.5, 1)
    bins_ball_y = np.arange(-0.5, height + 0.5, 1)
    h_ball = axes[3].hist2d(ball_x_list, ball_y_list, bins=[bins_ball_x, bins_ball_y], cmap='Blues', vmin=0)
    axes[3].set_title(f'Ball Position Density (N={n_samples})')
    axes[3].set_xlabel('Ball X')
    axes[3].set_ylabel('Ball Y')
    axes[3].set_aspect('equal')
    axes[3].invert_yaxis()
    axes[3].grid(True, linestyle='--', alpha=0.3)
    fig.colorbar(h_ball[3], ax=axes[3], label='Count')
    print(f"Ball Position Density: Min={np.min(h_ball[0])}, Max={np.max(h_ball[0])}")

    # Box Position Density (2D Histogram)
    bins_box_x = np.arange(-0.5, width + 0.5, 1)
    bins_box_y = np.arange(-0.5, height + 0.5, 1)
    h_box = axes[4].hist2d(box_x_list, box_y_list, bins=[bins_box_x, bins_box_y], cmap='Reds', vmin=0)
    axes[4].set_title(f'Box Position Density (N={n_samples})')
    axes[4].set_xlabel('Box X')
    axes[4].set_ylabel('Box Y')
    axes[4].set_aspect('equal')
    axes[4].invert_yaxis()
    axes[4].grid(True, linestyle='--', alpha=0.3)
    fig.colorbar(h_box[3], ax=axes[4], label='Count')
    print(f"Box Position Density: Min={np.min(h_box[0])}, Max={np.max(h_box[0])}")

    # Total Position Density (Box + Ball)
    combined_x = ball_x_list + box_x_list
    combined_y = ball_y_list + box_y_list
    h_total = axes[5].hist2d(combined_x, combined_y, bins=[bins_box_x, bins_box_y], cmap='Greens', vmin=0)
    axes[5].set_title(f'Total Position Density (Box + Ball)')
    axes[5].set_xlabel('X')
    axes[5].set_ylabel('Y')
    axes[5].set_aspect('equal')
    axes[5].invert_yaxis()
    axes[5].grid(True, linestyle='--', alpha=0.3)
    fig.colorbar(h_total[3], ax=axes[5], label='Count')
    print(f"Total Position Density: Min={np.min(h_total[0])}, Max={np.max(h_total[0])}")

    print('\nTheta histogram (radians) (bin_range -> count):')
    for i in range(len(th_counts)): print(f'  [{theta_edges[i]:.2f}, {theta_edges[i+1]:.2f}) -> {th_counts[i]}')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def init_sampler(width, height, n_samples, r_bins=None, theta_bins=None):
    """
    Initializes the sampler if the current RANDOM_FLAG requires it (e.g. Stratified Polar).
    """
    if r_bins is None: r_bins = 10
    if theta_bins is None: theta_bins = 12

    if RANDOM_FLAG == 5:
        _initialize_stratified_polar_sampler(n_samples, width, height, num_r_bins=r_bins, num_theta_bins=theta_bins)
    elif RANDOM_FLAG == 6:
        _initialize_stratified_difficulty_sampler(n_samples, width, height, num_r_bins=r_bins, num_theta_bins=theta_bins)
    elif RANDOM_FLAG == 7:
        _initialize_stratified_hand_difficulty_sampler(n_samples, width, height, num_r_bins=r_bins, num_theta_bins=theta_bins)
    elif RANDOM_FLAG == 8:
        _initialize_simple_random_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 9:
        _initialize_numpy_random_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 10:
        _initialize_redundancy_aware_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 11:
        _initialize_sobol_ball_diversity_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 12:
        _initialize_balanced_spatial_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 13:
        _initialize_edge_biased_spatial_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 14:
        _initialize_edge_biased_spatial_sampler_v2(n_samples, width, height)
    elif RANDOM_FLAG == 15:
        _initialize_deterministic_spatial_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 16:
        _initialize_fully_deterministic_spatial_sampler(n_samples, width, height)
    elif RANDOM_FLAG == 17:
        _initialize_fully_deterministic_spatial_sampler_v3(n_samples, width, height)

def generate_positions(width, height):
    """
    Dispatches to the correct sampling function based on the global RANDOM_FLAG.
    """
    if RANDOM_FLAG == 0:
        return _stratified_sampling(width, height)
    elif RANDOM_FLAG == 1:
        return _distance_based_sampling(width, height)
    elif RANDOM_FLAG == 2:
        return _uniform_sampling(width, height)
    elif RANDOM_FLAG == 3:
        return _sobol_cartesian_sampling(width, height)
    elif RANDOM_FLAG == 4:
        return _sobol_relative_polar_sampling(width, height)
    elif RANDOM_FLAG == 5:
        # Note: _stratified_polar_sampling requires initialization
        return _stratified_polar_sampling(width, height)
    elif RANDOM_FLAG == 6:
        return _stratified_difficulty_sampling(width, height)
    elif RANDOM_FLAG == 7:
        return _stratified_hand_difficulty_sampling(width, height)
    elif RANDOM_FLAG == 8:
        return _simple_random_sampling(width, height)
    elif RANDOM_FLAG == 9:
        return _numpy_random_sampling(width, height)
    elif RANDOM_FLAG == 10:
        return _redundancy_aware_sampling(width, height)
    elif RANDOM_FLAG == 11:
        return _sobol_ball_diversity_sampling(width, height)
    elif RANDOM_FLAG == 12:
        return _balanced_spatial_sampling(width, height)
    elif RANDOM_FLAG == 13:
        return _edge_biased_spatial_sampling(width, height)
    elif RANDOM_FLAG == 14:
        return _edge_biased_spatial_sampling_v2(width, height)
    elif RANDOM_FLAG == 15:
        return _deterministic_spatial_sampling(width, height)
    elif RANDOM_FLAG == 16:
        return _fully_deterministic_spatial_sampling(width, height)
    elif RANDOM_FLAG == 17:
        return _fully_deterministic_spatial_sampling_v3(width, height)
    else:
        # Default to uniform sampling if the flag is not recognized
        return _uniform_sampling(width, height)


if __name__ == '__main__':
    # Test Configuration: Change this flag to test different strategies
    # 0:Stratified, 1:Distance-Based, 2:Uniform, 3:Sobol-Cartesian, 4:Sobol-Polar, 5:Stratified-Polar, 6:Stratified-Difficulty, 7:Stratified-Hand-Difficulty, 8:Simple-Random, 9:Numpy-Random, 10:Redundancy-Aware, 11:Sobol-Ball-Diversity, 12=Balanced-Spatial
    RANDOM_FLAG = 17 # 15-17

    grid_width, grid_height = 8, 6
    n_samples_for_hist = 164
    r_bins_for_hist = 10
    theta_bins_for_hist = 12

    # Initialize sampler if needed (e.g. for Stratified Polar) based on RANDOM_FLAG
    init_sampler(grid_width, grid_height, n_samples_for_hist, r_bins=r_bins_for_hist, theta_bins=theta_bins_for_hist)

    # Now, analyze the distribution using the currently selected sampling strategy
    show_sampling_distribution_hist(
        sampling_func=generate_positions,
        width=grid_width,
        height=grid_height,
        n_samples=n_samples_for_hist,
        hist_r_bins=r_bins_for_hist,
        hist_theta_bins=theta_bins_for_hist
    )
