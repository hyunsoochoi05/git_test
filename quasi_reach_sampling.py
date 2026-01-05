import random
import math
import numpy as np

try:
    from scipy.stats import qmc
    _HAS_SCIPY_QMC = True
except Exception:
    _HAS_SCIPY_QMC = False


def _van_der_corput(n_samples, base=2):
    seq = []
    for i in range(n_samples):
        n_th_number = 0
        denom = 1
        index = i
        while index:
            denom *= base
            index, remainder = divmod(index, base)
            n_th_number += remainder / denom
        seq.append(n_th_number)
    return seq


def _halton_sequence(dim, n_samples, skip=0):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    if dim > len(primes):
        raise ValueError('dim too large for simple Halton implementation')
    seq = np.empty((n_samples, dim), dtype=float)
    for d in range(dim):
        base = primes[d]
        seq[:, d] = _van_der_corput(n_samples + skip, base=base)[skip:skip + n_samples]
    return seq


def quasi_sequence(dim, n, method='sobol'):
    """Return an (n,dim) array in [0,1) of quasi-random samples.

    method: 'sobol' prefers SciPy's Sobol if available, falls back to 'halton'.
    """
    if method == 'sobol' and _HAS_SCIPY_QMC:
        engine = qmc.Sobol(d=dim, scramble=False)
        return engine.random(n)
    return _halton_sequence(dim, n)


def _conditional_polar_offset(r_min, r_max):
    r = math.sqrt(random.uniform(r_min * r_min, r_max * r_max))
    theta = random.uniform(0, 2 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)


def _is_in_fov(pt, cam_pos=(0, 0), cam_dir=(1, 0), fov_deg=120):
    vx, vy = pt[0] - cam_pos[0], pt[1] - cam_pos[1]
    if vx == 0 and vy == 0:
        return True
    pdot = vx * cam_dir[0] + vy * cam_dir[1]
    norm_v = math.hypot(vx, vy)
    norm_dir = math.hypot(cam_dir[0], cam_dir[1])
    cosang = pdot / (norm_v * norm_dir)
    cosang = max(-1.0, min(1.0, cosang))
    ang = math.degrees(math.acos(cosang))
    return ang <= (fov_deg / 2.0)


def _reachable_by_arm(pt, arm_base, min_reach, max_reach):
    d = math.hypot(pt[0] - arm_base[0], pt[1] - arm_base[1])
    return (d >= min_reach) and (d <= max_reach)


def sample_dist_and_dir_pairs(width,
                              height,
                              n_samples=100,
                              recipe=(0.4, 0.4, 0.2),
                              r_min=1.0,
                              r_max=None,
                              min_distance=1.0,
                              robot_params=None,
                              camera_params=None,
                              quasi_method='sobol'):
    """High-level sampler implementing the mixed recipe described in the design.

    Returns a list of tuples: (box_pos, ball_pos, tag, reach_type)
    """
    if r_max is None:
        r_max = math.hypot(width - 1, height - 1)

    if robot_params is None:
        robot_params = {
            'left': {'base': (0, height / 2), 'min_reach': 0.0, 'max_reach': r_max * 0.7},
            'right': {'base': (width - 1, height / 2), 'min_reach': 0.0, 'max_reach': r_max * 0.7},
        }

    if camera_params is None:
        camera_params = {'pos': (width / 2, -10), 'dir': (0, 1), 'fov_deg': 180}

    n_global = int(round(n_samples * recipe[0]))
    n_task = int(round(n_samples * recipe[1]))
    n_hard = n_samples - n_global - n_task

    samples = []

    pts = quasi_sequence(2, n_global * 2, method=quasi_method)
    for i in range(0, n_global * 2, 2):
        bx = min(int(pts[i, 0] * width), width - 1)
        by = min(int(pts[i, 1] * height), height - 1)
        rx = min(int(pts[i + 1, 0] * width), width - 1)
        ry = min(int(pts[i + 1, 1] * height), height - 1)
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
        samples.append(((bx, by), (rx, ry), 'global'))

    box_pts = quasi_sequence(2, n_task, method=quasi_method)
    for i in range(n_task):
        bx = min(int(box_pts[i, 0] * width), width - 1)
        by = min(int(box_pts[i, 1] * height), height - 1)
        dx, dy = _conditional_polar_offset(r_min, r_max)
        rx = max(0, min(width - 1, int(round(bx + dx))))
        ry = max(0, min(height - 1, int(round(by + dy))))
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
        samples.append(((bx, by), (rx, ry), 'task'))

    edge_band = max(1, int(min(width, height) * 0.12))
    for _ in range(n_hard):
        if random.random() < 0.5:
            bx = random.choice(list(range(0, edge_band)) + list(range(width - edge_band, width)))
            by = random.randint(0, height - 1)
        else:
            bx = random.randint(0, width - 1)
            by = random.choice(list(range(0, edge_band)) + list(range(height - edge_band, height)))
        dx, dy = _conditional_polar_offset(max(r_max * 0.75, r_min), r_max)
        rx = max(0, min(width - 1, int(round(bx + dx))))
        ry = max(0, min(height - 1, int(round(by + dy))))
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
        samples.append(((bx, by), (rx, ry), 'hard'))

    assigned = []
    attempts = 0
    max_attempts = len(samples) * 5
    for pair in samples:
        b, r, tag = pair
        left_ok = _reachable_by_arm(r, robot_params['left']['base'], robot_params['left']['min_reach'], robot_params['left']['max_reach'])
        right_ok = _reachable_by_arm(r, robot_params['right']['base'], robot_params['right']['min_reach'], robot_params['right']['max_reach'])
        reach_type = 'none'
        if left_ok and right_ok:
            reach_type = 'both'
        elif left_ok:
            reach_type = 'left'
        elif right_ok:
            reach_type = 'right'

        if not _is_in_fov(b, cam_pos=camera_params['pos'], cam_dir=camera_params['dir'], fov_deg=camera_params['fov_deg']):
            continue
        if not _is_in_fov(r, cam_pos=camera_params['pos'], cam_dir=camera_params['dir'], fov_deg=camera_params['fov_deg']):
            continue

        assigned.append((b, r, tag, reach_type))
        attempts += 1
        if attempts >= max_attempts:
            break

    if len(assigned) < n_samples:
        for pair in samples:
            b, r, tag = pair
            left_ok = _reachable_by_arm(r, robot_params['left']['base'], robot_params['left']['min_reach'], robot_params['left']['max_reach'])
            right_ok = _reachable_by_arm(r, robot_params['right']['base'], robot_params['right']['min_reach'], robot_params['right']['max_reach'])
            reach_type = 'both' if (left_ok and right_ok) else ('left' if left_ok else ('right' if right_ok else 'none'))
            assigned.append((b, r, tag, reach_type))
            if len(assigned) >= n_samples:
                break

    assigned = assigned[:n_samples]
    return assigned


def _conditional_polar_offset_distance_uniform(r_min, r_max):
    """Sample polar offset where radius r is uniform in [r_min, r_max].

    This produces equal counts per distance (not equal per area).
    """
    r = random.uniform(r_min, r_max)
    theta = random.uniform(0, 2 * math.pi)
    return r * math.cos(theta), r * math.sin(theta)


def sample_dist_and_dir_pairs_distance_uniform(width,
                                               height,
                                               n_samples=100,
                                               recipe=(0.4, 0.4, 0.2),
                                               r_min=1.0,
                                               r_max=None,
                                               min_distance=1.0,
                                               robot_params=None,
                                               camera_params=None,
                                               quasi_method='sobol'):
    """Variant of sample_dist_and_dir_pairs that uses distance-uniform radius sampling.

    Keeps all other logic identical but replaces area-uniform radius sampling
    with uniform-in-r sampling (equal counts per radial distance).
    """
    if r_max is None:
        r_max = math.hypot(width - 1, height - 1)

    if robot_params is None:
        robot_params = {
            'left': {'base': (0, height / 2), 'min_reach': 0.0, 'max_reach': r_max * 0.7},
            'right': {'base': (width - 1, height / 2), 'min_reach': 0.0, 'max_reach': r_max * 0.7},
        }

    if camera_params is None:
        camera_params = {'pos': (width / 2, -10), 'dir': (0, 1), 'fov_deg': 180}

    n_global = int(round(n_samples * recipe[0]))
    n_task = int(round(n_samples * recipe[1]))
    n_hard = n_samples - n_global - n_task

    samples = []

    pts = quasi_sequence(2, n_global * 2, method=quasi_method)
    for i in range(0, n_global * 2, 2):
        bx = min(int(pts[i, 0] * width), width - 1)
        by = min(int(pts[i, 1] * height), height - 1)
        rx = min(int(pts[i + 1, 0] * width), width - 1)
        ry = min(int(pts[i + 1, 1] * height), height - 1)
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset_distance_uniform(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
        samples.append(((bx, by), (rx, ry), 'global'))

    box_pts = quasi_sequence(2, n_task, method=quasi_method)
    for i in range(n_task):
        bx = min(int(box_pts[i, 0] * width), width - 1)
        by = min(int(box_pts[i, 1] * height), height - 1)
        dx, dy = _conditional_polar_offset_distance_uniform(r_min, r_max)
        rx = max(0, min(width - 1, int(round(bx + dx))))
        ry = max(0, min(height - 1, int(round(by + dy))))
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset_distance_uniform(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
        samples.append(((bx, by), (rx, ry), 'task'))

    edge_band = max(1, int(min(width, height) * 0.12))
    for _ in range(n_hard):
        if random.random() < 0.5:
            bx = random.choice(list(range(0, edge_band)) + list(range(width - edge_band, width)))
            by = random.randint(0, height - 1)
        else:
            bx = random.randint(0, width - 1)
            by = random.choice(list(range(0, edge_band)) + list(range(height - edge_band, height)))
        dx, dy = _conditional_polar_offset_distance_uniform(max(r_max * 0.75, r_min), r_max)
        rx = max(0, min(width - 1, int(round(bx + dx))))
        ry = max(0, min(height - 1, int(round(by + dy))))
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset_distance_uniform(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
        samples.append(((bx, by), (rx, ry), 'hard'))

    assigned = []
    attempts = 0
    max_attempts = len(samples) * 5
    for pair in samples:
        b, r, tag = pair
        left_ok = _reachable_by_arm(r, robot_params['left']['base'], robot_params['left']['min_reach'], robot_params['left']['max_reach'])
        right_ok = _reachable_by_arm(r, robot_params['right']['base'], robot_params['right']['min_reach'], robot_params['right']['max_reach'])
        reach_type = 'none'
        if left_ok and right_ok:
            reach_type = 'both'
        elif left_ok:
            reach_type = 'left'
        elif right_ok:
            reach_type = 'right'

        if not _is_in_fov(b, cam_pos=camera_params['pos'], cam_dir=camera_params['dir'], fov_deg=camera_params['fov_deg']):
            continue
        if not _is_in_fov(r, cam_pos=camera_params['pos'], cam_dir=camera_params['dir'], fov_deg=camera_params['fov_deg']):
            continue

        assigned.append((b, r, tag, reach_type))
        attempts += 1
        if attempts >= max_attempts:
            break

    if len(assigned) < n_samples:
        for pair in samples:
            b, r, tag = pair
            left_ok = _reachable_by_arm(r, robot_params['left']['base'], robot_params['left']['min_reach'], robot_params['left']['max_reach'])
            right_ok = _reachable_by_arm(r, robot_params['right']['base'], robot_params['right']['min_reach'], robot_params['right']['max_reach'])
            reach_type = 'both' if (left_ok and right_ok) else ('left' if left_ok else ('right' if right_ok else 'none'))
            assigned.append((b, r, tag, reach_type))
            if len(assigned) >= n_samples:
                break

    assigned = assigned[:n_samples]
    return assigned


def sample_dist_and_dir_pairs_annulus(width,
                                     height,
                                     n_samples=100,
                                     recipe=(0.4, 0.4, 0.2),
                                     r_min=1.0,
                                     r_max=None,
                                     n_bins=6,
                                     min_distance=1.0,
                                     robot_params=None,
                                     camera_params=None,
                                     quasi_method='sobol'):
    """Stratified-annulus sampler.

    - Splits radial range [r_min, r_max] into `n_bins` annuli.
    - Ensures (approximately) uniform counts per radial bin by round-robin
      allocation of task samples across bins, and uses quasi-random
      1D samples for theta inside each bin to guarantee angular coverage.
    - Keeps global/hard groups similar to other samplers; hard group
      preferentially samples outer bins.

    Returns list of (box_pos, ball_pos, tag, reach_type).
    """
    if r_max is None:
        r_max = math.hypot(width - 1, height - 1)

    if robot_params is None:
        robot_params = {
            'left': {'base': (0, height / 2), 'min_reach': 0.0, 'max_reach': r_max * 0.7},
            'right': {'base': (width - 1, height / 2), 'min_reach': 0.0, 'max_reach': r_max * 0.7},
        }

    if camera_params is None:
        camera_params = {'pos': (width / 2, -10), 'dir': (0, 1), 'fov_deg': 180}

    n_global = int(round(n_samples * recipe[0]))
    n_task = int(round(n_samples * recipe[1]))
    n_hard = n_samples - n_global - n_task

    samples = []

    # Global: pair quasi samples as before
    pts = quasi_sequence(2, n_global * 2, method=quasi_method)
    for i in range(0, n_global * 2, 2):
        bx = min(int(pts[i, 0] * width), width - 1)
        by = min(int(pts[i, 1] * height), height - 1)
        rx = min(int(pts[i + 1, 0] * width), width - 1)
        ry = min(int(pts[i + 1, 1] * height), height - 1)
        r_cont = math.hypot(rx - bx, ry - by)
        theta_cont = math.atan2(ry - by, rx - bx)
        if r_cont < min_distance:
            dx, dy = _conditional_polar_offset(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
            r_cont = math.hypot(rx - bx, ry - by)
            theta_cont = math.atan2(ry - by, rx - bx)
        samples.append(((bx, by), (rx, ry), 'global', r_cont, theta_cont))

    # Task-centric: stratify radius into bins and use quasi theta per bin
    box_pts = quasi_sequence(2, n_task, method=quasi_method)
    bins = np.linspace(r_min, r_max, n_bins + 1)
    # allocate boxes to bins round-robin to produce approximately equal counts
    bin_lists = {k: [] for k in range(n_bins)}
    for i in range(n_task):
        bin_idx = i % n_bins
        bin_lists[bin_idx].append(i)

    # For each bin, draw theta quasi-sequence to ensure angular coverage
    for bin_idx, indices in bin_lists.items():
        if len(indices) == 0:
            continue
        theta_seq = quasi_sequence(1, len(indices), method=quasi_method).flatten()
        r_lo, r_hi = bins[bin_idx], bins[bin_idx + 1]
        for j, i_box in enumerate(indices):
            bx = min(int(box_pts[i_box, 0] * width), width - 1)
            by = min(int(box_pts[i_box, 1] * height), height - 1)
            # radius uniform inside bin
            r = random.uniform(r_lo, r_hi)
            theta = theta_seq[j] * 2.0 * math.pi
            dx, dy = r * math.cos(theta), r * math.sin(theta)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
            r_cont, theta_cont = r, theta
            if math.hypot(bx - rx, by - ry) < min_distance:
                # nudge outward within the same bin
                r = max(r_lo, min(r_hi, r + min_distance))
                theta = (theta + 0.37) % (2.0 * math.pi)
                dx, dy = r * math.cos(theta), r * math.sin(theta)
                rx = max(0, min(width - 1, int(round(bx + dx))))
                ry = max(0, min(height - 1, int(round(by + dy))))
                r_cont, theta_cont = r, theta
            samples.append(((bx, by), (rx, ry), 'task', r_cont, theta_cont))

    # Hard-edge: pick boxes near boundary and sample ball from outer bins
    edge_band = max(1, int(min(width, height) * 0.12))
    outer_bins = list(range(max(0, n_bins - 2), n_bins))
    for idx in range(n_hard):
        if random.random() < 0.5:
            bx = random.choice(list(range(0, edge_band)) + list(range(width - edge_band, width)))
            by = random.randint(0, height - 1)
        else:
            bx = random.randint(0, width - 1)
            by = random.choice(list(range(0, edge_band)) + list(range(height - edge_band, height)))
        bin_idx = random.choice(outer_bins)
        r_lo, r_hi = bins[bin_idx], bins[bin_idx + 1]
        r = random.uniform(r_lo, r_hi)
        theta = random.uniform(0, 2.0 * math.pi)
        dx, dy = r * math.cos(theta), r * math.sin(theta)
        rx = max(0, min(width - 1, int(round(bx + dx))))
        ry = max(0, min(height - 1, int(round(by + dy))))
        r_cont, theta_cont = r, theta
        if math.hypot(bx - rx, by - ry) < min_distance:
            dx, dy = _conditional_polar_offset(min_distance, min_distance + 2)
            rx = max(0, min(width - 1, int(round(bx + dx))))
            ry = max(0, min(height - 1, int(round(by + dy))))
            r_cont = math.hypot(rx - bx, ry - by)
            theta_cont = math.atan2(ry - by, rx - bx)
        samples.append(((bx, by), (rx, ry), 'hard', r_cont, theta_cont))

    # Assign reachability + FOV filters as in other samplers
    assigned = []
    attempts = 0
    max_attempts = len(samples) * 5
    for pair in samples:
        b, r, tag, r_cont, theta_cont = pair
        left_ok = _reachable_by_arm(r, robot_params['left']['base'], robot_params['left']['min_reach'], robot_params['left']['max_reach'])
        right_ok = _reachable_by_arm(r, robot_params['right']['base'], robot_params['right']['min_reach'], robot_params['right']['max_reach'])
        reach_type = 'none'
        if left_ok and right_ok:
            reach_type = 'both'
        elif left_ok:
            reach_type = 'left'
        elif right_ok:
            reach_type = 'right'

        if not _is_in_fov(b, cam_pos=camera_params['pos'], cam_dir=camera_params['dir'], fov_deg=camera_params['fov_deg']):
            continue
        if not _is_in_fov(r, cam_pos=camera_params['pos'], cam_dir=camera_params['dir'], fov_deg=camera_params['fov_deg']):
            continue

        assigned.append((b, r, tag, reach_type, r_cont, theta_cont))
        attempts += 1
        if attempts >= max_attempts:
            break

    if len(assigned) < n_samples:
        for pair in samples:
            b, r, tag, r_cont, theta_cont = pair
            left_ok = _reachable_by_arm(r, robot_params['left']['base'], robot_params['left']['min_reach'], robot_params['left']['max_reach'])
            right_ok = _reachable_by_arm(r, robot_params['right']['base'], robot_params['right']['min_reach'], robot_params['right']['max_reach'])
            reach_type = 'both' if (left_ok and right_ok) else ('left' if left_ok else ('right' if right_ok else 'none'))
            assigned.append((b, r, tag, reach_type, r_cont, theta_cont))
            if len(assigned) >= n_samples:
                break

    assigned = assigned[:n_samples]
    return assigned


def sample_dist_and_dir_pairs_uniform_coverage(width,
                                               height,
                                               n_samples=100,
                                               n_dist_bins=6,
                                               n_theta_bins=12):
    """Simple uniform coverage sampler for small grids.

    Strategy: Generate all valid pairs on the grid, categorize by distance and theta bins,
    then sample uniformly from each bin to ensure balanced distribution.
    Best for small grids (e.g., 8x6).
    """
    pairs_by_bin = {}
    r_max = math.hypot(width - 1, height - 1)
    
    # Collect all pairs and bin them
    for bx in range(width):
        for by in range(height):
            for rx in range(width):
                for ry in range(height):
                    if (bx, by) == (rx, ry):
                        continue
                    dx = rx - bx
                    dy = ry - by
                    dist = math.hypot(dx, dy)
                    theta = math.atan2(dy, dx)
                    if theta < 0:
                        theta += 2 * math.pi
                    
                    dist_bin = int(min(n_dist_bins - 1, dist / r_max * n_dist_bins))
                    theta_bin = int(min(n_theta_bins - 1, theta / (2 * math.pi) * n_theta_bins))
                    
                    key = (dist_bin, theta_bin)
                    if key not in pairs_by_bin:
                        pairs_by_bin[key] = []
                    pairs_by_bin[key].append(((bx, by), (rx, ry), dist, theta))
    
    # Sample uniformly from each bin
    samples = []
    target_per_bin = n_samples / len(pairs_by_bin)
    
    for key, pairs in sorted(pairs_by_bin.items()):
        n_to_sample = max(1, int(round(target_per_bin)))
        if len(pairs) > 0:
            selected = random.sample(pairs, min(n_to_sample, len(pairs)))
            for b, r, d, th in selected:
                samples.append((b, r, d, th, 'uniform'))
    
    # If we have fewer samples than requested, oversample high-population bins
    if len(samples) < n_samples:
        shortage = n_samples - len(samples)
        for key, pairs in sorted(pairs_by_bin.items(), key=lambda x: len(x[1]), reverse=True):
            if shortage <= 0:
                break
            n_extra = min(shortage, len(pairs))
            extra = random.sample(pairs, n_extra)
            for b, r, d, th in extra:
                samples.append((b, r, d, th, 'uniform'))
                shortage -= 1
    
    # Trim to exact size
    samples = samples[:n_samples]
    return samples


if __name__ == '__main__':
    demo = sample_dist_and_dir_pairs(40, 20, n_samples=100)
    counts = {'global': 0, 'task': 0, 'hard': 0}
    reach_counts = {'left': 0, 'right': 0, 'both': 0, 'none': 0}
    for b, r, tag, reach in demo:
        counts[tag] = counts.get(tag, 0) + 1
        reach_counts[reach] = reach_counts.get(reach, 0) + 1
    print('Sample groups:', counts)
    print('Reachability:', reach_counts)
