import os
import numpy as np
import math
from tqdm import tqdm

# Radial concentration of the crater's rings. < 1 puts smaller facets near the
# rim (better shadowing resolution at grazing angles), > 1 near the centre.
# 0.75 is the production value; the env override exists so a mesh-resolution
# study can vary it without touching the science default.
RING_POWER = float(os.environ.get('CRATER_RING_POWER', '0.75'))


def _calculate_facet_properties(p1, p2, p3):
    """Compute facet normal (outward) and area for triangle (p1, p2, p3)."""
    vertices = np.array([p1, p2, p3])
    v1 = p2 - p1
    v2 = p3 - p1
    if np.linalg.norm(v1) < 1e-9 or np.linalg.norm(v2) < 1e-9:
        return None
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    area = 0.5 * norm
    normal = n / norm
    # Ensure outward normal for concave cap
    center_vec = np.mean(vertices, axis=0)
    if np.dot(normal, center_vec) < 0:
        normal = -normal
        vertices = np.array([p1, p3, p2])
    return {'vertices': vertices, 'normal': normal, 'area': area}


def _compute_equilateral_rings(n_subfacets, phi, sphere_r, min_pts=6, max_rings=50, power=1.0):
    """
    Pick ring count and points per ring to approximate equilateral triangles.
    Returns best (ring_counts, triangle_count).
    If power < 1, rings are concentrated near the rim (higher theta).
    If power > 1, rings are concentrated near the center (lower theta).
    """
    best = None
    best_diff = float('inf')
    for R in range(2, max_rings + 1):
        ring_counts = []
        # Calculate theta for each ring boundary
        thetas = [phi * ((i / R) ** power) for i in range(R + 1)]
        
        for i in range(1, R + 1):
            theta_prev = thetas[i-1]
            theta_curr = thetas[i]
            theta_mid = (theta_prev + theta_curr) / 2
            
            # Segment length along meridian
            seg_len = sphere_r * (theta_curr - theta_prev)
            
            # Circumference at mid-radius of the ring
            circ = 2 * math.pi * sphere_r * math.sin(theta_mid)
            
            # Points needed for approx equilateral triangles in this ring
            Ni = max(min_pts, int(round(circ / seg_len)))
            ring_counts.append(Ni)
            
        N1 = ring_counts[0]
        # Triangles in first cap + triangles in subsequent strips
        # First cap: N1 triangles
        # Strips: (N_prev + N_curr) triangles per strip
        tri_count = N1
        for r in range(len(ring_counts) - 1):
             tri_count += ring_counts[r] + ring_counts[r+1]
             
        diff = abs(tri_count - n_subfacets)
        if diff < best_diff:
            best = (ring_counts, tri_count)
            best_diff = diff
            if diff == 0:
                break
    return best


def generate_canonical_spherical_cap(n_subfacets, profile_angle_deg, return_rings=False):
    """
    Generate a concave spherical cap mesh. Returns list of facets dicts.
    If return_rings=True, also return the ring_counts used.
    Ensures opening area exactly 1 by a final scale correction.
    """
    # Flat disc case
    if profile_angle_deg <= 1e-3:
        radius = 1.0 / math.sqrt(math.pi)
        angles = np.linspace(0, 2 * math.pi, n_subfacets + 1)[:-1]
        pts = [np.array([radius * math.cos(a), radius * math.sin(a), 0.0]) for a in angles]
        facets = []
        center = np.zeros(3)
        for i, p in enumerate(pts):
            p2 = pts[(i + 1) % len(pts)]
            f = _calculate_facet_properties(center, p, p2)
            if f:
                f['normal'] = np.array([0.0, 0.0, 1.0])
                facets.append(f)
        return (facets, []) if return_rings else facets

    # Clamp angle
    if profile_angle_deg >= 180:
        profile_angle_deg = 179.9
    phi = math.radians(profile_angle_deg)
    # Sphere radius from theoretical base disc area
    sphere_r = 1.0 / (math.sqrt(math.pi) * math.sin(phi))
    z_base = sphere_r * math.cos(phi)

    # Use power < 1 to concentrate rings near the rim
    # This helps with shadowing resolution at grazing angles
    # Changed from 0.5 to 0.75 to keep high density near rim but increase resolution at the center
    power = RING_POWER

    # Determine rings
    ring_counts, actual_n = _compute_equilateral_rings(n_subfacets, phi, sphere_r, power=power)
    R = len(ring_counts)
    print(f"Using {R} rings (power={power}): points per ring = {ring_counts}")
    if actual_n != n_subfacets:
        print(f"Warning: target {n_subfacets}, got {actual_n} facets.")

    # Build point list
    points = []
    pts_per = []
    # Apex at concave tip
    apex_z = z_base - sphere_r
    points.append(np.array([0.0, 0.0, apex_z]))
    pts_per.append(1)
    
    # Calculate theta for each ring boundary
    thetas = [phi * ((i / R) ** power) for i in range(R + 1)]
    
    for idx, N in enumerate(ring_counts, start=1):
        theta = thetas[idx]
        for i in range(N):
            ang = 2 * math.pi * i / N
            x = sphere_r * math.sin(theta) * math.cos(ang)
            y = sphere_r * math.sin(theta) * math.sin(ang)
            z = sphere_r * math.cos(theta)
            z = z_base - z
            points.append(np.array([x, y, z]))
        pts_per.append(N)

    # Triangulate
    facets = []
    # Apex fan
    for i in range(pts_per[1]):
        f = _calculate_facet_properties(points[0], points[1 + i], points[1 + (i + 1) % pts_per[1]])
        if f:
            f['normal'] = -f['normal']
            facets.append(f)
    # Stitch rings
    offset = 1
    for r in tqdm(range(1, R), desc="Triangulating Rings"):
        Np, Nc = pts_per[r], pts_per[r + 1]
        prev_idx = list(range(offset, offset + Np))
        curr_idx = list(range(offset + Np, offset + Np + Nc))
        a = b = 0
        # merge-stitch
        while a < Np - 1 and b < Nc - 1:
            p_prev, p_prev_n = points[prev_idx[a]], points[prev_idx[a+1]]
            p_curr, p_curr_n = points[curr_idx[b]], points[curr_idx[b+1]]
            if (a+1)/Np < (b+1)/Nc:
                verts = (p_prev, p_curr, p_prev_n)
                a += 1
            else:
                verts = (p_prev, p_curr, p_curr_n)
                b += 1
            f = _calculate_facet_properties(*verts)
            if f:
                f['normal'] = -f['normal']
                facets.append(f)
        # finish remainders
        while a < Np - 1:
            verts = (points[prev_idx[a]], points[curr_idx[-1]], points[prev_idx[a+1]])
            f = _calculate_facet_properties(*verts)
            if f:
                f['normal'] = -f['normal']
                facets.append(f)
            a += 1
        while b < Nc - 1:
            verts = (points[prev_idx[-1]], points[curr_idx[b]], points[curr_idx[b+1]])
            f = _calculate_facet_properties(*verts)
            if f:
                f['normal'] = -f['normal']
                facets.append(f)
            b += 1
        # close seam
        for verts in [
            (points[prev_idx[-1]], points[curr_idx[-1]], points[curr_idx[0]]),
            (points[prev_idx[-1]], points[curr_idx[0]], points[prev_idx[0]])
        ]:
            f = _calculate_facet_properties(*verts)
            if f:
                f['normal'] = -f['normal']
                facets.append(f)
        offset += Np

        # Normalize opening area to exactly 1
    # Identify last-ring points
    last_start = 1 + sum(ring_counts[:-1])
    N_last = ring_counts[-1]
    last_pts = [points[last_start + i] for i in range(N_last)]
    center = np.zeros(3)
    # compute current opening area
    opening_area = 0.0
    for i in range(N_last):
        f = _calculate_facet_properties(center, last_pts[i], last_pts[(i+1)%N_last])
        if f:
            opening_area += f['area']
    # scale factor to normalize
    scale = math.sqrt(1.0 / opening_area)
    # apply scale
    for f in facets:
        f['vertices'] *= scale
        f['area'] *= scale * scale
    # recompute opening area after scaling for verification
    opening_area_after = 0.0
    for i in range(N_last):
        p1 = last_pts[i] * scale
        p2 = last_pts[(i+1)%N_last] * scale
        f = _calculate_facet_properties(center, p1, p2)
        if f:
            opening_area_after += f['area']
    print(f"Scaled geometry by factor {scale:.6f} to normalize opening area ({opening_area:.6f} → {opening_area_after:.6f}).")

    if return_rings:
        return facets, ring_counts
    return facets


def visualize_spherical_cap(n_subfacets, profile_angle_deg):
    try:
        import pyvista as pv
    except ImportError:
        raise ImportError("pyvista is required: pip install pyvista tqdm")

    mesh, ring_counts = generate_canonical_spherical_cap(n_subfacets, profile_angle_deg, return_rings=True)
    print(f"Final facet count: {len(mesh)} (requested: {n_subfacets})")
    # compute and print opening area and RMS slope (optional, already normalized)
    normals = np.array([f['normal'] for f in mesh])
    areas = np.array([f['area'] for f in mesh])
    slope_deg = np.degrees(np.arccos(np.abs(normals[:,2])))
    rms_slope = math.sqrt(np.sum(areas * slope_deg**2)/np.sum(areas))
    print(f"RMS slope angle (area-weighted): {rms_slope:.2f}°")

    # build and display mesh
    verts, faces = [], []
    idx = 0
    for f in mesh:
        verts.extend(f['vertices'])
        faces.extend([3, idx, idx+1, idx+2])
        idx += 3
    pts = np.array(verts)
    faces = np.array(faces)
    plotter = pv.Plotter(window_size=[800,800])
    plotter.add_mesh(pv.PolyData(pts, faces=faces), show_edges=True, color='lightblue')
    plotter.add_axes(); plotter.add_bounding_box()
    print(f"Visualizing cap at {profile_angle_deg}° with {len(mesh)} facets.")
    plotter.show()

if __name__ == "__main__":
    N_facets_target = 100
    profile_angle_deg = 90
    
    visualize_spherical_cap(N_facets_target, profile_angle_deg)
