import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import numpy as np
import yaml
rel_right_fingers_closed = np.array([[ 0.     ,  0.03545,  0.06046,  0.09203,  0.11413,  0.03693,  0.096  ,  0.12819,  0.13895,  0.14612,  0.03476,  0.09565,  0.13208,  0.14383,  0.15235,  0.03478,  0.08869,  0.12502,  0.1377 ,
         0.14418,  0.03407,  0.0779 ,  0.10785,  0.12232,  0.13406],
       [ 0.     ,  0.02889,  0.04728,  0.04922,  0.04988,  0.01916,  0.02355,  0.02466,  0.02432,  0.02251,  0.00295,  0.00173,  0.00285,  0.00364,  0.00425, -0.01499, -0.01747, -0.01884, -0.01627,
        -0.01173, -0.023  , -0.03505, -0.04186, -0.04151, -0.0363 ],
       [ 0.     , -0.01808, -0.02775, -0.03964, -0.05048, -0.0105 , -0.00732, -0.02733, -0.04912, -0.07025, -0.00815, -0.00254, -0.02522, -0.05012, -0.07361, -0.00602, -0.00653, -0.02063, -0.04385,
        -0.06691, -0.00942, -0.01369, -0.01345, -0.02771, -0.04552],
       [ 1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,
         1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ]])

# hand open
rel_right_fingers_open = np.array([[ 0.     ,  0.03229,  0.05329,  0.07798,  0.08908,  0.03534,  0.096  ,  0.13379,  0.15778,  0.17995,  0.03318,  0.09565,  0.13683,  0.1627 ,  0.1867 ,  0.03478,  0.08869,  0.12597,  0.15095,
         0.17418,  0.03407,  0.0779 ,  0.10587,  0.12355,  0.14378],
       [ 0.     ,  0.03214,  0.0566 ,  0.07964,  0.10159,  0.02078,  0.02355,  0.02356,  0.02321,  0.02139,  0.00457,  0.00173, -0.00523, -0.00977, -0.01464, -0.01499, -0.01747, -0.02678, -0.03281,
        -0.03724, -0.023  , -0.03505, -0.04746, -0.05635, -0.0644 ],
       [ 0.     , -0.01168, -0.01587, -0.01712, -0.01608, -0.0073 , -0.00732, -0.01044, -0.01433, -0.01689, -0.00655, -0.00254, -0.01247, -0.02077, -0.02578, -0.00602, -0.00653, -0.01319, -0.01996,
        -0.02588, -0.00942, -0.01369, -0.01637, -0.02095, -0.02382],
       [ 1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ,
         1.     ,  1.     ,  1.     ,  1.     ,  1.     ,  1.     ]])

tip_indices = [4, 9, 14, 19, 24]


assets_path = Path('/home/dpsh/kteleop/src/lerobot/teleoperators/vuer_vr/assets')
RetargetingConfig.set_default_urdf_dir(str(assets_path))
with (assets_path / 'inspire_hand/inspire_hand.yml').open('r') as f:
    cfg = yaml.safe_load(f)
    # Try a range of scaling factors
# for scale in [1,2,3,4,5,6,7]:
#     print(f"Testing scale {scale}")
#     cfg_copy = cfg['right'].copy()
#     cfg_copy['scaling_factor'] = scale
#     test_config = RetargetingConfig.from_dict(cfg_copy)
#     test_retargeting = test_config.build()
#     result = test_retargeting.retarget(rel_right_fingers_open[:3,tip_indices].T)
#     print(f"Result: {result}")
right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
right_retargeting = right_retargeting_config.build()

import numpy as np
from typing import Tuple, List

def test_all_coordinate_transforms(retargeting, target_vecs_original, title="Transform Testing"):
    """
    Test all possible coordinate transformations systematically
    Uses the fact that there are 8 directions for +X, 7 remaining for +Y, and +Z is determined by cross product
    
    Args:
        retargeting: The retargeting object
        target_vecs_original: Original target vectors (N, 3)
        title: Test title
    """
    
    def compute_error(retargeting_obj, target_vecs):
        """Helper to compute retargeting error"""
        try:
            result = retargeting_obj.retarget(target_vecs)
            robot = retargeting_obj.optimizer.robot
            optimizer = retargeting_obj.optimizer
            
            full_qpos = np.zeros(robot.dof)
            full_qpos[optimizer.idx_pin2target] = result[optimizer.idx_pin2target]
            if optimizer.adaptor is not None:
                full_qpos = optimizer.adaptor.forward_qpos(full_qpos)
            
            robot.compute_forward_kinematics(full_qpos)
            target_link_poses = [robot.get_link_pose(idx) for idx in optimizer.computed_link_indices]
            actual_positions = np.array([pose[:3, 3] for pose in target_link_poses])
            
            origin_positions = actual_positions[optimizer.origin_link_indices]
            task_positions = actual_positions[optimizer.task_link_indices]
            actual_vectors = task_positions - origin_positions
            
            vector_errors = np.linalg.norm(actual_vectors - target_vecs, axis=1)
            return vector_errors.mean(), vector_errors.max(), result[optimizer.idx_pin2target]
        except Exception as e:
            return float('inf'), float('inf'), None
    
    print(f"\n{title}")
    print("=" * 80)
    
    # Define all 8 possible directions for coordinate axes
    # Each direction is a unit vector along +/-X, +/-Y, +/-Z
    directions = np.array([
        [ 1,  0,  0],  # +X
        [-1,  0,  0],  # -X
        [ 0,  1,  0],  # +Y
        [ 0, -1,  0],  # -Y
        [ 0,  0,  1],  # +Z
        [ 0,  0, -1],  # -Z
    ])
    
    direction_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    results = []
    transform_count = 0
    
    print("Testing all 48 possible right-handed coordinate system transforms...")
    print("(8 choices for new +X axis × 6 remaining choices for new +Y axis)")
    print("-" * 80)
    
    # Loop through all possible +X directions (8 choices including diagonals)
    all_x_directions = np.array([
        [ 1,  0,  0],  # +X
        [-1,  0,  0],  # -X  
        [ 0,  1,  0],  # +Y
        [ 0, -1,  0],  # -Y
        [ 0,  0,  1],  # +Z
        [ 0,  0, -1],  # -Z
    ])
    
    x_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    for i, new_x in enumerate(all_x_directions):
        # Loop through all possible +Y directions that are orthogonal to the chosen +X
        for j, candidate_y in enumerate(all_x_directions):
            # Skip if this would be the same as +X or parallel/antiparallel to +X
            if np.abs(np.dot(new_x, candidate_y)) > 1e-10:
                continue
                
            new_y = candidate_y
            # Compute +Z via cross product to ensure right-handed system
            new_z = np.cross(new_x, new_y)
            
            # Create transformation matrix
            # This matrix transforms points from original coordinates to new coordinates
            # Each row represents where the original [1,0,0], [0,1,0], [0,0,1] map to
            transform_matrix = np.array([new_x, new_y, new_z])
            
            # Apply transformation
            transformed = (transform_matrix @ target_vecs_original.T).T
            
            # Compute error
            mean_err, max_err, qpos = compute_error(retargeting, transformed)
            
            transform_count += 1
            transform_name = f"{x_names[i]}→X, {direction_names[j]}→Y"
            results.append((transform_name, mean_err, max_err, transformed, qpos, transform_matrix))
            
            if transform_count <= 20 or mean_err < 0.1:  # Show first 20 and any good ones
                z_name = f"({new_z[0]:+.0f},{new_z[1]:+.0f},{new_z[2]:+.0f})"
                print(f"{transform_count:2d}. {transform_name:<15} Z={z_name} → Mean: {mean_err:.4f}, Max: {max_err:.4f}")
    
    print(f"\nTested {transform_count} total transformations")
    print("-" * 80)
    
    # Test different scaling factors with the identity transform
    print("\nTesting scaling factors with original coordinates:")
    print("-" * 50)
    
    scales = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    for scale in scales:
        transformed = target_vecs_original * scale
        mean_err, max_err, qpos = compute_error(retargeting, transformed)
        results.append((f"Scale {scale}x", mean_err, max_err, transformed, qpos, np.eye(3) * scale))
        print(f"  Scale {scale:>4}x: Mean={mean_err:.4f}, Max={max_err:.4f}")
    
    # Find and test the best coordinate transform with different scales
    coord_results = [(name, err, matrix) for name, err, _, _, _, matrix in results if not name.startswith('Scale')]
    if coord_results:
        best_coord = min(coord_results, key=lambda x: x[1])
        best_matrix = best_coord[2]
        
        print(f"\nTesting best coordinate transform ({best_coord[0]}) with different scales:")
        print("-" * 60)
        
        for scale in scales:
            # Apply best coordinate transform then scale
            transformed = (best_matrix @ target_vecs_original.T).T * scale
            mean_err, max_err, qpos = compute_error(retargeting, transformed)
            results.append((f"{best_coord[0]} + Scale {scale}x", mean_err, max_err, transformed, qpos, best_matrix * scale))
            print(f"  Scale {scale:>4}x: Mean={mean_err:.4f}, Max={max_err:.4f}")
    
    # Sort all results by mean error
    results.sort(key=lambda x: x[1])
    
    print("\n" + "=" * 80)
    print("🏆 TOP 15 BEST RESULTS:")
    print("=" * 80)
    print(f"{'Rank':<4} {'Transform':<35} {'Mean Error':<12} {'Max Error':<11} {'Joint Angles'}")
    print("-" * 80)
    
    for i, (name, mean_err, max_err, transformed, qpos, matrix) in enumerate(results[:15]):
        if qpos is not None:
            qpos_str = f"[{', '.join([f'{x:.2f}' for x in qpos[:min(4, len(qpos))]])}{'...' if len(qpos) > 4 else ''}]"
        else:
            qpos_str = "Failed"
        print(f"{i+1:<4} {name:<35} {mean_err:<12.4f} {max_err:<11.4f} {qpos_str}")
    
    # Show the best transformation matrix
    best_result = results[0]
    print(f"\n🎯 OPTIMAL TRANSFORMATION:")
    print(f"   Name: {best_result[0]}")
    print(f"   Mean Error: {best_result[1]:.4f}")
    print(f"   Max Error: {best_result[2]:.4f}")
    print(f"   Transformation Matrix:")
    print(f"   {best_result[5]}")
    
    if best_result[4] is not None:
        print(f"   Resulting Joint Angles: {best_result[4]}")
    
    # Show input vs output vectors for best result
    print(f"\n📊 VECTOR COMPARISON (Best Transform):")
    print("-" * 60)
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    
    best_transformed = best_result[3]
    for i, finger in enumerate(finger_names):
        if i < len(target_vecs_original):
            orig = target_vecs_original[i]
            trans = best_transformed[i]
            print(f"{finger:>6}: Original=({orig[0]:+.3f},{orig[1]:+.3f},{orig[2]:+.3f}) → "
                  f"Transformed=({trans[0]:+.3f},{trans[1]:+.3f},{trans[2]:+.3f})")
    
    return best_result

# Usage:
print("Testing coordinate transforms on open hand...")
target_vecs_open = rel_right_fingers_open[:3, tip_indices].T
best_open = test_all_coordinate_transforms(right_retargeting, target_vecs_open, "OPEN HAND TRANSFORMS")

print("\n" + "="*80)
print("Testing coordinate transforms on closed hand...")
target_vecs_closed = rel_right_fingers_closed[:3, tip_indices].T  
best_closed = test_all_coordinate_transforms(right_retargeting, target_vecs_closed, "CLOSED HAND TRANSFORMS")