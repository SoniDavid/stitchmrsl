#!/usr/bin/env python3
"""
Multi-Image ArUco Stitching for Stereo Pair Structure
Processes AB_overlap and BC_overlap directories with multiple image pairs
"""

import cv2
import numpy as np
import json
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
CALIBRATION_FILE = 'buena.json'
STEREO_IMAGES_DIR = 'stereo_images'
OUTPUT_DIR = 'multi_stereo_stitched_output'

ARUCO_DICT = cv2.aruco.DICT_5X5_1000
MIN_COMMON_MARKERS = 3
MIN_TOTAL_POINTS = 8


def load_calibrations(filepath):
    """Loads camera calibrations from JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
            calibrations = data.get("cameras", data)
        print(f"✅ Loaded calibrations for: {list(calibrations.keys())}")
        return calibrations
    except Exception as e:
        print(f"❌ Error loading calibrations: {e}")
        return None


def discover_stereo_pairs(stereo_dir):
    """
    Discover stereo image pairs from the AB_overlap and BC_overlap structure.
    Returns: {pair_name: [(cam1_img, cam2_img), ...]}
    """
    stereo_dir = Path(stereo_dir)
    pairs = {}
    
    # Process AB_overlap (izquierda + central)
    ab_dir = stereo_dir / 'AB_overlap'
    if ab_dir.exists():
        izq_dir = ab_dir / 'izquierda'
        central_dir = ab_dir / 'central'
        
        if izq_dir.exists() and central_dir.exists():
            # Get matching image pairs
            izq_images = sorted(glob.glob(str(izq_dir / "*.jpg")))
            central_images = sorted(glob.glob(str(central_dir / "*.jpg")))
            
            ab_pairs = []
            for izq_img in izq_images:
                filename = Path(izq_img).name
                central_img = central_dir / filename
                if central_img.exists():
                    ab_pairs.append((str(izq_img), str(central_img)))
            
            pairs['AB'] = {
                'cam1': 'izquierda',
                'cam2': 'central', 
                'image_pairs': ab_pairs
            }
            print(f"📸 Found {len(ab_pairs)} AB image pairs")
    
    # Process BC_overlap (central + derecha)
    bc_dir = stereo_dir / 'BC_overlap'
    if bc_dir.exists():
        central_dir = bc_dir / 'central'
        derecha_dir = bc_dir / 'derecha'
        
        if central_dir.exists() and derecha_dir.exists():
            # Get matching image pairs
            central_images = sorted(glob.glob(str(central_dir / "*.jpg")))
            derecha_images = sorted(glob.glob(str(derecha_dir / "*.jpg")))
            
            bc_pairs = []
            for central_img in central_images:
                filename = Path(central_img).name
                derecha_img = derecha_dir / filename
                if derecha_img.exists():
                    bc_pairs.append((str(central_img), str(derecha_img)))
            
            pairs['BC'] = {
                'cam1': 'central',
                'cam2': 'derecha',
                'image_pairs': bc_pairs
            }
            print(f"📸 Found {len(bc_pairs)} BC image pairs")
    
    return pairs


def rectify_image_full_view(image, camera_params):
    """Rectifies a single image using the fisheye model."""
    K = np.array(camera_params['K'])
    D = np.array([d[0] for d in camera_params['dist']])
    image_size = (image.shape[1], image.shape[0])

    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, image_size, np.eye(3), balance=0.0
    )

    rectified_img = cv2.fisheye.undistortImage(image, K, D, Knew=new_K)
    return rectified_img


def detect_aruco_markers_with_pose(image, camera_name="", image_name=""):
    """Detect ArUco markers and calculate their centers and orientations."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    detectorParams = cv2.aruco.DetectorParameters()
    detectorParams.adaptiveThreshWinSizeMin = 3
    detectorParams.adaptiveThreshWinSizeMax = 23
    detectorParams.adaptiveThreshWinSizeStep = 10
    
    detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(gray)
    
    markers = {}
    if marker_ids is not None and len(marker_ids) > 0:
        for i, marker_id in enumerate(marker_ids.flatten()):
            corners = marker_corners[i][0]
            
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            
            dx = corners[1, 0] - corners[0, 0]
            dy = corners[1, 1] - corners[0, 1]
            angle = np.arctan2(dy, dx)
            
            size1 = np.linalg.norm(corners[2] - corners[0])
            size2 = np.linalg.norm(corners[3] - corners[1])
            avg_size = (size1 + size2) / 2.0
            
            markers[marker_id] = {
                'center': (center_x, center_y),
                'corners': corners,
                'angle': angle,
                'size': avg_size,
                'image_name': image_name
            }
        
        print(f"    {image_name} - {camera_name}: {len(markers)} markers: {list(markers.keys())}")
    else:
        print(f"    {image_name} - {camera_name}: No markers detected")
    
    return markers


def process_stereo_pair_sequence(pair_info, calibrations):
    """Process a sequence of stereo image pairs and accumulate correspondences."""
    cam1_name = pair_info['cam1']
    cam2_name = pair_info['cam2']
    image_pairs = pair_info['image_pairs']
    
    if cam1_name not in calibrations or cam2_name not in calibrations:
        print(f"❌ Missing calibrations for {cam1_name} or {cam2_name}")
        return [], [], []
    
    cam1_params = calibrations[cam1_name]
    cam2_params = calibrations[cam2_name]
    
    all_points1 = []
    all_points2 = []
    correspondence_info = []
    
    print(f"  Processing {len(image_pairs)} image pairs...")
    
    for i, (img1_path, img2_path) in enumerate(image_pairs):
        image_name = Path(img1_path).stem
        print(f"    📷 Processing pair {i+1}/{len(image_pairs)}: {image_name}")
        
        # Load images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            print(f"      ⚠️ Failed to load images")
            continue
        
        # Rectify images
        rect_img1 = rectify_image_full_view(img1, cam1_params)
        rect_img2 = rectify_image_full_view(img2, cam2_params)
        
        # Detect markers
        markers1 = detect_aruco_markers_with_pose(rect_img1, cam1_name, image_name)
        markers2 = detect_aruco_markers_with_pose(rect_img2, cam2_name, image_name)
        
        # Find common markers
        common_ids = set(markers1.keys()) & set(markers2.keys())
        
        if len(common_ids) >= MIN_COMMON_MARKERS:
            for marker_id in common_ids:
                point1 = markers1[marker_id]['center']
                point2 = markers2[marker_id]['center']
                
                all_points1.append(point1)
                all_points2.append(point2)
                correspondence_info.append({
                    'image_name': image_name,
                    'marker_id': marker_id,
                    'point1': point1,
                    'point2': point2,
                    'cam1': cam1_name,
                    'cam2': cam2_name
                })
            
            print(f"      ✅ Found {len(common_ids)} common markers")
            
            # Save debug visualization for this pair
            debug_viz = create_pair_debug_visualization(
                rect_img1, rect_img2, markers1, markers2, common_ids, 
                f"{cam1_name}_{cam2_name}_{image_name}"
            )
            debug_path = os.path.join(OUTPUT_DIR, f"debug_{cam1_name}_{cam2_name}_{image_name}.jpg")
            cv2.imwrite(debug_path, debug_viz)
            
        else:
            print(f"      ❌ Only {len(common_ids)} common markers (need {MIN_COMMON_MARKERS})")
    
    print(f"  📊 Total correspondences accumulated: {len(all_points1)}")
    return np.array(all_points1), np.array(all_points2), correspondence_info


def create_pair_debug_visualization(img1, img2, markers1, markers2, common_ids, pair_name):
    """Create debug visualization for a single image pair."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    max_h = max(h1, h2)
    combined_w = w1 + w2
    combined_img = np.zeros((max_h, combined_w, 3), dtype=np.uint8)
    
    combined_img[:h1, :w1] = img1
    combined_img[:h2, w1:w1+w2] = img2
    
    # Draw markers
    for marker_id, data in markers1.items():
        x, y = data['center']
        color = (0, 255, 0) if marker_id in common_ids else (0, 255, 255)
        cv2.circle(combined_img, (int(x), int(y)), 6, color, 2)
        cv2.putText(combined_img, str(marker_id), (int(x+8), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    for marker_id, data in markers2.items():
        x, y = data['center']
        color = (0, 255, 0) if marker_id in common_ids else (0, 255, 255)
        cv2.circle(combined_img, (int(x + w1), int(y)), 6, color, 2)
        cv2.putText(combined_img, str(marker_id), (int(x + w1 + 8), int(y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Draw correspondence lines
    for marker_id in common_ids:
        if marker_id in markers1 and marker_id in markers2:
            x1, y1 = markers1[marker_id]['center']
            x2, y2 = markers2[marker_id]['center']
            cv2.line(combined_img, (int(x1), int(y1)), (int(x2 + w1), int(y2)), (255, 0, 0), 1)
    
    # Add title
    cv2.putText(combined_img, pair_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined_img, f"Common: {len(common_ids)}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return combined_img


def calculate_robust_transformation(points1, points2, method="homography"):
    """Calculate transformation with robust statistics."""
    if len(points1) < MIN_TOTAL_POINTS:
        print(f"    ❌ Insufficient points: {len(points1)} < {MIN_TOTAL_POINTS}")
        return np.eye(3), {"error": float('inf'), "method": method}
    
    points1 = points1.astype(np.float32)
    points2 = points2.astype(np.float32)
    
    if method == "similarity":
        result = cv2.estimateAffinePartial2D(
            points2, points1, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        transform_cv, mask = result
        
        if transform_cv is not None:
            transform = np.eye(3, dtype=np.float32)
            transform[:2, :] = transform_cv
            
            # Calculate statistics
            transformed_points2 = cv2.transform(points2.reshape(-1, 1, 2), transform_cv).reshape(-1, 2)
            residuals = np.linalg.norm(points1 - transformed_points2, axis=1)
            
            # Extract transformation parameters
            scale = np.sqrt(transform_cv[0, 0]**2 + transform_cv[0, 1]**2)
            rotation = np.arctan2(transform_cv[0, 1], transform_cv[0, 0])
            translation = transform_cv[:, 2]
            
            info = {
                "method": "similarity",
                "mean_error": float(np.mean(residuals)),
                "median_error": float(np.median(residuals)),
                "std_error": float(np.std(residuals)),
                "max_error": float(np.max(residuals)),
                "num_points": len(points1),
                "inliers": int(np.sum(mask)) if mask is not None else len(points1),
                "rotation_deg": float(np.degrees(rotation)),
                "scale": float(scale),
                "translation_x": float(translation[0]),
                "translation_y": float(translation[1]),
                "residuals": residuals.tolist()
            }
        else:
            transform = np.eye(3)
            info = {"method": "similarity", "error": float('inf'), "failed": True}
            
    elif method == "homography":
        if len(points1) >= 4:
            transform, mask = cv2.findHomography(
                points2, points1, 
                cv2.RANSAC, 
                ransacReprojThreshold=5.0,
                maxIters=2000,
                confidence=0.99
            )
            
            if transform is not None:
                transformed_points2 = cv2.perspectiveTransform(points2.reshape(-1, 1, 2), transform).reshape(-1, 2)
                residuals = np.linalg.norm(points1 - transformed_points2, axis=1)
                
                info = {
                    "method": "homography",
                    "mean_error": float(np.mean(residuals)),
                    "median_error": float(np.median(residuals)),
                    "std_error": float(np.std(residuals)),
                    "max_error": float(np.max(residuals)),
                    "num_points": len(points1),
                    "inliers": int(np.sum(mask)) if mask is not None else len(points1),
                    "residuals": residuals.tolist()
                }
            else:
                transform = np.eye(3)
                info = {"method": "homography", "error": float('inf'), "failed": True}
        else:
            print(f"    ⚠️ Not enough points for homography: {len(points1)}")
            return calculate_robust_transformation(points1, points2, "similarity")
    
    return transform, info


def analyze_spatial_errors(correspondence_info, residuals, pair_name):
    """Analyze spatial distribution of errors."""
    print(f"\n  🔍 SPATIAL ERROR ANALYSIS for {pair_name}")
    
    if not correspondence_info or not residuals:
        print("    No data for spatial analysis")
        return
    
    # Group by image regions (assuming ~2000px images)
    regions = {'left': [], 'center': [], 'right': [], 'top': [], 'bottom': []}
    
    for corr, error in zip(correspondence_info, residuals):
        x, y = corr['point1']
        
        # Horizontal regions
        if x < 800:
            regions['left'].append(error)
        elif x > 1200:
            regions['right'].append(error)
        else:
            regions['center'].append(error)
        
        # Vertical regions  
        if y < 600:
            regions['top'].append(error)
        else:
            regions['bottom'].append(error)
    
    # Print regional statistics
    for region, errors in regions.items():
        if errors:
            mean_err = np.mean(errors)
            print(f"    {region:8}: {len(errors):2} pts, mean={mean_err:.2f}px")
    
    # Check for systematic bias
    all_errors = [e for errs in regions.values() for e in errs if errs]
    if all_errors:
        region_means = [np.mean(errs) if errs else 0 for errs in [regions['left'], regions['center'], regions['right']]]
        error_variance = np.var([m for m in region_means if m > 0])
        
        print(f"    📊 Horizontal error variance: {error_variance:.3f}")
        if error_variance > 2.0:
            print("    ⚠️  HIGH horizontal error variance - check intrinsics!")
        else:
            print("    ✅ Spatial errors look uniform")


def apply_transformation_and_stitch(img1, img2, transform, method="homography", blend_mode="average"):
    """Apply transformation and stitch two images."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    if method == "similarity":
        # For similarity transform, use affine warping
        transform_2x3 = transform[:2, :]
        
        # Find corners of img2 after transformation
        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        transformed_corners2 = cv2.transform(corners2.reshape(-1, 1, 2), transform_2x3).reshape(-1, 2)
        
        # Calculate canvas size
        all_corners = np.vstack([
            [[0, 0], [w1, 0], [w1, h1], [0, h1]],  # img1 corners
            transformed_corners2
        ])
        
        min_x, min_y = np.min(all_corners, axis=0)
        max_x, max_y = np.max(all_corners, axis=0)
        
        canvas_w = int(np.ceil(max_x - min_x))
        canvas_h = int(np.ceil(max_y - min_y))
        
        # Adjust transformation matrix for canvas offset
        offset_transform = transform_2x3.copy()
        offset_transform[0, 2] -= min_x
        offset_transform[1, 2] -= min_y
        
        # Create canvas and place img1
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        img1_offset_x, img1_offset_y = int(-min_x), int(-min_y)
        canvas[img1_offset_y:img1_offset_y+h1, img1_offset_x:img1_offset_x+w1] = img1
        
        # Warp and place img2
        warped_img2 = cv2.warpAffine(img2, offset_transform, (canvas_w, canvas_h))
        
    else:  # homography
        # Find corners of img2 after transformation
        corners2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
        transformed_corners2 = cv2.perspectiveTransform(corners2, transform).reshape(-1, 2)
        
        # Calculate canvas size
        all_corners = np.vstack([
            [[0, 0], [w1, 0], [w1, h1], [0, h1]],
            transformed_corners2
        ])
        
        min_x, min_y = np.min(all_corners, axis=0)
        max_x, max_y = np.max(all_corners, axis=0)
        
        canvas_w = int(np.ceil(max_x - min_x))
        canvas_h = int(np.ceil(max_y - min_y))
        
        # Adjust transformation matrix for canvas offset
        offset_transform = transform.copy()
        offset_transform[0, 2] -= min_x
        offset_transform[1, 2] -= min_y
        
        # Create canvas and place img1
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        img1_offset_x, img1_offset_y = int(-min_x), int(-min_y)
        canvas[img1_offset_y:img1_offset_y+h1, img1_offset_x:img1_offset_x+w1] = img1
        
        # Warp and place img2
        warped_img2 = cv2.warpPerspective(img2, offset_transform, (canvas_w, canvas_h))
    
    # Blend overlapping regions
    mask1 = (np.sum(canvas, axis=2) > 0)
    mask2 = (np.sum(warped_img2, axis=2) > 0)
    overlap_mask = mask1 & mask2
    
    if blend_mode == "average":
        # Simple average blending
        result = canvas.astype(np.float32)
        warped_float = warped_img2.astype(np.float32)
        
        # Blend overlap regions
        result[overlap_mask] = (result[overlap_mask] + warped_float[overlap_mask]) / 2
        
        # Add non-overlapping parts of img2
        result[mask2 & ~mask1] = warped_float[mask2 & ~mask1]
        
    elif blend_mode == "weighted":
        # Distance-based weighted blending (more sophisticated)
        result = canvas.astype(np.float32)
        warped_float = warped_img2.astype(np.float32)
        
        if np.any(overlap_mask):
            # Create distance transforms for weighting
            dist1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
            dist2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)
            
            # Normalize weights in overlap region
            total_dist = dist1 + dist2
            weight1 = np.divide(dist1, total_dist, out=np.zeros_like(dist1), where=total_dist!=0)
            weight2 = np.divide(dist2, total_dist, out=np.zeros_like(dist2), where=total_dist!=0)
            
            # Apply weighted blending
            for c in range(3):
                result[overlap_mask, c] = (weight1[overlap_mask] * result[overlap_mask, c] + 
                                          weight2[overlap_mask] * warped_float[overlap_mask, c])
        
        # Add non-overlapping parts
        result[mask2 & ~mask1] = warped_float[mask2 & ~mask1]
    
    return result.astype(np.uint8)


def create_stitched_images(stereo_pairs, calibrations, results, best_transforms):
    """Create stitched images using the calculated transformations."""
    print(f"\n{'='*60}")
    print("🎨 CREATING STITCHED IMAGES")
    print(f"{'='*60}")
    
    stitched_images = {}
    
    for pair_name, pair_info in stereo_pairs.items():
        cam1_name = pair_info['cam1']
        cam2_name = pair_info['cam2']
        
        print(f"\n🖼️  Creating stitched images for {pair_name} pair...")
        
        # Use the first image pair as representative for stitching
        if not pair_info['image_pairs']:
            continue
            
        img1_path, img2_path = pair_info['image_pairs'][0]
        print(f"    Using representative images: {Path(img1_path).name}")
        
        # Load and rectify images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        if img1 is None or img2 is None:
            continue
            
        cam1_params = calibrations[cam1_name]
        cam2_params = calibrations[cam2_name]
        
        rect_img1 = rectify_image_full_view(img1, cam1_params)
        rect_img2 = rectify_image_full_view(img2, cam2_params)
        
        # Try both methods if available
        for method in ["similarity", "homography"]:
            result_key = f"{pair_name}_{method}"
            
            if result_key not in best_transforms:
                continue
                
            transform = best_transforms[result_key]
            info = results[result_key]
            
            print(f"    🔄 Stitching with {method} (error: {info['mean_error']:.2f}px)...")
            
            # Create stitched image with average blending
            stitched_avg = apply_transformation_and_stitch(
                rect_img1, rect_img2, transform, method, "average"
            )
            
            # Create stitched image with weighted blending
            stitched_weighted = apply_transformation_and_stitch(
                rect_img1, rect_img2, transform, method, "weighted"
            )
            
            # Save both versions
            avg_path = os.path.join(OUTPUT_DIR, f"stitched_{result_key}_average.jpg")
            weighted_path = os.path.join(OUTPUT_DIR, f"stitched_{result_key}_weighted.jpg")
            
            cv2.imwrite(avg_path, stitched_avg)
            cv2.imwrite(weighted_path, stitched_weighted)
            
            print(f"      ✅ Saved: {Path(avg_path).name}")
            print(f"      ✅ Saved: {Path(weighted_path).name}")
            
            # Store for potential panorama creation
            stitched_images[result_key] = {
                'image': stitched_weighted, # Store the weighted blended image
                'transform': transform,
                'method': method,
                'error': info['mean_error'],
                'cam1_img': rect_img1,
                'cam2_img': rect_img2
            }
    
    return stitched_images

def create_full_panorama(stitched_images, stereo_pairs, calibrations):
    """
    Creates a full vertical panorama by stacking the three rectified camera views.
    It uses the transformations to place 'izquierda' and 'derecha' relative to 'central'.
    """
    print(f"\n🌅 CREATING FULL PANORAMA (Vertical Stack)")
    
    # Find the best transformation from central to izquierda (AB) and central to derecha (BC)
    best_ab_key = min([k for k in stitched_images.keys() if 'AB_' in k], 
                      key=lambda k: stitched_images[k]['error']) if any('AB_' in k for k in stitched_images.keys()) else None
    
    best_bc_key = min([k for k in stitched_images.keys() if 'BC_' in k], 
                      key=lambda k: stitched_images[k]['error']) if any('BC_' in k for k in stitched_images.keys()) else None
    
    if not best_ab_key or not best_bc_key:
        print("    ❌ Need stitched results for both AB and BC pairs to create a full panorama.")
        return None

    # Get the data for the best transformations and rectified images
    ab_data = stitched_images[best_ab_key]
    bc_data = stitched_images[best_bc_key]
    
    # Get the rectified images from the stored data
    # These are already rectified, so no need to reload and rectify them again
    base_izq = ab_data['cam1_img']
    base_central_ab = ab_data['cam2_img']
    base_central_bc = bc_data['cam1_img']
    base_der = bc_data['cam2_img']

    # Use a consistent central image. We'll use the one from the AB pair.
    base_central = base_central_ab

    h_izq, w_izq = base_izq.shape[:2]
    h_central, w_central = base_central.shape[:2]
    h_der, w_der = base_der.shape[:2]
    
    # Your transformations are from cam2 -> cam1, so:
    # AB pair (izquierda + central): transform is from central -> izquierda
    # BC pair (central + derecha): transform is from derecha -> central
    
    transform_central_to_izq = ab_data['transform']
    transform_der_to_central = bc_data['transform']
    
    # To place 'izquierda' on top of 'central', we need to warp 'izquierda' relative to 'central'.
    # The transformation 'transform_central_to_izq' warps 'central' onto 'izquierda'.
    # We want to warp 'izquierda' onto the 'central' canvas. So we need the inverse.
    transform_izq_to_central = np.linalg.inv(transform_central_to_izq)
    
    # To place 'derecha' on the bottom of 'central', we need to warp 'derecha' relative to 'central'.
    # The transformation 'transform_der_to_central' warps 'derecha' onto 'central'. This is what we need.
    transform_der_to_central = bc_data['transform']

    print(f"    📐 Base Image Sizes: Izquierda={w_izq}x{h_izq}, Central={w_central}x{h_central}, Derecha={w_der}x{h_der}")

    # --- Determine the canvas size for the final vertical panorama ---
    
    # 1. Place the 'central' image at (0, 0) of a temporary coordinate system.
    # Its corners are at (0,0), (w_central,0), (w_central,h_central), (0,h_central)
    # FIX: Reshape the corners to be a 2D array (4, 2) instead of (4, 1, 2)
    central_corners = np.float32([[0, 0], [w_central, 0], [w_central, h_central], [0, h_central]]).reshape(-1, 2)
    
    # 2. Find the transformed corners of the 'izquierda' image relative to the 'central' camera.
    izq_corners = np.float32([[0, 0], [w_izq, 0], [w_izq, h_izq], [0, h_izq]]).reshape(-1, 2)
    izq_transformed_corners = cv2.perspectiveTransform(izq_corners.reshape(-1, 1, 2), transform_izq_to_central).reshape(-1, 2)
    
    # 3. Find the transformed corners of the 'derecha' image relative to the 'central' camera.
    der_corners = np.float32([[0, 0], [w_der, 0], [w_der, h_der], [0, h_der]]).reshape(-1, 2)
    der_transformed_corners = cv2.perspectiveTransform(der_corners.reshape(-1, 1, 2), transform_der_to_central).reshape(-1, 2)
    
    # 4. Combine all corners to find the bounding box of the final canvas.
    all_corners = np.vstack([
        central_corners,
        izq_transformed_corners,
        der_transformed_corners
    ])

    min_x, min_y = np.min(all_corners, axis=0)
    max_x, max_y = np.max(all_corners, axis=0)
    
    # Calculate canvas dimensions
    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    print(f"    🖼️  Panorama Canvas Size: {canvas_w}x{canvas_h}")

    # --- Create the vertical panorama by warping and blending ---
    
    # Create the canvas
    panorama = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Create an offset transformation to shift everything to the positive coordinate space.
    offset_transform = np.eye(3, dtype=np.float32)
    offset_transform[0, 2] = -min_x
    offset_transform[1, 2] = -min_y
    
    # Warp each image into the canvas using the offset transformations
    
    # 1. Warp the central image (base)
    # The central image is our reference, so we just apply the offset
    warped_central = cv2.warpPerspective(base_central, offset_transform, (canvas_w, canvas_h))
    
    # 2. Warp the izquierda image
    # Combine the transform from izq -> central with the offset
    izq_to_canvas_transform = offset_transform @ transform_izq_to_central
    warped_izq = cv2.warpPerspective(base_izq, izq_to_canvas_transform, (canvas_w, canvas_h))
    
    # 3. Warp the derecha image
    # Combine the transform from der -> central with the offset
    der_to_canvas_transform = offset_transform @ transform_der_to_central
    warped_der = cv2.warpPerspective(base_der, der_to_canvas_transform, (canvas_w, canvas_h))

    # Blend the images. Use a simple maximum value blending to avoid black regions.
    # This assumes the images will stack on top of each other.
    panorama = np.maximum(warped_izq, warped_central)
    panorama = np.maximum(panorama, warped_der)

    # Optional: You can refine the blending at the seams if needed, but for now, max blending works.
    
    # Final cleanup to remove black borders
    gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_panorama, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Crop the image to the bounding box of the largest contour
        panorama_cropped = panorama[y:y+h, x:x+w]
        print("    ✂️  Cropped final panorama to remove black borders.")
    else:
        panorama_cropped = panorama

    # Save the final panorama
    panorama_path = os.path.join(OUTPUT_DIR, "full_panorama_final_fixed.jpg")
    cv2.imwrite(panorama_path, panorama_cropped)
    
    print(f"    ✅ FINAL vertical panorama saved: {Path(panorama_path).name}")
    print(f"    📐 Final size: {panorama_cropped.shape[1]}x{panorama_cropped.shape[0]} (W×H)")
    print(f"    🎯 Layout: IZQUIERDA (top) → CENTRAL (middle) → DERECHA (bottom)")
    
    # Create debug version with clear labels and boundaries
    debug_panorama = panorama.copy()
    
    # Add colored labels and lines for debug
    # Now that we fixed the corner array shape, these calculations should work.
    izq_y_center = (izq_transformed_corners[:, 1].min() + izq_transformed_corners[:, 1].max()) / 2
    central_y_center = (central_corners[:, 1].min() + central_corners[:, 1].max()) / 2
    der_y_center = (der_transformed_corners[:, 1].min() + der_transformed_corners[:, 1].max()) / 2
    
    cv2.putText(debug_panorama, "IZQUIERDA (TOP)", (50, int(izq_y_center - min_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
    cv2.putText(debug_panorama, "CENTRAL (MIDDLE)", (50, int(central_y_center - min_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 4)  
    cv2.putText(debug_panorama, "DERECHA (BOTTOM)", (50, int(der_y_center - min_y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    
    # Draw boundary lines (these will be diagonal in perspective)
    # The boundaries are where the original images end.
    
    debug_path = os.path.join(OUTPUT_DIR, "full_panorama_final_debug_fixed.jpg")
    cv2.imwrite(debug_path, debug_panorama)
    print(f"    🔍 Debug version with labels: {Path(debug_path).name}")
    
    return panorama_cropped

def main():
    """Main execution pipeline for stereo pair processing."""
    print("🚀 Starting multi-image stereo stitching pipeline")
    print(f"📁 Looking for images in: {STEREO_IMAGES_DIR}")
    
    # Load calibrations
    calibrations = load_calibrations(CALIBRATION_FILE)
    if not calibrations:
        return False
    
    # Discover stereo pairs
    stereo_pairs = discover_stereo_pairs(STEREO_IMAGES_DIR)
    if not stereo_pairs:
        print("❌ No stereo pairs found!")
        return False
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each stereo pair type
    results = {}
    best_transforms = {}
    
    for pair_name, pair_info in stereo_pairs.items():
        print(f"\n{'='*60}")
        print(f"🔄 PROCESSING {pair_name} PAIR: {pair_info['cam1']} + {pair_info['cam2']}")
        print(f"{'='*60}")
        
        # Process all image pairs and accumulate correspondences
        points1, points2, correspondence_info = process_stereo_pair_sequence(pair_info, calibrations)
        
        if len(points1) < MIN_TOTAL_POINTS:
            print(f"❌ Insufficient correspondences for {pair_name}: {len(points1)}")
            continue
        
        # Test both transformation methods
        for method in ["similarity", "homography"]:
            print(f"\n  🧮 Testing {method.upper()} transformation:")
            
            transform, info = calculate_robust_transformation(points1, points2, method)
            
            if info.get('failed', False):
                print(f"    ❌ {method} failed")
                continue
            
            # Store results and transform
            result_key = f"{pair_name}_{method}"
            results[result_key] = info
            best_transforms[result_key] = transform
            
            # Print summary
            if method == "similarity":
                print(f"    ✅ Mean error: {info['mean_error']:.2f}px")
                print(f"    📐 Rotation: {info['rotation_deg']:.2f}°")
                print(f"    📏 Scale: {info['scale']:.3f}")
                print(f"    📍 Translation: ({info['translation_x']:.1f}, {info['translation_y']:.1f})")
            else:
                print(f"    ✅ Mean error: {info['mean_error']:.2f}px")
                print(f"    👥 Inliers: {info['inliers']}/{info['num_points']}")
            
            # Analyze spatial errors
            if 'residuals' in info:
                analyze_spatial_errors(correspondence_info, info['residuals'], f"{pair_name}_{method}")
    
    # Create stitched images using best transformations
    stitched_images = create_stitched_images(stereo_pairs, calibrations, results, best_transforms)
    
    # Create full panorama if both pairs were processed
    if stitched_images:
        create_full_panorama(stitched_images, stereo_pairs, calibrations)
    
    # Save comprehensive results
    output_file = os.path.join(OUTPUT_DIR, 'stereo_stitching_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'processed_pairs': list(stereo_pairs.keys()),
            'total_image_pairs': sum(len(p['image_pairs']) for p in stereo_pairs.values()),
            'results': results,
            'stitched_images_created': list(stitched_images.keys())
        }, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("🎉 FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for result_key, info in results.items():
        if not info.get('failed', False):
            print(f"{result_key:20}: {info['mean_error']:.2f}px ({info['num_points']} points)")
            if 'rotation_deg' in info:
                print(f"{'':20}  → Rotation: {info['rotation_deg']:.1f}°, Scale: {info['scale']:.3f}")
    
    print(f"\n📁 Results saved in: {OUTPUT_DIR}")
    print(f"📄 Detailed results: {output_file}")
    print(f"🖼️  Stitched images: {len(stitched_images) * 2} files")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
