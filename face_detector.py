#!/usr/bin/env python3
"""
Face Detector Script (DNN Version)
===================================
Mendeteksi wajah dalam gambar menggunakan Deep Neural Network dan memindahkan file ke folder:
- no_face/     : Gambar tanpa wajah
- 1_face/      : Gambar dengan 1 wajah
- multi_face/  : Gambar dengan 2+ wajah

Usage:
    python face_detector.py <source_folder> [--output <output_folder>]

Example:
    python face_detector.py ./images
    python face_detector.py ./images --output ./sorted_images
"""

import os
import sys
import shutil
import argparse
import urllib.request
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

# Output folder names
FOLDER_NO_FACE = 'no_face'
FOLDER_ONE_FACE = '1_face'
FOLDER_MULTI_FACE = 'multi_face'

# DNN Model files
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
PROTOTXT_FILE = os.path.join(MODEL_DIR, 'deploy.prototxt')
CAFFEMODEL_FILE = os.path.join(MODEL_DIR, 'res10_300x300_ssd_iter_140000.caffemodel')

# Model URLs (OpenCV's official face detection model)
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
CAFFEMODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"


def download_models():
    """Download DNN model files if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(PROTOTXT_FILE):
        print(f"📥 Downloading deploy.prototxt...")
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_FILE)
        print(f"   ✅ Downloaded to {PROTOTXT_FILE}")
    
    if not os.path.exists(CAFFEMODEL_FILE):
        print(f"📥 Downloading face detection model (10.7 MB)...")
        urllib.request.urlretrieve(CAFFEMODEL_URL, CAFFEMODEL_FILE)
        print(f"   ✅ Downloaded to {CAFFEMODEL_FILE}")


def load_face_detector():
    """
    Load the DNN face detector.
    
    Returns:
        cv2.dnn.Net: The loaded DNN model
    """
    # Download models if needed
    download_models()
    
    if not os.path.exists(PROTOTXT_FILE) or not os.path.exists(CAFFEMODEL_FILE):
        print("❌ Error: Model files not found")
        sys.exit(1)
    
    # Load the DNN model
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_FILE, CAFFEMODEL_FILE)
    
    return net


def detect_faces(image_path: str, net, confidence_threshold: float = 0.5) -> Tuple[int, List]:
    """
    Detect faces in an image using DNN.
    
    Args:
        image_path: Path to the image file
        net: The DNN face detector
        confidence_threshold: Minimum confidence for detection (0.0 - 1.0)
    
    Returns:
        Tuple of (face_count, list of face rectangles with confidence)
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"  ⚠️  Could not read image: {image_path}")
            return -1, []
        
        (h, w) = image.shape[:2]
        
        # Create blob from image
        # The model expects 300x300 images with mean subtraction
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)  # Mean subtraction values
        )
        
        # Pass through network
        net.setInput(blob)
        detections = net.forward()
        
        faces = []
        
        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter by confidence threshold
            if confidence > confidence_threshold:
                # Get bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within image bounds
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)
                
                # Add to faces list
                faces.append({
                    'box': (startX, startY, endX - startX, endY - startY),
                    'confidence': float(confidence)
                })
        
        return len(faces), faces
        
    except Exception as e:
        print(f"  ⚠️  Error processing {image_path}: {str(e)}")
        return -1, []


def get_output_folder(face_count: int) -> str:
    """
    Determine the output folder based on face count.
    
    Args:
        face_count: Number of faces detected
    
    Returns:
        The folder name for the image
    """
    if face_count == 0:
        return FOLDER_NO_FACE
    elif face_count == 1:
        return FOLDER_ONE_FACE
    else:
        return FOLDER_MULTI_FACE


def create_output_folders(base_path: str) -> None:
    """
    Create the output folders if they don't exist.
    
    Args:
        base_path: The base directory for output folders
    """
    for folder in [FOLDER_NO_FACE, FOLDER_ONE_FACE, FOLDER_MULTI_FACE]:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)


def move_image(source_path: str, dest_folder: str, dest_base: str) -> bool:
    """
    Move an image to the destination folder.
    
    Args:
        source_path: Path to the source image
        dest_folder: Name of the destination folder
        dest_base: Base path for destination folders
    
    Returns:
        True if successful, False otherwise
    """
    try:
        filename = os.path.basename(source_path)
        dest_path = os.path.join(dest_base, dest_folder, filename)
        
        # Handle duplicate filenames
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(filename)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_base, dest_folder, f"{name}_{counter}{ext}")
                counter += 1
        
        shutil.move(source_path, dest_path)
        return True
        
    except Exception as e:
        print(f"  ⚠️  Error moving {source_path}: {str(e)}")
        return False


def get_image_files(folder_path: str, recursive: bool = False) -> List[str]:
    """
    Get all image files from a folder.
    
    Args:
        folder_path: Path to the folder to scan
        recursive: Whether to scan subdirectories
    
    Returns:
        List of image file paths
    """
    images = []
    
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            # Skip output folders
            dirs[:] = [d for d in dirs if d not in [FOLDER_NO_FACE, FOLDER_ONE_FACE, FOLDER_MULTI_FACE]]
            
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    images.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check extension
            ext = os.path.splitext(file)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                images.append(file_path)
    
    return sorted(images)


def print_summary(stats: dict) -> None:
    """
    Print processing summary.
    
    Args:
        stats: Dictionary containing processing statistics
    """
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"  📁 Total images processed: {stats['total']}")
    print(f"  😶 No face detected:       {stats['no_face']}")
    print(f"  🙂 One face detected:      {stats['one_face']}")
    print(f"  👥 Multiple faces:         {stats['multi_face']}")
    print(f"  ⚠️  Errors/Skipped:         {stats['errors']}")
    print("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Detect faces in images and sort them into folders (DNN version - more accurate)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python face_detector.py ./images
  python face_detector.py ./images --output ./sorted
  python face_detector.py ./photos --copy
  python face_detector.py ./photos --confidence 0.7
        """
    )
    
    parser.add_argument(
        'source',
        help='Source folder containing images'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output folder for sorted images (default: same as source)',
        default=None
    )
    
    parser.add_argument(
        '--copy', '-c',
        action='store_true',
        help='Copy files instead of moving them'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed face detection info'
    )
    
    parser.add_argument(
        '--confidence', '-conf',
        type=float,
        default=0.5,
        help='Confidence threshold (0.0-1.0, default: 0.5). Lower = more sensitive, Higher = more strict'
    )
    
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Recursively scan subdirectories'
    )
    
    args = parser.parse_args()
    
    # Validate confidence
    if not 0.0 <= args.confidence <= 1.0:
        print("❌ Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Validate source folder
    source_folder = os.path.abspath(args.source)
    if not os.path.isdir(source_folder):
        print(f"❌ Error: Source folder not found: {source_folder}")
        sys.exit(1)
    
    # Set output folder
    output_folder = os.path.abspath(args.output) if args.output else source_folder
    
    print("\n" + "=" * 50)
    print("🔍 FACE DETECTOR (DNN)")
    print("=" * 50)
    print(f"  📂 Source:      {source_folder}")
    print(f"  📁 Output:      {output_folder}")
    print(f"  📋 Mode:        {'Copy' if args.copy else 'Move'}")
    print(f"  🎯 Confidence:  {args.confidence}")
    print(f"  📂 Recursive:   {'Yes' if args.recursive else 'No'}")
    print("=" * 50 + "\n")
    
    # Load face detector
    print("🔄 Loading DNN face detector...")
    net = load_face_detector()
    print("✅ Face detector loaded\n")
    
    # Create output folders
    create_output_folders(output_folder)
    
    # Get image files
    images = get_image_files(source_folder, args.recursive)
    
    if not images:
        print("❌ No image files found in the source folder")
        sys.exit(0)
    
    print(f"📸 Found {len(images)} image(s) to process\n")
    
    # Statistics
    stats = {
        'total': 0,
        'no_face': 0,
        'one_face': 0,
        'multi_face': 0,
        'errors': 0
    }
    
    # Process each image
    for i, image_path in enumerate(images, 1):
        filename = os.path.basename(image_path)
        print(f"[{i}/{len(images)}] Processing: {filename}")
        
        # Detect faces
        face_count, faces = detect_faces(image_path, net, args.confidence)
        
        if face_count < 0:
            stats['errors'] += 1
            continue
        
        stats['total'] += 1
        
        # Determine destination folder
        dest_folder = get_output_folder(face_count)
        
        # Update statistics
        if face_count == 0:
            stats['no_face'] += 1
            icon = "😶"
        elif face_count == 1:
            stats['one_face'] += 1
            icon = "🙂"
        else:
            stats['multi_face'] += 1
            icon = "👥"
        
        # Show confidence info
        if faces and args.verbose:
            conf_str = ", ".join([f"{f['confidence']:.1%}" for f in faces])
            print(f"  {icon} Detected {face_count} face(s) [conf: {conf_str}] → {dest_folder}/")
        else:
            print(f"  {icon} Detected {face_count} face(s) → {dest_folder}/")
        
        if args.verbose and face_count > 0:
            for j, face in enumerate(faces, 1):
                x, y, w, h = face['box']
                conf = face['confidence']
                print(f"     Face {j}: pos=({x}, {y}), size={w}x{h}, confidence={conf:.1%}")
        
        # Move or copy file
        if args.copy:
            try:
                dest_path = os.path.join(output_folder, dest_folder, os.path.basename(image_path))
                # Handle duplicates
                if os.path.exists(dest_path):
                    name, ext = os.path.splitext(os.path.basename(image_path))
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(output_folder, dest_folder, f"{name}_{counter}{ext}")
                        counter += 1
                shutil.copy2(image_path, dest_path)
            except Exception as e:
                print(f"  ⚠️  Error copying: {str(e)}")
                stats['errors'] += 1
        else:
            if not move_image(image_path, dest_folder, output_folder):
                stats['errors'] += 1
    
    # Print summary
    print_summary(stats)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
