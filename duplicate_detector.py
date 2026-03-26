#!/usr/bin/env python3
"""
Visual Duplicate Image Detector

Detects duplicate images based on visual similarity (perceptual hash),
not just file hash. Can detect duplicates even if:
- Different file names
- Different formats (JPG, PNG, WebP, etc.)
- Different compression levels

Selection priority for keeping the "best" file:
1. Largest dimensions (width x height)
2. If same dimensions: Smallest file size (most efficient compression)

Usage:
    python duplicate_detector.py <source_folder> [--duplicates-folder <path>] [--threshold <0-64>] [--dry-run]

Example:
    python duplicate_detector.py /path/to/images --duplicates-folder /path/to/duplicates --threshold 5
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them with: pip install Pillow imagehash")
    sys.exit(1)


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp', '.tiff', '.tif', '.heic', '.heif'}


@dataclass
class ImageInfo:
    """Store information about an image file."""
    path: Path
    width: int
    height: int
    file_size: int
    phash: str
    dhash: str
    
    @property
    def pixel_count(self) -> int:
        """Total number of pixels (width x height)."""
        return self.width * self.height
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Return dimensions as tuple."""
        return (self.width, self.height)
    
    def __str__(self) -> str:
        return f"{self.path.name} ({self.width}x{self.height}, {self.file_size:,} bytes)"


def get_image_info(file_path: Path) -> Optional[ImageInfo]:
    """
    Extract image information including dimensions and perceptual hashes.
    
    Args:
        file_path: Path to the image file
        
    Returns:
        ImageInfo object or None if the file cannot be processed
    """
    try:
        with Image.open(file_path) as img:
            # Get dimensions
            width, height = img.size
            
            # Convert to RGB if necessary (for hash calculation)
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            
            # Calculate perceptual hashes
            phash = str(imagehash.phash(img))
            dhash = str(imagehash.dhash(img))
            
            # Get file size
            file_size = file_path.stat().st_size
            
            return ImageInfo(
                path=file_path,
                width=width,
                height=height,
                file_size=file_size,
                phash=phash,
                dhash=dhash
            )
    except Exception as e:
        print(f"⚠️  Warning: Cannot process {file_path}: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Calculate the Hamming distance between two hash strings.
    
    Args:
        hash1: First hash string
        hash2: Second hash string
        
    Returns:
        Number of differing bits
    """
    h1 = imagehash.hex_to_hash(hash1)
    h2 = imagehash.hex_to_hash(hash2)
    return h1 - h2


def are_visually_similar(img1: ImageInfo, img2: ImageInfo, threshold: int = 5) -> bool:
    """
    Check if two images are visually similar based on perceptual hash.
    
    Args:
        img1: First image info
        img2: Second image info
        threshold: Maximum Hamming distance to consider as similar (0-64)
                   0 = exact match, higher = more lenient
        
    Returns:
        True if images are visually similar
    """
    # Use both pHash and dHash for more accurate comparison
    phash_distance = hamming_distance(img1.phash, img2.phash)
    dhash_distance = hamming_distance(img1.dhash, img2.dhash)
    
    # Both hashes should be within threshold
    return phash_distance <= threshold and dhash_distance <= threshold


def select_best_image(images: List[ImageInfo]) -> ImageInfo:
    """
    Select the best image from a list of duplicates.
    
    Priority:
    1. Largest dimensions (width x height)
    2. If same dimensions: Smallest file size
    
    Args:
        images: List of duplicate images
        
    Returns:
        The best image to keep
    """
    return max(images, key=lambda img: (img.pixel_count, -img.file_size))


def find_all_images(source_folder: Path) -> List[Path]:
    """
    Find all image files in the source folder and subfolders.
    
    Args:
        source_folder: Root folder to search
        
    Returns:
        List of image file paths
    """
    images = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                images.append(file_path)
    return images


def group_duplicates(images: List[ImageInfo], threshold: int = 5) -> List[List[ImageInfo]]:
    """
    Group visually similar images together.
    
    Uses Union-Find algorithm for efficient grouping.
    
    Args:
        images: List of image info objects
        threshold: Similarity threshold
        
    Returns:
        List of duplicate groups (each group has 2+ images)
    """
    n = len(images)
    
    # Union-Find data structure
    parent = list(range(n))
    rank = [0] * n
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
    
    # Group by pHash first for efficiency
    hash_groups: Dict[str, List[int]] = defaultdict(list)
    for i, img in enumerate(images):
        hash_groups[img.phash].append(i)
    
    # Compare within same hash group (exact matches)
    for indices in hash_groups.values():
        for i in range(1, len(indices)):
            union(indices[0], indices[i])
    
    # Compare across different hash groups for near-duplicates
    unique_hashes = list(hash_groups.keys())
    for i in range(len(unique_hashes)):
        for j in range(i + 1, len(unique_hashes)):
            # Quick check: if any pair is similar, merge the groups
            idx1 = hash_groups[unique_hashes[i]][0]
            idx2 = hash_groups[unique_hashes[j]][0]
            
            if are_visually_similar(images[idx1], images[idx2], threshold):
                union(idx1, idx2)
    
    # Build groups
    groups: Dict[int, List[ImageInfo]] = defaultdict(list)
    for i, img in enumerate(images):
        groups[find(i)].append(img)
    
    # Return only groups with duplicates (2+ images)
    return [group for group in groups.values() if len(group) > 1]


def process_duplicates(
    source_folder: Path,
    duplicates_folder: Path,
    threshold: int = 5,
    dry_run: bool = False,
    max_workers: int = 4
) -> Tuple[int, int, int]:
    """
    Main function to find and process duplicate images.
    
    Args:
        source_folder: Folder to scan for duplicates
        duplicates_folder: Folder to move duplicates to
        threshold: Similarity threshold (0-64)
        dry_run: If True, only report duplicates without moving
        max_workers: Number of parallel workers for image processing
        
    Returns:
        Tuple of (total_images, duplicate_groups, files_moved)
    """
    print(f"\n{'='*60}")
    print(f"🔍 Visual Duplicate Image Detector")
    print(f"{'='*60}")
    print(f"📁 Source folder: {source_folder}")
    print(f"📁 Duplicates folder: {duplicates_folder}")
    print(f"🎯 Similarity threshold: {threshold}")
    print(f"{'='*60}\n")
    
    # Find all images
    print("📂 Scanning for images...")
    image_paths = find_all_images(source_folder)
    
    # Exclude images already in duplicates folder
    if duplicates_folder.exists():
        image_paths = [p for p in image_paths if not str(p).startswith(str(duplicates_folder))]
    
    print(f"   Found {len(image_paths)} images\n")
    
    if not image_paths:
        print("❌ No images found.")
        return 0, 0, 0
    
    # Process images in parallel
    print("🔄 Analyzing images (this may take a while)...")
    images: List[ImageInfo] = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_image_info, path): path for path in image_paths}
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                images.append(result)
            
            # Progress indicator
            if i % 100 == 0 or i == len(image_paths):
                print(f"   Processed {i}/{len(image_paths)} images...")
    
    print(f"   Successfully analyzed {len(images)} images\n")
    
    # Find duplicate groups
    print("🔍 Finding visual duplicates...")
    duplicate_groups = group_duplicates(images, threshold)
    
    if not duplicate_groups:
        print("✅ No duplicates found!")
        return len(images), 0, 0
    
    print(f"   Found {len(duplicate_groups)} groups of duplicates\n")
    
    # Process each group
    files_moved = 0
    
    if not dry_run:
        duplicates_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print("📋 Duplicate Groups:")
    print(f"{'='*60}\n")
    
    for i, group in enumerate(duplicate_groups, 1):
        best = select_best_image(group)
        duplicates = [img for img in group if img != best]
        
        print(f"Group {i}:")
        print(f"  ✅ KEEP: {best}")
        
        for dup in duplicates:
            print(f"  ❌ MOVE: {dup}")
            
            if not dry_run:
                # Create relative path structure in duplicates folder
                try:
                    rel_path = dup.path.relative_to(source_folder)
                except ValueError:
                    rel_path = dup.path.name
                
                dest_path = duplicates_folder / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Handle filename conflicts
                if dest_path.exists():
                    stem = dest_path.stem
                    suffix = dest_path.suffix
                    counter = 1
                    while dest_path.exists():
                        dest_path = dest_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1
                
                shutil.move(str(dup.path), str(dest_path))
                files_moved += 1
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("📊 Summary:")
    print(f"{'='*60}")
    print(f"   Total images analyzed: {len(images)}")
    print(f"   Duplicate groups found: {len(duplicate_groups)}")
    
    if dry_run:
        total_duplicates = sum(len(group) - 1 for group in duplicate_groups)
        print(f"   Files to be moved: {total_duplicates}")
        print(f"\n   ℹ️  DRY RUN - No files were moved")
        print(f"   Run without --dry-run to actually move files")
    else:
        print(f"   Files moved: {files_moved}")
        print(f"\n   ✅ Duplicates moved to: {duplicates_folder}")
    
    return len(images), len(duplicate_groups), files_moved


def main():
    parser = argparse.ArgumentParser(
        description="Detect and move visually duplicate images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/images
  %(prog)s /path/to/images --duplicates-folder /path/to/duplicates
  %(prog)s /path/to/images --threshold 10 --dry-run
  
Threshold values:
  0  = Exact visual match only
  5  = Default, catches most duplicates (recommended)
  10 = More lenient, may catch slightly different images
  15+= Very lenient, may have false positives
        """
    )
    
    parser.add_argument(
        "source_folder",
        type=str,
        help="Folder to scan for duplicate images (includes subfolders)"
    )
    
    parser.add_argument(
        "--duplicates-folder", "-d",
        type=str,
        default=None,
        help="Folder to move duplicates to (default: <source>/_duplicates)"
    )
    
    parser.add_argument(
        "--threshold", "-t",
        type=int,
        default=5,
        help="Similarity threshold 0-64 (lower = stricter, default: 5)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )
    
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel workers for processing (default: 4)"
    )
    
    args = parser.parse_args()
    
    source_folder = Path(args.source_folder).resolve()
    
    if not source_folder.exists():
        print(f"❌ Error: Source folder does not exist: {source_folder}")
        sys.exit(1)
    
    if not source_folder.is_dir():
        print(f"❌ Error: Source path is not a directory: {source_folder}")
        sys.exit(1)
    
    if args.duplicates_folder:
        duplicates_folder = Path(args.duplicates_folder).resolve()
    else:
        duplicates_folder = source_folder / "_duplicates"
    
    # Validate threshold
    if not 0 <= args.threshold <= 64:
        print(f"❌ Error: Threshold must be between 0 and 64")
        sys.exit(1)
    
    try:
        process_duplicates(
            source_folder=source_folder,
            duplicates_folder=duplicates_folder,
            threshold=args.threshold,
            dry_run=args.dry_run,
            max_workers=args.workers
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
