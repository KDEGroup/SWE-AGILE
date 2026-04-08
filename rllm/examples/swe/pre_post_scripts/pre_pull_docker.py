"""
Pre-pull Docker images required for SWE-Bench datasets

This script will:
1. Load data from the specified dataset registry
2. Extract all unique Docker image names
3. Pre-pull these images using the docker pull command

Usage:
    python pre_pull_docker.py --registry_dataset_name SWE_Bench_Verified --split test --max_workers 2
    python pre_pull_docker.py --registry_dataset_name multi_swe_bench_flash_subset_32 --split test --max_workers 4
    python pre_pull_docker.py --registry_dataset_name R2E_Gym_Subset --split train --max_workers 1
"""

import argparse
import logging
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from rllm.data.dataset import DatasetRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_docker_images(dataset_name: str, split: str = "test") -> list[str]:
    """Extract all unique Docker image names from the dataset
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split (default is "test")
        
    Returns:
        List of unique Docker image names
    """
    logger.info(f"Loading dataset: {dataset_name}, split: {split}")
    
    if not DatasetRegistry.dataset_exists(dataset_name, split):
        raise ValueError(f"Dataset '{dataset_name}' split '{split}' does not exist. Please run prepare_swe_data.py to create the dataset first.")
    
    dataset = DatasetRegistry.load_dataset(dataset_name, split)
    data = dataset.get_data()
    
    logger.info(f"Dataset contains {len(data)} task instances")
    
    # Extract Docker image names
    images = []
    missing_image_count = 0
    
    for entry in data:
        image_name = None
        if "docker_image" in entry:
            image_name = entry["docker_image"]
        elif "image_name" in entry:
            image_name = entry["image_name"]
        
        if image_name:
            images.append(image_name)
        else:
            missing_image_count += 1
    
    if missing_image_count > 0:
        logger.warning(f"{missing_image_count} task instances are missing Docker image information")
    
    # Get unique images and statistics
    unique_images = list(dict.fromkeys(images))
    image_counts = Counter(images)
    
    logger.info(f"Found {len(unique_images)} unique Docker images")
    logger.info("Image usage statistics:")
    for image, count in image_counts.most_common():
        logger.info(f"  {image}: {count} tasks")
    
    return unique_images


def get_local_images_set() -> set[str]:
    """Get a set of all local images in Repo:Tag format at once"""
    logger.info("Scanning local Docker image cache...")
    try:
        # Get list in repository:tag format
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            logger.warning("Unable to retrieve local image list, assuming local cache is empty")
            return set()

        local_images = set()
        for line in result.stdout.strip().splitlines():
            if line:
                # Add to set, e.g., {'python:3.9', 'ubuntu:latest'}
                local_images.add(line.strip())
        
        return local_images
    except Exception as e:
        logger.warning(f"Failed to scan local images: {e}")
        return set()


def pull_docker_image(image_name: str) -> tuple[str, bool, str]:
    """Execute pull operation only, no checks"""
    try:
        # logger.info(f"Starting download: {image_name}") # Optional: reduce log noise
        
        result = subprocess.run(
            ["docker", "pull", image_name],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minutes timeout
        )
        
        if result.returncode == 0:
            return (image_name, True, "Pull successful")
        else:
            error_msg = result.stderr.strip() or result.stdout.strip()
            # Even here, docker pull might find the image is up to date (rare concurrency case)
            if "Image is up to date" in error_msg:
                return (image_name, True, "Pull successful (Up to date)")
            return (image_name, False, error_msg)
    
    except subprocess.TimeoutExpired:
        return (image_name, False, "Pull timed out")
    except Exception as e:
        return (image_name, False, f"System error: {str(e)}")

def get_continuous_ranges(indices: list[int]) -> list[list[int]]:
    """将离散的索引列表转换为连续区间，例如 [0, 1, 2, 5, 6] -> [[0, 2], [5, 6]]"""
    if not indices:
        return []
    
    indices = sorted(indices)
    ranges = []
    start = indices[0]
    
    for i in range(1, len(indices)):
        # 如果当前数字不是前一个数字 + 1，说明区间断开了
        if indices[i] != indices[i-1] + 1:
            ranges.append([start, indices[i-1]])
            start = indices[i]
    
    # 添加最后一个区间
    ranges.append([start, indices[-1]])
    return ranges


def pull_all_images(images: list[str], max_workers: int = 1) -> dict:
    # 1. Check if Docker daemon is alive
    try:
        subprocess.run(["docker", "version"], capture_output=True, check=True)
    except Exception:
        raise RuntimeError("Cannot connect to Docker, please check if Docker is running")

    results = {
        "success": [],
        "failed": [],
        "already_exists": []
    }

    # 2. Get local image cache
    local_images_cache = get_local_images_set()
    
    results = {
        "success": [],
        "failed": [],
        "already_exists_indices": []  # 记录原始索引
    }

    # 3. 筛选需要下载的镜像，并记录已存在的索引
    images_to_pull = []
    
    for idx, image in enumerate(images):
        check_name = image if ":" in image else f"{image}:latest"
        
        if check_name in local_images_cache:
            results["already_exists_indices"].append(idx)
        else:
            # 记录需要拉取的镜像及其在原列表中的索引，方便后续统计
            images_to_pull.append((idx, image))

    # 4. 计算连续区间
    existing_ranges = get_continuous_ranges(results["already_exists_indices"])
    
    logger.info(f"Total tasks: {len(images)}")
    logger.info(f"Already exists (indices): {existing_ranges}")
    logger.info(f"Needs download: {len(images_to_pull)}")

    if not images_to_pull:
        logger.info("All images already exist!")
        return results
    # images_to_pull = images_to_pull[-1500:]
    # images_to_pull.reverse()
    # 5. 并行下载 (注意这里 images_to_pull 变成了元组列表)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_image = {
            executor.submit(pull_docker_image, img): img 
            for _, img in images_to_pull
        }
        
        # Progress bar only shows the number needed to pull for better visibility
        with tqdm(total=len(images_to_pull), desc="Pull Progress") as pbar:
            for future in as_completed(future_to_image):
                image_name, success, message = future.result()
                
                if success:
                    results["success"].append(image_name)
                    # Only log when an actual pull occurred to avoid spamming
                    logger.info(f"✓ {image_name}: {message}")
                else:
                    results["failed"].append((image_name, message))
                    logger.error(f"✗ {image_name}: {message}")
                
                pbar.update(1)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Pre-pull Docker images required for SWE-Bench datasets"
    )
    parser.add_argument(
        "--registry_dataset_name",
        type=str,
        default="SWE_Bench_Verified",
        help="Dataset name (default: SWE_Bench_Verified)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=2,
        help="Maximum parallel pull threads (default: 2, recommended <= 4 to avoid Docker daemon overload)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List required images only, do not actually pull"
    )
    
    args = parser.parse_args()
    
    # Extract image list
    try:
        images = extract_docker_images(args.registry_dataset_name, args.split)
    except Exception as e:
        logger.error(f"Failed to extract image list: {e}")
        return 1
    
    if not images:
        logger.warning("No Docker images found")
        return 0
    
    # If dry-run mode, only list images
    if args.dry_run:
        logger.info("Dry-run mode: Listing required images only")
        for i, image in enumerate(images, 1):
            print(f"{i}. {image}")
        return 0
    
    # Pull images
    try:
        results = pull_all_images(images, max_workers=args.max_workers)
    except Exception as e:
        logger.error(f"Failed to pull images: {e}")
        return 1
    
    # Output summary
    logger.info("\n" + "=" * 60)
    logger.info("Image pull completed!")
    logger.info(f"Total: {len(images)} images")
    logger.info(f"Newly pulled: {len(results['success'])}")
    logger.info(f"Already exists: {len(results['already_exists'])}")
    logger.info(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        logger.warning("\nFailed images:")
        for image, error in results["failed"]:
            logger.warning(f"  - {image}: {error}")
        return 1
    
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())