import fastdup
import os
import shutil
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd


def setup_logging(logs_path, log_level=logging.INFO):
    """
    Setup logging configuration with both file and console handlers.
    
    Args:
        logs_path (str): Path to the logs directory
        log_level: Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(logs_path, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('fastdup_deduplication')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # File handler for detailed logs (append mode)
    log_file_path = os.path.join(logs_path, 'fastdup_deduplication.log')
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_relative_path(file_path, base_dir):
    """
    Get relative path from base directory, handling both absolute and relative paths.
    
    Args:
        file_path (str): Full path to file
        base_dir (str): Base directory path
        
    Returns:
        str: Relative path from base_dir
    """
    return os.path.relpath(file_path, base_dir)


def find_all_images(input_dir):
    """
    Recursively find all image files in directory and subdirectories.
    
    Args:
        input_dir (str): Directory to search
        
    Returns:
        list: List of relative paths to image files
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    image_files = []
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                full_path = os.path.join(root, file)
                rel_path = get_relative_path(full_path, input_dir)
                image_files.append(rel_path)
    
    return image_files


def clean_image_duplicates(input_dir, work_dir="./fastdup_work",
                           log_file="cleaning_log.json", threshold=0.9):
    """
    Clean image dataset by finding duplicates with fastdup and moving them to a separate folder.
    Handles nested folder structures recursively.
    
    Args:
        input_dir (str): Path to input directory containing images (can have subdirectories)
        work_dir (str): Working directory for fastdup intermediate files (auto-deleted after run)
        log_file (str): Name of the JSON log file
        threshold (float): Distance threshold for duplicate detection (lower = stricter)
    """
    # Create duplicates folder inside input directory
    duplicates_path = os.path.join(input_dir, "duplicates")
    logs_path = os.path.join(duplicates_path, "logs")
    os.makedirs(duplicates_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(logs_path)
    
    logger.info("Starting image deduplication with fastdup...")
    logger.debug(f"Input directory: {input_dir}")
    logger.debug(f"Duplicates directory: {duplicates_path}")
    logger.debug(f"Work directory: {work_dir}")
    logger.debug(f"Threshold: {threshold}")
    
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.debug(f"Cleaned existing work directory: {work_dir}")

    # Initialize logging data
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "input_directory": input_dir,
        "work_directory": work_dir,
        "duplicates_directory": duplicates_path,
        "logs_directory": logs_path,
        "operation": "fastdup_image_deduplication",
        "clusters": [],
        "summary": {
            "total_images_processed": 0,
            "total_duplicate_pairs": 0,
            "images_kept": 0,
            "images_moved_to_duplicates": 0,
            "errors": []
        }
    }

    try:
        # Step 1: Run fastdup with recursive processing
        logger.info(f"Running fastdup.run on: {input_dir}")
        ret = fastdup.run(input_dir=input_dir, work_dir=work_dir, threshold=threshold)
        if ret != 0:
            raise Exception(f"fastdup.run failed with return code: {ret}")
        logger.info("fastdup analysis completed successfully")

        # Step 2: Read similarity results
        similarity_file = os.path.join(work_dir, 'similarity.csv')
        if not os.path.exists(similarity_file):
            raise Exception(f"Similarity file not found: {similarity_file}")

        similarity_df = pd.read_csv(similarity_file)
        if len(similarity_df) == 0:
            logger.info("No duplicate pairs found")
            all_files = find_all_images(input_dir)
            
            log_data["summary"]["total_images_processed"] = len(all_files)
            log_data["summary"]["images_kept"] = len(all_files)
            logger.info(f"All {len(all_files)} images are unique")
        else:
            logger.info(f"Found {len(similarity_df)} duplicate pairs")
            log_data["summary"]["total_duplicate_pairs"] = len(similarity_df)

            # Step 3: Build clusters using full relative paths
            clusters = build_image_clusters(similarity_df, input_dir)
            logger.info(f"Found {len(clusters)} clusters of similar images")

            total_moved, total_kept = 0, 0
            for cluster_id, cluster_images in clusters.items():
                logger.info(f"Processing cluster {cluster_id} with {len(cluster_images)} images")
                logger.debug(f"Cluster {cluster_id} images: {list(cluster_images)}")
                
                images_with_time = []
                for rel_path in cluster_images:
                    full_path = os.path.join(input_dir, rel_path)
                    if os.path.exists(full_path):
                        images_with_time.append((rel_path, os.path.getctime(full_path)))
                    else:
                        logger.warning(f"Image not found: {rel_path}")
                        
                images_with_time.sort(key=lambda x: x[1])

                if not images_with_time:
                    logger.warning(f"No valid images found in cluster {cluster_id}")
                    continue

                keeper_path = images_with_time[0][0]
                duplicates = [path for path, _ in images_with_time[1:]]

                cluster_log = {
                    "cluster_id": cluster_id,
                    "keeper_image": keeper_path,
                    "duplicate_images": [],
                    "cluster_size": len(cluster_images)
                }

                logger.info(f"  Keeping: {keeper_path}")
                logger.debug(f"  Keeper creation time: {datetime.fromtimestamp(images_with_time[0][1])}")
                total_kept += 1

                for duplicate_rel_path in duplicates:
                    try:
                        src_path = os.path.join(input_dir, duplicate_rel_path)
                        if not os.path.exists(src_path):
                            logger.warning(f"Source file not found, skipping: {duplicate_rel_path}")
                            continue
                        
                        # Preserve directory structure in duplicates folder
                        dst_rel_path = duplicate_rel_path
                        dst_path = os.path.join(duplicates_path, dst_rel_path)
                        
                        # Create directory structure if needed
                        dst_dir = os.path.dirname(dst_path)
                        if dst_dir:
                            os.makedirs(dst_dir, exist_ok=True)
                        
                        # Handle filename conflicts
                        if os.path.exists(dst_path):
                            base_name, ext = os.path.splitext(dst_path)
                            counter = 1
                            while os.path.exists(dst_path):
                                dst_path = f"{base_name}_{counter}{ext}"
                                counter += 1
                            dst_rel_path = get_relative_path(dst_path, duplicates_path)
                            
                        shutil.move(src_path, dst_path)
                        logger.info(f"  Moved: {duplicate_rel_path}")
                        logger.debug(f"  Moved to: {dst_path}")
                        
                        cluster_log["duplicate_images"].append({
                            "original_path": duplicate_rel_path,
                            "moved_to": dst_rel_path,
                            "moved_successfully": True
                        })
                        total_moved += 1
                        
                    except Exception as e:
                        err = f"Failed to move {duplicate_rel_path}: {e}"
                        logger.error(err)
                        log_data["summary"]["errors"].append(err)
                        cluster_log["duplicate_images"].append({
                            "original_path": duplicate_rel_path,
                            "moved_successfully": False,
                            "error": str(e)
                        })

                log_data["clusters"].append(cluster_log)

            log_data["summary"].update({
                "images_kept": total_kept,
                "images_moved_to_duplicates": total_moved,
                "total_images_processed": total_kept + total_moved
            })

        # Save logs in logs folder
        save_logs(log_data, logs_path, log_file, logger)

    except Exception as e:
        err = f"Error during processing: {e}"
        logger.error(err)
        log_data["summary"]["errors"].append(err)
        save_logs(log_data, logs_path, log_file, logger)
        return None
    finally:
        # Always delete work dir after run
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info(f"Temporary work directory {work_dir} deleted")

    return log_data


def build_image_clusters(similarity_df, input_dir):
    """
    Build clusters of similar images from similarity dataframe.
    Uses full relative paths to handle nested directories properly.
    
    Args:
        similarity_df: DataFrame containing similarity pairs
        input_dir: Base input directory for resolving relative paths
        
    Returns:
        dict: Dictionary mapping cluster_id to set of relative image paths
    """
    logger = logging.getLogger('fastdup_deduplication')
    
    image_to_cluster, clusters, next_cluster_id = {}, {}, 0
    logger.debug(f"Building clusters from {len(similarity_df)} similarity pairs")
    
    for idx, row in similarity_df.iterrows():
        # Get full paths from fastdup results
        img1_full = str(row.get('from', row.iloc[0]))
        img2_full = str(row.get('to', row.iloc[1]))
        
        # Convert to relative paths
        try:
            img1_rel = get_relative_path(img1_full, input_dir)
            img2_rel = get_relative_path(img2_full, input_dir)
        except ValueError:
            # If paths are already relative or there's an issue, use as-is
            img1_rel = img1_full
            img2_rel = img2_full
        
        cluster1, cluster2 = image_to_cluster.get(img1_rel), image_to_cluster.get(img2_rel)
        
        if cluster1 is None and cluster2 is None:
            # Create new cluster
            clusters[next_cluster_id] = {img1_rel, img2_rel}
            image_to_cluster[img1_rel] = image_to_cluster[img2_rel] = next_cluster_id
            logger.debug(f"Created new cluster {next_cluster_id} with images: {img1_rel}, {img2_rel}")
            next_cluster_id += 1
        elif cluster1 is None:
            # Add img1 to existing cluster2
            clusters[cluster2].add(img1_rel)
            image_to_cluster[img1_rel] = cluster2
            logger.debug(f"Added {img1_rel} to existing cluster {cluster2}")
        elif cluster2 is None:
            # Add img2 to existing cluster1
            clusters[cluster1].add(img2_rel)
            image_to_cluster[img2_rel] = cluster1
            logger.debug(f"Added {img2_rel} to existing cluster {cluster1}")
        elif cluster1 != cluster2:
            # Merge two clusters
            logger.debug(f"Merging cluster {cluster2} into cluster {cluster1}")
            clusters[cluster1] |= clusters[cluster2]
            for img in clusters[cluster2]:
                image_to_cluster[img] = cluster1
            del clusters[cluster2]
            
    logger.debug(f"Final cluster count: {len(clusters)}")
    return clusters


def save_logs(log_data, logs_path, log_file, logger):
    """
    Save detailed logs and summary to files.
    
    Args:
        log_data: Dictionary containing all log information
        logs_path: Path to logs directory
        log_file: Name of JSON log file
        logger: Logger instance
    """
    try:
        # Save detailed JSON log
        log_path = os.path.join(logs_path, log_file)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed JSON log saved to: {log_path}")

        # Save human-readable summary
        summary_path = os.path.join(logs_path, "deduplication_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("FASTDUP IMAGE DEDUPLICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {log_data['timestamp']}\n")
            f.write(f"Input Directory: {log_data['input_directory']}\n")
            f.write(f"Duplicates Directory: {log_data['duplicates_directory']}\n\n")
            f.write("SUMMARY:\n")
            f.write(f"- Total images processed: {log_data['summary']['total_images_processed']}\n")
            f.write(f"- Total duplicate pairs found: {log_data['summary']['total_duplicate_pairs']}\n")
            f.write(f"- Number of clusters: {len(log_data['clusters'])}\n")
            f.write(f"- Images kept (originals): {log_data['summary']['images_kept']}\n")
            f.write(f"- Images moved to duplicates: {log_data['summary']['images_moved_to_duplicates']}\n")
            if log_data['summary']['errors']:
                f.write(f"- Errors encountered: {len(log_data['summary']['errors'])}\n\n")
                f.write("ERRORS:\n")
                for i, error in enumerate(log_data['summary']['errors'], 1):
                    f.write(f"{i}. {error}\n")
        logger.info(f"Summary saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to save logs: {e}")


def main():
    parser = argparse.ArgumentParser(description="Image deduplication with fastdup")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input directory containing images")
    parser.add_argument("--work_dir", type=str, default="./fastdup_work", help="Temporary work directory")
    parser.add_argument("--log_file", type=str, default="fastdup_cleaning_log.json", help="Log file name")
    parser.add_argument("--threshold", type=float, default=0.9, help="Distance threshold for duplicate detection")
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Convert string log level to logging constant
    log_level = getattr(logging, args.log_level.upper())

    log = clean_image_duplicates(
        input_dir=args.input_dir,
        work_dir=args.work_dir,
        log_file=args.log_file,
        threshold=args.threshold
    )

    # Get logger for final output
    logger = logging.getLogger('fastdup_deduplication')
    
    if log:
        logger.info("Deduplication completed successfully")
        logger.info(f"Processed {log['summary']['total_images_processed']} images")
        logger.info(f"Found {len(log['clusters'])} duplicate clusters")
        logger.info(f"Kept {log['summary']['images_kept']} originals")
        logger.info(f"Moved {log['summary']['images_moved_to_duplicates']} duplicates")
        
        if log['summary']['errors']:
            logger.warning(f"Encountered {len(log['summary']['errors'])} errors during processing")
    else:
        logger.error("Deduplication failed. See logs for details")


if __name__ == "__main__":
    main()
