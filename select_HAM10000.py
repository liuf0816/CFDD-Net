import os
import random
import shutil
from pathlib import Path

# -------------------------- Parameters--------------------------
# Raw Data Root
RAW_DATA_ROOT = r"E:\360Downloads\archive"
# Output Data Root
OUTPUT_DATA_ROOT = "./ham10000_dataset"
# Split Ratio
SPLIT_RATIO = (0.6, 0.2, 0.2)
# Random Seed (for reproduction purporse)
RANDOM_SEED = 42
# Suffix Identifiers for Original Images and Masks (to match the renamed format)
IMG_SUFFIX = "_data.jpg"  # original image suffix, e.g.：0024306_data.jpg
MASK_SUFFIX = "_mask.png"  # mask suffix, e.g：0024306_mask.png


# -------------------------- Core functions --------------------------

def check_and_get_valid_samples(images_dir, masks_dir, img_suffix, mask_suffix):
    """
    Check the correspondence between the original images and masks, and return a list of valid sample IDs (only retain samples where both the original image and mask exist)

    Args:
        images_dir (str): Original image folder path
        masks_dir (str): Mask folder path
        img_suffix (str): Oringal image suffix
        mask_suffix (str): Mask suffix

    Returns:
        list: List of valid sample IDs（如 ["0024306", "0024307"]）
    """
    valid_samples = []

    # Go through all original image
    for img_filename in os.listdir(images_dir):
        # Filter original image files in non target formats
        if not img_filename.endswith(img_suffix):
            continue

        # Extract sample ID
        sample_id = img_filename.replace(img_suffix, "")
        # Construct the corresponding mask file name
        mask_filename = f"{sample_id}{mask_suffix}"
        mask_path = os.path.join(masks_dir, mask_filename)

        # Check if the mask file exists
        if os.path.exists(mask_path):
            valid_samples.append(sample_id)
        else:
            print(f"⚠️  Sample {sample_id} Missing mask file, skip")

    print(f"✅ Total number of valid samples found：{len(valid_samples)}（The original image and mask both exist）")
    return valid_samples


def split_dataset(samples, split_ratio, random_seed):
    """
    Randomly divide the training/validation/testing set proportionally

    Args:
        samples (list): List of valid sample IDs
        split_ratio (tuple): split_ratio (train, valid, test)
        random_seed (int): random_seed

    Returns:
        tuple: (train_samples, val_samples, test_samples)
    """
    # Fixed random seed
    random.seed(random_seed)
    # Disrupt the order of samples
    random.shuffle(samples)

    # Calculate the quantity of each dataset
    total = len(samples)
    train_num = int(total * split_ratio[0])
    val_num = int(total * split_ratio[1])
    # test sets = total - training set - validation set（Avoid floating point errors）
    test_num = total - train_num - val_num

    # split the dataset
    train_samples = samples[:train_num]
    val_samples = samples[train_num:train_num + val_num]
    test_samples = samples[train_num + val_num:]

    print(f"\n📊 Dataset partitioning results：")
    print(f"   training set：{len(train_samples)} samples")
    print(f"   validation set：{len(val_samples)} samples")
    print(f"   testing set：{len(test_samples)} samples")

    return train_samples, val_samples, test_samples


def copy_samples_to_target(samples, images_dir, masks_dir, target_dir, img_suffix, mask_suffix):
    """
    Copy the original image and mask of the sample to the target folder

    Args:
        samples (list): List of sample IDs to be copied
        images_dir (str): Original image source path
        masks_dir (str): Mask source path
        target_dir (str): Target folder path
        img_suffix (str): Original image suffix
        mask_suffix (str): Mask suffix
    """
    # Create target folder (recursive creation)
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    copied_count = 0
    error_count = 0

    for sample_id in samples:
        # Construct source file path
        img_src = os.path.join(images_dir, f"{sample_id}{img_suffix}")
        mask_src = os.path.join(masks_dir, f"{sample_id}{mask_suffix}")

        # Construct target file path
        img_dst = os.path.join(target_dir, f"{sample_id}{img_suffix}")
        mask_dst = os.path.join(target_dir, f"{sample_id}{mask_suffix}")

        try:
            # Copy the original image and mask
            shutil.copy2(img_src, img_dst)  # copy2 retain file metadata
            shutil.copy2(mask_src, mask_dst)
            copied_count += 1
        except Exception as e:
            print(f"❌ Copy sample {sample_id} failed：{str(e)}")
            error_count += 1

    print(f"✅ Folder {target_dir} copy completed：{copied_count} sampled succeeded，{error_count} samples failed")


if __name__ == "__main__":
    # 1. Define source folder path
    images_dir = os.path.join(RAW_DATA_ROOT, "imagesnew")
    masks_dir = os.path.join(RAW_DATA_ROOT, "masksnew")

    # Check if the source folder exists
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"❌ The original image folder does not exist：{images_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"❌ Mask folder does not exist：{masks_dir}")

    # 2. Obtain valid samples (both original image and mask present)
    print("🔍 Check the correspondence between the original image and the mask ...")
    valid_samples = check_and_get_valid_samples(images_dir, masks_dir, IMG_SUFFIX, MASK_SUFFIX)

    if not valid_samples:
        raise ValueError("❌ No valid samples were found (both original image and mask exist), please check the file path and naming format")

    # 3. split the dataset
    print("\n🔀 Divide the dataset proportionally...")
    train_samples, val_samples, test_samples = split_dataset(valid_samples, SPLIT_RATIO, RANDOM_SEED)

    # 4. Copy files to the target folder
    print("\n📤 Start copying files to the target directory...")
    # Training set
    copy_samples_to_target(train_samples, images_dir, masks_dir,
                           os.path.join(OUTPUT_DATA_ROOT, "train"),
                           IMG_SUFFIX, MASK_SUFFIX)
    # Validation set
    copy_samples_to_target(val_samples, images_dir, masks_dir,
                           os.path.join(OUTPUT_DATA_ROOT, "valid"),
                           IMG_SUFFIX, MASK_SUFFIX)
    # Testing set
    copy_samples_to_target(test_samples, images_dir, masks_dir,
                           os.path.join(OUTPUT_DATA_ROOT, "test"),
                           IMG_SUFFIX, MASK_SUFFIX)

    print("\n🎉 Dataset partitioning completed!")
    print(f"📁 Path of partitioned dataset：{OUTPUT_DATA_ROOT}")
    print(f"📂 folder structure：")
    print(f"   {OUTPUT_DATA_ROOT}/train （Training set, including original image and mask）")
    print(f"   {OUTPUT_DATA_ROOT}/valid   （Verification set, including original image and mask）")
    print(f"   {OUTPUT_DATA_ROOT}/test  （Test set, including original image and mask）")