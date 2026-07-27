import os
import random

def get_all_files(root_folder):
    """
    Recursively collect all file paths under root_folder,
    skipping hidden directories and hidden files.

    Returns:
        List of file paths including the original root_folder prefix, e.g.
        ["data/Welle/kinematics/20210427/58/249.csv", ...]
    """
    file_paths = []

    for current_root, dirs, files in os.walk(root_folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.startswith("."):
                continue

            full_path = os.path.join(current_root, file)
            file_paths.append(full_path)

    return file_paths


def shuffle_and_split(file_list, train_ratio=0.8, seed=None):
    """
    Shuffle and split a list into train/test sets.

    Args:
        file_list (list): list of file paths
        train_ratio (float): fraction for training set (e.g. 0.8)
        seed (int, optional): for reproducibility

    Returns:
        train_list, test_list
    """
    if seed is not None:
        random.seed(seed)

    file_list = file_list.copy()  # don't modify original
    random.shuffle(file_list)

    split_idx = int(len(file_list) * train_ratio)

    train_list = file_list[:split_idx]
    test_list = file_list[split_idx:]

    return train_list, test_list


def setup_files(path, train_ratio=0.8, seed=42):
    if os.path.isfile(path):
        print("single kinematics")
        train_files = [path]
        test_files = [path]

    elif os.path.isdir(path):
        print("multiple kinematics")
        files = get_all_files(path)
        train_files, test_files = shuffle_and_split(files, train_ratio=train_ratio, seed=seed)

    else:
        raise Exception("kinematics folder/file DNE")
    
    return train_files, test_files