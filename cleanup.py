import os
import re
import tqdm

# Define the pattern for matching files
pattern = re.compile(r'checkpoint_\d+_\d+\.pt')

def find_checkpoints(directory):
    """
    Recursively finds all files in the directory tree that match the checkpoint pattern.
    """
    checkpoints = []
    pbar = tqdm.tqdm()

    for root, _, files in os.walk(directory):
        for file in files:
            pbar.set_description(f"Searching in {root}, file: {file}")
            pbar.update(1)
            if pattern.match(file):
                checkpoints.append(os.path.join(root, file))
    return checkpoints

def find_wandb_directories(directory):
    """
    Recursively finds all directories named 'wandb' in the directory tree.
    """
    wandb_dirs = []
    pbar = tqdm.tqdm()

    for root, dirs, _ in os.walk(directory):
        pbar.set_description(f"Searching in {root}, dirs: {dirs}")
        pbar.update(1)
        if 'wandb' in dirs:
            wandb_dirs.append(os.path.join(root, 'wandb'))
    return wandb_dirs

def confirm_and_delete(files):
    """
    Shows the list of files and asks for confirmation before deleting.
    """
    if not files:
        print("No matching checkpoint files found.")
        return
    
    print("The following files will be deleted:")
    for file in files:
        print(file)
    
    # Ask for user confirmation
    confirm = input("Are you sure you want to delete these files? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        for file in files:
            #file or directory
            if os.path.isdir(file):
                os.system(f"rm -r {file}")
            else:
                os.remove(file)
            print(f"Deleted: {file}")
    else:
        print("Deletion canceled.")

if __name__ == "__main__":
    # Specify the directory to search
    directory = "./model_outputs/continued_pretraining"
    print(f"Searching for checkpoint files in {directory}...")
    checkpoint_files = find_checkpoints(directory)
    print("Checkpoint files found:")
    confirm_and_delete(checkpoint_files)
    print(f"Searching for wandb directories in {directory}...")

    wandb_directories = find_wandb_directories(directory)
    print("Wandb directories found:")
    confirm_and_delete(wandb_directories)