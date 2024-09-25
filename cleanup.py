import os
import re

# Define the pattern for matching files
pattern = re.compile(r'checkpoint_\d+_\d+\.pt')

def find_checkpoints(directory):
    """
    Recursively finds all files in the directory tree that match the checkpoint pattern.
    """
    checkpoints = []
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                checkpoints.append(os.path.join(root, file))
    return checkpoints

def find_wandb_directories(directory):
    """
    Recursively finds all directories named 'wandb' in the directory tree.
    """
    wandb_dirs = []
    for root, dirs, _ in os.walk(directory):
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
            os.remove(file)
            print(f"Deleted: {file}")
    else:
        print("Deletion canceled.")

if __name__ == "__main__":
    # Specify the directory to search
    directory = input("Enter the directory path to search for checkpoint files: ").strip()
    
    checkpoint_files = find_checkpoints(directory)
    confirm_and_delete(checkpoint_files)


    wandb_directories = find_wandb_directories(directory)
    confirm_and_delete(wandb_directories)