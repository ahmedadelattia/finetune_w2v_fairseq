#updates the header of the manifest file with the a certain path to accomodate different machines.
import os 
import glob
def update_manifest(manifest_tsv, target_header_root):
    with open(manifest_tsv, 'r') as f:
        lines = f.readlines()
    target_header = get_header(lines[0], target_header_root) 
    lines[0] = target_header + "\n"
    with open(manifest_tsv, 'w') as f:
        f.writelines(lines)
    return


def get_header(current_header, target_header_root):
    if "NCTE - Consolidated" in current_header:
        return os.path.join(target_header_root, "NCTE - Consolidated", "Audio")
    elif "NCTE" in current_header:
        return os.path.join(target_header_root, "NCTE", "Audio")
    elif "Fall" in current_header:
        return os.path.join(target_header_root, "Fall", "Audio")
    
    raise ValueError("Header not found in current header")

def get_tsv_files(dataset_manifest):
    #tsv files either in the root or in the subfolders
    
    tsv_files = glob.glob(os.path.join(dataset_manifest, "*.tsv"))
    subfolders = [f.path for f in os.scandir(dataset_manifest) if f.is_dir()]
    for subfolder in subfolders:
        tsv_files += glob.glob(os.path.join(subfolder, "*.tsv"))
    return tsv_files


def main():
    import sys
    manifest_root = "./manifest/"
    target_header_root = sys.argv[1] if len(sys.argv) > 1 else "/scr/aadel4/Data/"
    datasets = os.listdir(manifest_root)
    for dataset in datasets:
        dataset_manifest = os.path.join(manifest_root, dataset)
        assert os.path.exists(dataset_manifest), f"Dataset manifest {dataset_manifest} does not exist"
        tsv_files = get_tsv_files(dataset_manifest)
        for tsv in tsv_files:
            print(f"Updating {tsv} with header {target_header_root}")
            # update_manifest(tsv, target_header_root)
    return


if __name__ == "__main__":
    main()