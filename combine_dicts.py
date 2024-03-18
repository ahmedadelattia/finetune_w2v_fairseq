#combine dict files in manifest/Fall and manifest/NCTE
# Usage: python combine_dicts.py
# Output: combined dict files in manifest/ in place of the original files
import os

def open_dict(manifest_dir):
    dict_file = os.path.join(manifest_dir, 'dict.ltr.txt')
    if os.path.exists(dict_file):
        with open(dict_file, 'r') as f:
            dict_list = f.readlines()
    else:
        raise FileNotFoundError(f'{dict_file} not found')
    dict_out = {}
    for line in dict_list:
        line = line.strip()
        if line:
            word, token = line.split()
            dict_out[word] = token
    return dict_out

def update_dict(current_dict, new_dict):
    curr_tokens = set(current_dict.values())
    if len(curr_tokens) == 0:
        return new_dict
    last_token = max(curr_tokens)
    for word, token in new_dict.items():
        if word not in current_dict:
            last_token = str(int(last_token) + 1)
            current_dict[word] = last_token
    return current_dict


def write_dict(manifest_dir, dict_out):
    dict_file = os.path.join(manifest_dir, 'dict.ltr.txt')
    print(f'Writing to {dict_file}')
    with open(dict_file, 'w') as f:
        for word, token in dict_out.items():
            f.write(f'{word} {token}\n')
            
def main():
    manifest_dir = 'manifest'
    fall_manifest = os.path.join(manifest_dir, 'Fall')
    ncte_manifest = os.path.join(manifest_dir, 'NCTE')
    manfiests = [os.path.join(fall_manifest, fold) for fold in os.listdir(fall_manifest) if os.path.isdir(os.path.join(fall_manifest, fold))]  \
              + [os.path.join(ncte_manifest, fold) for fold in os.listdir(ncte_manifest) if os.path.isdir(os.path.join(ncte_manifest, fold))]
    combined_dict = {}
    for manifest in manfiests:
        print(f'Updating {manifest}')
        combined_dict = update_dict(combined_dict, open_dict(manifest))
    for manifest in manfiests:
        write_dict(manifest, combined_dict)
        
        
if __name__ == '__main__':
    main()