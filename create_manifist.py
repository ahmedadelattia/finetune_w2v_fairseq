import sys
import soundfile as sf
import glob
import os,tqdm
import json
from whisper_normalizer.english import EnglishTextNormalizer
# p2root = sys.argv[1] if len(sys.argv) > 1 else "../../Data/NCTE/NCTE_Noise_turn_based"
# ext = sys.argv[2] if len(sys.argv) > 2 else "wav"

dataset = sys.argv[1] if len(sys.argv) > 1 else "NCTE"
manifest = "./manifest/"

if dataset == "NCTE":
    p2root = "/media/ahmed/DATA 1/Research/Data/NCTE/NCTE_Noise_turn_based"
    ext = ".wav"
    manifest += "NCTE/"    
elif dataset == "Fall":
    p2root = "/media/ahmed/DATA 1/Research/Data/Fall Pilot/Noisy Audio Chunks/"
    ext = ".wav"
    manifest += "Fall/"
    
elif dataset == "NCTE_Full" or dataset == "NCTE_Full_all_mics":
    p2root = "/media/ahmed/DATA 1/Research/Data/NCTE - Consolidated/"
    ext = ".wav"
    if dataset == "NCTE_Full":
        manifest += "NCTE_Full/"
    else:
        manifest += "NCTE_Full_all_mics/"
        p2root += "All_Mics/"
elif dataset == "NCTE_2000hr":
    p2root = "/media/ahmed/DATA 1/Research/Data/NCTE_2000hr/"
    ext = ".wav"
    manifest += "NCTE_2000hr/"
elif dataset == "Librispeech_Noise":
    p2root = "/media/ahmed/DATA 2/Research/Data/Librispeech/Noisy"
    ext = ".flac"
    manifest += "Librispeech_Noise/"
elif dataset == "Librispeech":
    p2root = "/media/ahmed/DATA 2/Research/Data/Librispeech/"
    ext = ".flac"
    manifest += "Librispeech/"
       
    
normalize = EnglishTextNormalizer()
os.makedirs(manifest,exist_ok=True)

charset = set()


def parse_folder(folder,search_pattern = None, p2root = p2root, ext = ext, audio_dir = "Audio", max_duration = 1000000000000000000000):
    wavs = []
    for r, d, f in os.walk(search_pattern):
        f = sorted(f)
        for file in f:
            if file.endswith(ext):
                wavs.append(os.path.join(r, file))
                
 

    sr = sf.info(wavs[0]).samplerate
    
    # samples = [len(sf.read(w)[0]) for w in tqdm.tqdm(wavs) if sum(samples)/sr < max_duration*60]
    samples = []
    for n, w in tqdm.tqdm(enumerate(wavs)):
        s = len(sf.read(w)[0])
        samples.append(s)
        if sum(samples)/sr > max_duration*60:
            break
    wavs = wavs[:n+1]
    assert len(wavs) == len(samples), (len(wavs), len(samples))
    print("Found",len(samples),"samples in folder",folder)
    
    wavs = [os.path.realpath(w) for w in wavs]
    wav2trans = dict()
    
    with open(os.path.join(p2root,"Transcripts",folder+".json")) as transcrip:
        lines = transcrip.read().strip().split('\n')
    print("Found",len(lines),"transcripts in folder",folder)
    for line in (lines):
        line = json.loads(line)

        if os.path.realpath(line["audio_path"]) not in wavs:
            print("Skipping",line["audio_path"])
            # print(os.path.realpath(line["audio_path"]));exit()
            continue
        file, trans = line['audio_path'], normalize(line['text'])
        file = "/".join(file.split("/")[-2:]).replace(ext,"")
        # file = os.path.join(subdir,file)
        if file in wav2trans.keys():
            raise ValueError(f"Duplicate file {file} in {folder}")
        wav2trans[file] = trans
        charset.update(trans.replace(" ","|"))
    
    assert len(wavs) == len(wav2trans), (len(wavs), len(wav2trans))
    return wavs, samples, wav2trans, charset
        
def write_manifest(wavs, samples, wav2trans, manifest,split_name, ext = ext):
    os.makedirs(manifest,exist_ok=True)
    
    #extract root which is the common path for all files
    root = os.path.commonpath(wavs)
    wavs_rel = [os.path.relpath(w,root) for w in wavs]
    with open(os.path.join(manifest,split_name+".tsv"),'w') as tsv, \
        open(os.path.join(manifest,split_name+".wrd"),'w') as wrd, \
        open(os.path.join(manifest,split_name+".ltr"),'w') as ltr:
        root = os.path.abspath(root)
        print(root,file=tsv)
        for n_rel, n,d in zip(wavs_rel, wavs,samples):
            print(n_rel,d,sep='\t',file=tsv)
            print(wav2trans["/".join(n.split("/")[-2:]).replace(ext,"")],file=wrd)
            print(" ".join(list(wav2trans["/".join(n.split("/")[-2:]).replace(ext,"")].replace(" ", "|"))) + " |", file=ltr)


if __name__ == "__main__":
    if dataset == "NCTE_Full" or dataset == "NCTE_Full_all_mics":
        dataset_lens = []
        #create train valid splits for x-validation. For each split, create a manifest subfolder

        
        files = [f for f in os.listdir(os.path.join(p2root, "Audio"))]

        validation_files = ["2535", "2684", "2757", "4191"]
        
        validation_files = [f for f in files if f.split("_")[0] in validation_files]
        print("Validation files",validation_files)
        train_files      = [f for f in files if f not in validation_files]
        # train_files = [f for f in train_files if f.split("_")[0] in ["144", "622", "2619", "2709", "2944", "4724"]]
        train_wavs, train_samples, train_root, train_wav2trans = [], [], [], dict()
        valid_wavs, valid_samples, valid_root, valid_wav2trans = [], [], [], dict() 
               
        #no cross validation in full dataset
        for valid_folder in tqdm.tqdm(validation_files):
            search_pattern = os.path.join(p2root, "Audio", valid_folder)
            wavs, samples, wav2trans, charset = parse_folder(valid_folder, search_pattern=search_pattern)
            write_manifest(wavs, samples, wav2trans, f"{manifest}", f"valid_{valid_folder.split('_')[0]}")
            valid_wavs.extend(wavs); valid_samples.extend(samples);  valid_wav2trans.update(wav2trans)
        write_manifest(valid_wavs, valid_samples, valid_wav2trans, f"{manifest}", "valid")
        valid_len = len(valid_wavs)
        for train_folder in tqdm.tqdm(train_files):
            search_pattern = os.path.join(p2root, "Audio", train_folder)
            wavs, samples, wav2trans, charset = parse_folder(train_folder, search_pattern=search_pattern)
            write_manifest(wavs, samples, wav2trans, f"{manifest}", f"valid_{train_folder.split('_')[0]}")
            train_wavs.extend(wavs); train_samples.extend(samples);  train_wav2trans.update(wav2trans)

        write_manifest(train_wavs, train_samples, train_wav2trans, f"{manifest}", "train")
        
        charset = sorted(list(charset))
        with open(os.path.join(manifest,"dict.ltr.txt"),'w') as dct:
            for e,c in enumerate(charset):
                print(c,e,file=dct)
        
    elif dataset == "NCTE":
        dataset_lens = []
        #create train valid splits for x-validation. For each split, create a manifest subfolder
        train_files = ["144", "622", "2619", "2709", "2944", "4724"]
        for valid_folder in tqdm.tqdm(train_files):
            if valid_folder == "Transcripts":
                continue
            
            wavs, samples, root, wav2trans, charset = parse_folder(valid_folder)
            write_manifest(wavs, samples, root, wav2trans, f"{manifest}/fold_{valid_folder}", "valid")
            train_wavs, train_samples, train_root, train_wav2trans = [], [], [], dict()
            valid_len = len(wavs)
            for folder in tqdm.tqdm(train_files):
                if folder == "Transcripts":
                    continue
                if folder == valid_folder:
                    continue
                wavs, samples, root, wav2trans, charset = parse_folder(folder)
                train_wavs.extend(wavs); train_samples.extend(samples); train_root.append(root); train_wav2trans.update(wav2trans)
            for root in train_root:
                assert root == train_root[0]
            train_root = train_root[0]    
            write_manifest(train_wavs, train_samples, train_root, train_wav2trans, f"{manifest}/fold_{valid_folder}", "train")
            
            train_len = len(train_wavs)
            dataset_lens.append((train_len, valid_len, train_len+valid_len))
        for train_len, valid_len, total_len in dataset_lens:
            assert train_len + valid_len == total_len
        #sort the charset and write it to a file
        charset = sorted(list(charset))
        for fold in os.listdir(manifest):
            with open(os.path.join(manifest,fold,"dict.ltr.txt"),'w') as dct:
                for e,c in enumerate(charset):
                    print(c,e,file=dct)
                    
        test_files = os.listdir(os.path.join(p2root, "Audio"))
        test_files = [f for f in test_files if f not in train_files]
        charset = set()
        
        for folder in tqdm.tqdm(test_files):
            if folder == "Transcripts":
                continue
            wavs, samples, root, wav2trans, charset = parse_folder(folder)
            write_manifest(wavs, samples, root, wav2trans, f"{manifest}/fold_{folder}", "test")
            
    
    elif dataset == "NCTE_2000hr":
        dataset_lens = []
        #list of files to remove to avoid overlap between different datasets
        files_to_remove = os.listdir("/home/ahmed/Research_Data_1/Data/NCTE - Consolidated/Audio_alt")
        #set seed for reproducibility
        import numpy as np
        np.random.seed(0)
        files = os.listdir(os.path.join(p2root, "Audio"))
        #remove files that are in the files_to_remove list
        files = [f for f in files if f not in files_to_remove]
        #create train, valid, test splits
        train_files = np.random.choice(files, size = int(0.8*len(files)), replace = False)
        validation_files = np.random.choice([f for f in files if f not in train_files], size = int(0.1*len(files)), replace = False)
        test_files = [f for f in files if f not in train_files and f not in validation_files]
        
        train_wavs, train_samples, train_root, train_wav2trans = [], [], [], dict()
        valid_wavs, valid_samples, valid_root, valid_wav2trans = [], [], [], dict()        
        test_wavs, test_samples, test_root, test_wav2trans = [], [], [], dict() 
        charset = set()
        #no cross validation in full dataset
        for valid_folder in tqdm.tqdm(validation_files):
            search_pattern = os.path.join(p2root, "Audio", valid_folder)
            wavs, samples, wav2trans, charset_valid = parse_folder(valid_folder, search_pattern=search_pattern)
            valid_wavs.extend(wavs); valid_samples.extend(samples);  valid_wav2trans.update(wav2trans)
            charset.update(charset_valid)

        write_manifest(valid_wavs, valid_samples, valid_wav2trans, f"{manifest}", "valid")
        valid_len = len(valid_wavs)
        
        for test_folder in tqdm.tqdm(test_files):
            search_pattern = os.path.join(p2root, "Audio", test_folder)
            wavs, samples, wav2trans, charset_test = parse_folder(test_folder, search_pattern=search_pattern)
            test_wavs.extend(wavs); test_samples.extend(samples); test_wav2trans.update(wav2trans)
            charset.update(charset_test)
        write_manifest(test_wavs, test_samples, test_wav2trans, f"{manifest}", "valid")
        test_len = len(test_wavs)
       
        for train_folder in tqdm.tqdm(train_files):
            search_pattern = os.path.join(p2root, "Audio", train_folder)
            wavs, samples, wav2trans, charset_train = parse_folder(train_folder, search_pattern=search_pattern)
            train_wavs.extend(wavs); train_samples.extend(samples); train_wav2trans.update(wav2trans)
            charset.update(charset_train)
        
        write_manifest(train_wavs, train_samples, train_wav2trans, f"{manifest}", "train")
        
        charset = sorted(list(charset))
        with open(os.path.join(manifest,"dict.ltr.txt"),'w') as dct:
            for e,c in enumerate(charset):
                print(c,e,file=dct)
                
    elif dataset == "Librispeech_Noise":
        snrs = ["-5", "0", "5", "10", "15", "20"]
        train_files = ["train-clean-100", "train-clean-360", "train-other-500"]
        valid_files = ["dev-clean"]
        test_files = ["test-clean"]
        
        train_files = [f"{snr}/{f}" for snr in snrs for f in train_files]
        valid_files = [f"{snr}/{f}" for snr in snrs for f in valid_files]
        test_files = [f"{snr}/{f}" for snr in snrs for f in test_files]
        # train_wavs, train_samples, train_root, train_wav2trans = [], [], [], dict()
        # valid_wavs, valid_samples, valid_root, valid_wav2trans = [], [], [], dict()
        # for folder in train_files:
        #     #audio files exist in this structure
        #     #{p2root}/{snr}/train-clean-100/LibriSpeech/train-clean-100/
        #     search_pattern = os.path.join(p2root, folder, "LibriSpeech", folder.split("/")[-1])
        #     wavs, samples, wav2trans, charset = parse_folder(folder, search_pattern= search_pattern)
        #     train_wavs.extend(wavs); train_samples.extend(samples); train_wav2trans.update(wav2trans)
        # write_manifest(train_wavs, train_samples, train_wav2trans, f"{manifest}", "train")
        
        # for folder in valid_files:
        #     search_pattern = os.path.join(p2root, folder, "LibriSpeech", folder.split("/")[-1])
        #     wavs, samples, wav2trans, charset = parse_folder(folder, search_pattern=search_pattern)
        #     valid_wavs.extend(wavs); valid_samples.extend(samples);  valid_wav2trans.update(wav2trans)
            
        # write_manifest(valid_wavs, valid_samples, valid_wav2trans, f"{manifest}", "valid")
        
        # charset = sorted(list(charset))
        # with open(os.path.join(manifest,"dict.ltr.txt"),'w') as dct:
        #     for e,c in enumerate(charset):
        #         print(c,e,file=dct)
       
        for folder in test_files:
            search_pattern = os.path.join(p2root, folder)
            wavs, samples, wav2trans, charset = parse_folder(folder, search_pattern=search_pattern)
            
            write_manifest(wavs, samples, wav2trans, f"{manifest}/test", f"test_snr_{folder.split('/')[0]}")

    elif dataset == "Librispeech":
        train_files = ["train-clean-100", "train-clean-360", "train-other-500"]
        valid_files = ["dev-clean"]
        
        train_wavs, train_samples, train_root, train_wav2trans = [], [], [], dict()
        valid_wavs, valid_samples, valid_root, valid_wav2trans = [], [], [], dict()
        for folder in train_files:
            #audio files exist in this structure
            #{p2root}/{snr}/train-clean-100/LibriSpeech/train-clean-100/
            search_pattern = os.path.join(p2root, folder, "LibriSpeech", folder.split("/")[-1])
            wavs, samples, wav2trans, charset = parse_folder(folder, search_pattern= search_pattern)
            train_wavs.extend(wavs); train_samples.extend(samples); train_wav2trans.update(wav2trans)
        write_manifest(train_wavs, train_samples, train_wav2trans, f"{manifest}", "train")
        
        for folder in valid_files:
            search_pattern = os.path.join(p2root, folder, "LibriSpeech", folder.split("/")[-1])
            wavs, samples, wav2trans, charset = parse_folder(folder, search_pattern=search_pattern)
            valid_wavs.extend(wavs); valid_samples.extend(samples);  valid_wav2trans.update(wav2trans)
            
        write_manifest(valid_wavs, valid_samples, valid_wav2trans, f"{manifest}", "valid")
        
        charset = sorted(list(charset))
        with open(os.path.join(manifest,"dict.ltr.txt"),'w') as dct:
            for e,c in enumerate(charset):
                print(c,e,file=dct)

                    
    
           
            