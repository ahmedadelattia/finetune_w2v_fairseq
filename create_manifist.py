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
    p2root = "../../Data/NCTE/NCTE_Noise_turn_based"
    ext = ".wav"
    manifest += "NCTE/"    
elif dataset == "Fall":
    p2root = "../../Data/Fall Pilot/Noisy Audio Chunks/"
    ext = ".wav"
    manifest += "Fall/"
    
    
normalize = EnglishTextNormalizer()
os.makedirs(manifest,exist_ok=True)

charset = set()


def parse_folder(folder,search_pattern = None, p2root = p2root, ext = ext, max_duration = 1000000000000000000000):
    if search_pattern is None:
        search_pattern = os.path.join(p2root,"Audio", folder,'*'+ext)
    print(search_pattern)
    wavs = glob.glob(search_pattern, recursive=True)
    print(len(wavs))

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
    root = search_pattern.split("/*")[0]
    root, subdir = os.path.dirname(root), os.path.split(root)[-1]
    wavs = [w.split(root+"/")[-1] for w in wavs]

    wav2trans = dict()

    with open(os.path.join(p2root,"Transcripts",folder+".json")) as transcrip:
        lines = transcrip.read().strip().split('\n')
    for line in (lines):
        line = json.loads(line)
        
        if "/".join(line["audio_path"].split("/")[-2:]) not in wavs:
            continue
        file, trans = line['audio_path'], normalize(line['text'])
        file = "/".join(file.split("/")[-2:]).replace(ext,"")
        # file = os.path.join(subdir,file)
        wav2trans[file] = trans
        charset.update(trans.replace(" ","|"))
    
    assert len(wavs) == len(wav2trans), (len(wavs), len(wav2trans))
    return wavs, samples, root, wav2trans, charset
        
def write_manifest(wavs, samples, root, wav2trans, manifest,split_name, ext = ext):
    os.makedirs(manifest,exist_ok=True)
    with open(os.path.join(manifest,split_name+".tsv"),'w') as tsv, \
        open(os.path.join(manifest,split_name+".wrd"),'w') as wrd, \
        open(os.path.join(manifest,split_name+".ltr"),'w') as ltr:
        root = os.path.abspath(root)
        print(root,file=tsv)
        for n,d in zip(wavs,samples):
            print(n,d,sep='\t',file=tsv)
            print(wav2trans[n.replace(ext,"")],file=wrd)
            print(" ".join(list(wav2trans[n.replace(ext,"")].replace(" ", "|"))) + " |", file=ltr)


if __name__ == "__main__":
    dataset_lens = []
    #create train test splits for x-validation. For each split, create a manifest subfolder
    for test_folder in tqdm.tqdm(os.listdir(os.path.join(p2root, "Audio"))):
        if test_folder == "Transcripts":
            continue
        wavs, samples, root, wav2trans, charset = parse_folder(test_folder)
        write_manifest(wavs, samples, root, wav2trans, f"{manifest}/fold_{test_folder}", "valid")
        train_wavs, train_samples, train_root, train_wav2trans = [], [], [], dict()
        test_len = len(wavs)
        for folder in tqdm.tqdm(os.listdir(os.path.join(p2root, "Audio"))):
            if folder == "Transcripts":
                continue
            if folder == test_folder:
                continue
            wavs, samples, root, wav2trans, charset = parse_folder(folder)
            train_wavs.extend(wavs); train_samples.extend(samples); train_root.append(root); train_wav2trans.update(wav2trans)
        for root in train_root:
            assert root == train_root[0]
        train_root = train_root[0]    
        write_manifest(train_wavs, train_samples, train_root, train_wav2trans, f"{manifest}/fold_{test_folder}", "train")
        
        train_len = len(train_wavs)
        dataset_lens.append((train_len, test_len, train_len+test_len))
        
    for train_len, test_len, total_len in dataset_lens:
        assert train_len + test_len == total_len
    #sort the charset and write it to a file
    charset = sorted(list(charset))
    for fold in os.listdir(manifest):
        with open(os.path.join(manifest,fold,"dict.ltr.txt"),'w') as dct:
            for e,c in enumerate(charset):
                print(c,e,file=dct)