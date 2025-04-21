from tqdm import tqdm
import os
import json
import urllib.request
import gzip

BASE_PATH = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/fengkairui-25026/objaverse_animated_metadata'
DOWNLOAD_PATH = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/fengkairui-25026/objaverse_dataset'

def filter_animation():
    metadata_path = os.path.join(BASE_PATH, "metadata")
    animation_list = []
    for i in tqdm(range(160)):
        i_id = f"{i // 1000:03d}-{i % 1000:03d}"
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        
        # check animation count
        for t in data:
            for animation_idx in range(data[t]['animationCount']):
                animation_list.append(f"{i_id}/{t}/{animation_idx}")
    return animation_list


if __name__ == '__main__':
    animation_list = filter_animation()
    
    print(f"Samplt Count. {len(animation_list)}")

    sample_path = '000-049/61bb272c9ea149e18511c2e9c6a77d49/4'
    # 完整的对象相对路径（带前缀 glbs/ 和后缀 .glb）
    object_rel_path = f"glbs/{sample_path}.glb"
    hf_url = (
        "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
        + object_rel_path
    )
    local_path = os.path.join(DOWNLOAD_PATH, object_rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, local_path)
    print(f"Downloaded to {local_path}")