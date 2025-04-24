from tqdm import tqdm
import os
import json
import urllib.request
import gzip

ROOT_PATH = '.'
BASE_PATH = './objaverse_animated_metadata'
DOWNLOAD_PATH = './objaverse_dataset'

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
    # output_json_path = os.path.join(ROOT_PATH, "animated_data_paths.json")
    # animation_list = filter_animation()
    # print(f"Sample Count. {len(animation_list)}")
    # with open(output_json_path, "w") as f:
    #     json.dump(animation_list, f)
    
    # with open(output_json_path, 'r') as f:
    #     entries = json.load(f)
    # unique_pairs = set()
    # for entry in entries:
    #     parts = entry.strip().split('/')
    #     if len(parts) >= 2:
    #         i_id, uid = parts[0], parts[1]
    #         unique_pairs.add(f"{i_id}/{uid}")
    # unique_pairs = list(unique_pairs)

    unique_pairs = ["000-000/0013bdaec08345ec9fd03214030baeb2"]
    
    print(f"glb Count. {len(unique_pairs)}")

    for entry in tqdm(unique_pairs):
        try:
            i_id, uid = entry.split('/')
            rel_path = f"glbs/{i_id}/{uid}.glb"
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{rel_path}"
            local_path = os.path.join(DOWNLOAD_PATH, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        except Exception as e:
            print(f"Failed to download {entry}: {e}")