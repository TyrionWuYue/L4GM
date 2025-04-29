import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import tyro


@dataclass
class Args:
    workers_per_gpu: int = 8
    """number of workers per gpu"""
    input_models_path: str = "animated_data_path.json"
    """Path to a json file containing a list of 3D object files"""
    num_gpus: int = 8
    """number of gpus to use. -1 means all available gpus"""


def parse_item(item_path: str) -> tuple:
    """
    input: "000-000/963dca3a0a7b4d6caacab65165829470/0"
    output: (model_id, output_subpath, animation_idx)
    """
    parts = item_path.split('/')
    assert len(parts) == 3
    output_subpath = '/'.join(parts[:2])
    model_id = parts[1]
    animation_idx = int(parts[2])
    return model_id, output_subpath, animation_idx


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        
        try:
            # Perform some operation on the item
            model_id, output_subpath, animation_idx = parse_item(item)

            command = (
                f"export DISPLAY=:0.{gpu} &&"
                f" blender/blender -b -P scripts/blender_script.py --"
                f"--obj ../objaverse_dataset/0013bdaec08345ec9fd03214030baeb2.glb"
                f"--otuput_folder ../random_clip"
                f"--views 32"
                f"--gpu 8"
                f"--camera_option random"
                f"--animation_idx {animation_idx}"
                f"--downsample 3"
            )

            subprocess.run(command, shell=True, check=True)

            with count.get_lock():
                count.value += 1

        except Exception as e:
            print(f"Error processing {item} on GPU{gpu}: {str(e)}")

        finally:
            queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for _ in range(args.workers_per_gpu):
            process = multiprocessing.Process(
                target=worker, args=(queue, count, gpu_i)
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    with open(args.input_models_path, "r") as f:
        model_paths = json.load(f)
    
    for item in model_paths:
        queue.put(item)

    try:
        while count.value < len(model_paths):
            print(f"\rProgress: {count.value}/{len(model_paths)}", end="")
            time.sleep(1)
        print("\nAll tasks completed!")
    
    finally:
        for _ in range(args.num_gpus * args.workers_per_gpu):
            queue.put(None)
        queue.join()