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
    workers_per_gpu: int = 2
    """number of workers per gpu"""
    input_models_path: str = "animated_data_paths.json"
    """Path to a json file containing a list of 3D object files"""
    num_gpus: int = 8
    """number of gpus to use. -1 means all available gpus"""


def parse_item(item_path: str) -> tuple:
    """
    Example: 
    input: "000-000/963dca3a0a7b4d6caacab65165829470/0"
    output: (i_id, model_id, animation_idx)
    """
    parts = item_path.split('/')
    assert len(parts) == 3
    uid = parts[1]
    animation_idx = int(parts[2])
    i_id = parts[0]
    return i_id, uid, animation_idx


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
            i_id, uid, animation_idx = parse_item(item)

            command = (
                f"export DISPLAY=:0.{gpu} &&"
                f" blender/blender -b -P blender_scripts/render_objaverse.py --"
                f" --obj objaverse_dataset/glbs/{i_id}/{uid}.glb"
                f" --output_folder /home/tjwr/rendered_objaverse/random_clip"
                f" --views 32"
                f" --fixed_animation_length 24"
                f" --camera_option random"
                f" --animation_idx {animation_idx}"
                f" --downsample 3"
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

    # Wait for all tasks to be completed
    queue.join()

    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)