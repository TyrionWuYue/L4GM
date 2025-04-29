Installation
1. Install Blender

wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz
tar -xf blender-3.2.2-linux-x64.tar.xz
rm blender-3.2.2-linux-x64.tar.xz

2. Update certificates for Blender to download URLs
update-ca-certificates --fresh
export SSL_CERT_DIR=/etc/ssl/certs

Blender env  

./blender/3.2/python/bin/python3.10 -m ensurepip  
./blender/3.2/python/bin/python3.10 -m pip install opencv-python loguru


8FPS
32 random cameras
blender/blender -b -P blender_scripts/render_objaverse.py -- \
    --obj ../objaverse_dataset/0013bdaec08345ec9fd03214030baeb2.glb \
    --output_folder ../random_clip \
    --views 32 \
    --gpu 8 \
    --camera_option random \
    --animation_idx 1 \
    --downsample 3

16 fixed cameras
blender/blender -b -P blender_scripts/render_objaverse.py -- \
    --obj ../objaverse_dataset/0013bdaec08345ec9fd03214030baeb2.glb \
    --output_folder ../fixed_16_clip \
    --views 16 \
    --gpu 8 \
    --camera_option fixed \
    --animation_idx 1 \
    --downsample 3

24FPS
32 random cameras
blender/blender -b -P blender_scripts/render_objaverse.py -- \
    --obj ../objaverse_dataset/0013bdaec08345ec9fd03214030baeb2.glb \
    --output_folder ../random_24fps \
    --views 32 \
    --gpu 8 \
    --camera_option random \
    --animation_idx 1 \


16 fixed cameras
blender/blender -b -P blender_scripts/render_objaverse.py -- \
    --obj ../objaverse_dataset/0013bdaec08345ec9fd03214030baeb2.glb \
    --output_folder ../fixed_16_24fps \
    --views 16 \
    --gpu 8 \
    --camera_option fixed \
    --animation_idx 16 \

