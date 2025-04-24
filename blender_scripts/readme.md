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


blender/blender -b -P blender_scripts/render_objaverse.py -- \
    --obj ../objaverse_dataset/glbs/000-000/0013bdaec08345ec9fd03214030baeb2.glb \
    --output_folder ../rendered_objaverse_dataset \
    --gpu 1 \
    --animation_idx 16