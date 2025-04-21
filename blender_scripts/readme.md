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
    --obj ../objaverse_dataset/glbs/000-049/61bb272c9ea149e18511c2e9c6a77d49.glb \
    --output_folder ../rendered_objaverse_dataset \
    --gpu 1