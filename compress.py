import os
import tarfile

root_dir       = '/home/tjwr/rendered_objaverse'
subdirs        = ['fixed_16_clip', 'random_clip']
compressed_dir = '/home/tjwr/wuyue/L4GM/dataset'
processed_file = '/home/tjwr/wuyue/L4GM/data_train/datalist_8fps.txt'

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_tar_name(subdir, parent, mid, last):
    return f"{subdir}-{parent}-{mid}-{last}.tar"

def load_processed(path):
    seen = set()
    if os.path.exists(path):
        with open(path, 'r') as f:
            seen.update(line.strip() for line in f if line.strip())
    return seen

def main():
    ensure_directory(compressed_dir)
    # 读取已写入的记录 (格式：parent/mid/last.tar)
    processed = load_processed(processed_file)

    # 确保 .txt 存在
    ensure_directory(os.path.dirname(processed_file))
    open(processed_file, 'a').close()

    for subdir in subdirs:
        base_path = os.path.join(root_dir, subdir)
        if not os.path.exists(base_path):
            continue

        for root, dirs, files in os.walk(base_path):
            rel_path = os.path.relpath(root, base_path)
            # 只处理形如 parent/mid/last 的三级目录
            if rel_path.count(os.sep) != 2:
                continue

            parent, mid, last = rel_path.split(os.sep)
            record = f"{parent}/{mid}/{last}.tar"

            # 1) 压缩文件，不管 record 是否已存在，都执行
            tar_name = generate_tar_name(subdir, parent, mid, last)
            tar_path = os.path.join(compressed_dir, tar_name)
            inner_root = os.path.join(base_path, rel_path)

            try:
                with tarfile.open(tar_path, 'w') as tar:
                    for dirpath, _, filenames in os.walk(inner_root):
                        for filename in filenames:
                            file_path = os.path.join(dirpath, filename)
                            arcname   = os.path.relpath(file_path, inner_root)
                            tar.add(file_path, arcname=arcname)

                size_kb = os.path.getsize(tar_path) / 1024
                print(f"✅ [{subdir}] 压缩成功: {tar_name} （{size_kb:.1f} KB）")
            except Exception as e:
                print(f"❌ [{subdir}] 压缩失败: {tar_name}，错误: {e}")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                continue

            # 2) 记录到 datalist_8fps.txt，若 record 未出现过才写入
            if record not in processed:
                with open(processed_file, 'a') as f:
                    f.write(record + '\n')
                processed.add(record)
                print(f"   ✍️ 记录: {record}")
            else:
                print(f"   ↩️ 记录已存在，跳过写入: {record}")

if __name__ == '__main__':
    main()
