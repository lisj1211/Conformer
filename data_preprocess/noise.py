import argparse
import os
import shutil

from utils import unzip, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--noise_zip_path", type=str, default="../data/rirs_noises.zip", help="存放噪声压缩文件的路径")
parser.add_argument("--noise_path", type=str, default="../data/noise/", help="存放噪声文件处理后的目录")
args = parser.parse_args()


def prepare_dataset(zip_path, noise_path):
    """unpack and move noise file."""
    unzip(zip_path, os.path.dirname(zip_path))
    data_dir = os.path.join(os.path.dirname(zip_path), 'RIRS_NOISES')
    # 移动噪声音频到指定文件夹
    os.makedirs(noise_path, exist_ok=True)
    json_lines = []
    data_types = ['pointsource_noises', 'real_rirs_isotropic_noises', 'simulated_rirs']
    for dtype in data_types:
        del json_lines[:]
        audio_dir = os.path.join(data_dir, dtype)
        for root, _, filelist in sorted(os.walk(audio_dir)):
            for filename in filelist:
                if '.wav' not in filename:
                    continue
                audio_path = os.path.join(root, filename)
                shutil.move(audio_path, os.path.join(noise_path, filename))
    shutil.rmtree(data_dir, ignore_errors=True)


def main():
    print_arguments(args)
    prepare_dataset(zip_path=args.noise_zip_path, noise_path=args.noise_path)


if __name__ == '__main__':
    main()
