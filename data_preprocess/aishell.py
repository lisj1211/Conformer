import argparse
import os

from utils import unpack, print_arguments


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tgz_path", type=str, default="../data/data_aishell.tgz", help="存放原始数据集的路径")
parser.add_argument("--annotation_text", type=str, default="../data/annotation/", help="存放音频标注文件的目录")
args = parser.parse_args()


def create_annotation_text(data_dir, annotation_path):
    print('Create Aishell annotation text ...')
    os.makedirs(annotation_path, exist_ok=True)
    transcript_path = os.path.join(data_dir, 'transcript', 'aishell_transcript_v0.8.txt')
    transcript_dict = {}
    for line in open(transcript_path, 'r', encoding='utf-8'):
        line = line.strip()
        if line == '':
            continue
        audio_id, text = line.split(' ', 1)
        # remove space
        text = ''.join(text.split())
        transcript_dict[audio_id] = text

    data_types = ['train', 'dev', 'test']
    for type_ in data_types:
        with open(os.path.join(annotation_path, f'{type_}.txt'), 'w', encoding='utf-8') as f:
            audio_dir = os.path.join(data_dir, 'wav', type_)
            for root, _, filelist in sorted(os.walk(audio_dir)):
                for filename in filelist:
                    audio_path = os.path.abspath(os.path.join(root, filename))
                    audio_id = os.path.splitext(filename)[0]
                    # if no transcription for audio then skipped
                    if audio_id not in transcript_dict:
                        continue
                    text = transcript_dict[audio_id]
                    f.write(audio_path + '\t' + text + '\n')


def prepare_dataset(tgz_path, annotation_path):
    """unpack and create manifest file."""
    unpack(tgz_path, os.path.dirname(tgz_path))
    data_path = os.path.join(os.path.dirname(tgz_path), 'data_aishell')
    audio_dir = os.path.join(data_path, 'wav')
    for root, dirs, filelist in sorted(os.walk(audio_dir)):
        for file in filelist:
            unpack(os.path.join(root, file), root, True)
    create_annotation_text(data_path, annotation_path)


def main():
    print_arguments(args)
    prepare_dataset(tgz_path=args.tgz_path, annotation_path=args.annotation_text)


if __name__ == '__main__':
    main()
