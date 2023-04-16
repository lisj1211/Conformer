import argparse
import functools
from collections import Counter

import yaml

from src.data_utils.normalizer import FeatureNormalizer
from src.data_utils.utils import create_manifest, create_noise, count_manifest, create_manifest_binary
from src.utils.logger import setup_logger
from src.utils.utils import print_arguments, add_arguments, dict_to_object

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs_path',         str,  'configs/conformer.yml',   '配置文件')
add_arg('annotation_dir',       str,  'data/annotation/',        '标注文件的路径')
add_arg('noise_dir',            str,  'data/noise/',             '噪声音频存放的文件夹路径')
add_arg('is_change_frame_rate', bool, True,                      '是否统一改变音频的采样率')
add_arg('only_keep_zh_en',      bool, True,                      '是否只保留中文和英文字符，训练其他语言可以设置为False')
add_arg('count_threshold',      int,  2,                         '字符计数的截断阈值，0为不做限制')
add_arg('num_workers',          int,  8,                         '读取数据的线程数量')
args = parser.parse_args()


def create_data(config,
                annotation_dir,
                noise_dir,
                count_threshold=2,
                is_change_frame_rate=True,
                only_keep_zh_en=True,
                target_sr=16000):
    """
    创建数据列表和词汇表
    :param config: 配置文件
    :param annotation_dir: 存放标注文件夹的路径
    :param noise_dir: 噪声音频存放的文件夹路径
    :param count_threshold: 字符计数的截断阈值，0为不做限制
    :param is_change_frame_rate: 是否统一改变音频的采样率
    :param only_keep_zh_en: 是否只保留中文和英文字符，训练其他语言可以设置为False
    :param target_sr: 统一的音频采样频率
    """
    logger.info('开始生成数据列表...')
    create_manifest(annotation_dir=annotation_dir,
                    train_manifest_path=config.dataset_conf.train_manifest,
                    dev_manifest_path=config.dataset_conf.dev_manifest,
                    test_manifest_path=config.dataset_conf.test_manifest,
                    only_keep_zh_en=only_keep_zh_en,
                    is_change_frame_rate=is_change_frame_rate,
                    target_sr=target_sr)
    logger.info('=' * 70)

    logger.info('开始生成噪声数据列表...')
    create_noise(path=noise_dir,
                 noise_manifest_path=config.dataset_conf.noise_manifest_path,
                 is_change_frame_rate=is_change_frame_rate,
                 target_sr=target_sr)
    logger.info('=' * 70)

    logger.info('开始生成数据字典...')
    counter = Counter()
    count_manifest(counter, config.dataset_conf.train_manifest)

    count_sorted = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(config.dataset_conf.dataset_vocab, 'w', encoding='utf-8') as f:
        f.write('<blank>\t-1\n')
        f.write('<unk>\t-1\n')
        for char, count in count_sorted:
            if char == ' ':
                char = '<space>'
            if count < count_threshold:
                break
            f.write(char + '\t' + str(count) + '\n')
        f.write('<eos>\t-1\n')
    logger.info('数据字典生成完成！')

    logger.info('=' * 70)
    normalizer = FeatureNormalizer(mean_istd_filepath=config.dataset_conf.mean_istd_path)
    normalizer.compute_mean_istd(manifest_path=config.dataset_conf.train_manifest,
                                 num_workers=config.train_conf.num_workers,
                                 preprocess_configs=config.preprocess_conf,
                                 batch_size=config.train_conf.batch_size)
    logger.info(f'计算的均值和标准值已保存在 {config.dataset_conf.mean_istd_path}！')

    if config.dataset_conf.manifest_type == 'binary':
        logger.info('=' * 70)
        logger.info('正在生成数据列表的二进制文件...')
        create_manifest_binary(train_manifest_path=config.dataset_conf.train_manifest,
                               test_manifest_path=config.dataset_conf.test_manifest)
        logger.info('数据列表的二进制文件生成完成！')


def main():
    with open(args.configs_path, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = dict_to_object(configs)
    print_arguments(args=args, configs=configs)
    create_data(config=configs,
                annotation_dir=args.annotation_dir,
                noise_dir=args.noise_dir)


if __name__ == '__main__':
    main()
