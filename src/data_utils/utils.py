import json
import os
import wave

import numpy as np
import resampy
import soundfile
from pydub import AudioSegment
from tqdm import tqdm
from zhconv import convert

from src.data_utils.binary import DatasetWriter
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_manifest(annotation_dir, train_manifest_path, dev_manifest_path, test_manifest_path,
                    is_change_frame_rate=True, only_keep_zh_en=True, target_sr=16000):
    train_list = []
    dev_list = []
    test_list = []
    durations = []
    for annotation_text in os.listdir(annotation_dir):
        annotation_text_path = os.path.join(annotation_dir, annotation_text)
        with open(annotation_text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in tqdm(lines, desc=f'create {os.path.splitext(annotation_text)[0]} manifest'):
            try:
                audio_path, text = line.strip().split('\t')
            except Exception as e:
                logger.warning(f'{line} 错误，已跳过，错误信息：{e}')
                continue
            # 重新调整音频格式并保存
            if is_change_frame_rate:
                change_rate(audio_path, target_sr=target_sr)
            # 获取音频长度
            audio_data, samplerate = soundfile.read(audio_path)
            duration = float(len(audio_data)) / samplerate
            durations.append(duration)
            text = text.lower().strip()
            if only_keep_zh_en:
                # 过滤非法的字符
                text = is_ustr(text)
            if len(text) == 0 or text == ' ':
                continue
            # 保证全部都是简体
            text = convert(text, 'zh-cn')
            # 加入数据列表中
            line = {'audio_filepath': audio_path, 'text': text, 'duration': duration}
            if annotation_text == 'train.txt':
                train_list.append(line)
            elif annotation_text == 'dev.txt':
                dev_list.append(line)
            else:
                test_list.append(line)

    # 按照音频长度降序
    train_list.sort(key=lambda x: x['duration'], reverse=False)
    dev_list.sort(key=lambda x: x['duration'], reverse=False)
    test_list.sort(key=lambda x: x['duration'], reverse=False)
    # 数据写入到文件中
    with open(train_manifest_path, 'w', encoding='utf-8') as f:
        for line in train_list:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')

    with open(dev_manifest_path, 'w', encoding='utf-8') as f:
        for line in dev_list:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')

    with open(test_manifest_path, 'w', encoding='utf-8') as f:
        for line in test_list:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')

    logger.info(f'完成生成数据列表，数据集总长度为{sum(durations) / 3600.:.1f}小时！, '
                f'最短长度{min(durations):.1f}, 最大长度{max(durations):.1f}')


def create_noise(path, noise_manifest_path, is_change_frame_rate=True, target_sr=16000):
    """生成噪声的数据列表"""
    if not os.path.exists(path):
        logger.info('噪声音频文件为空，已跳过！')
        return
    json_lines = []
    logger.info(f'创建噪声数据列表，路径：{path}，请等待 ...')
    for file in tqdm(os.listdir(path), desc='create_noise'):
        audio_path = os.path.join(path, file)
        try:
            # 噪声的标签可以标记为空
            text = ''
            # 重新调整音频格式并保存
            if is_change_frame_rate:
                change_rate(audio_path, target_sr=target_sr)
            f_wave = wave.open(audio_path, 'rb')
            duration = f_wave.getnframes() / f_wave.getframerate()
            json_lines.append(
                json.dumps({'audio_filepath': audio_path, 'duration': duration, 'text': text}, ensure_ascii=False)
            )
        except Exception:
            continue
    with open(noise_manifest_path, 'w', encoding='utf-8') as f_noise:
        for json_line in json_lines:
            f_noise.write(json_line + '\n')


# 获取全部字符
def count_manifest(counter, manifest_path):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines(), desc='count train manifest'):
            line = json.loads(line)
            for char in line['text']:
                counter.update(char)
    if os.path.exists(manifest_path.replace('train', 'dev')):
        with open(manifest_path.replace('train', 'dev'), 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines(), desc='count dev manifest'):
                line = json.loads(line)
                for char in line['text']:
                    counter.update(char)


def read_manifest(manifest_path, max_duration=float('inf'), min_duration=0.5):
    """解析数据列表
    持续时间在[min_duration, max_duration]之外的实例将被过滤。

    :param manifest_path: 数据列表的路径
    :type manifest_path: str
    :param max_duration: 过滤的最长音频长度
    :type max_duration: float
    :param min_duration: 过滤的最短音频长度
    :type min_duration: float
    :return: 数据列表，JSON格式
    :rtype: list
    :raises IOError: If failed to parse the manifest.
    """
    manifest = []
    for json_line in open(manifest_path, 'r', encoding='utf-8'):
        try:
            json_data = json.loads(json_line)
        except Exception as e:
            raise IOError("Error reading manifest: %s" % str(e))
        if max_duration >= json_data["duration"] >= min_duration:
            manifest.append(json_data)
    return manifest


def create_manifest_binary(train_manifest_path, test_manifest_path):
    """
    生成数据列表的二进制文件
    :param train_manifest_path: 训练列表的路径
    :param test_manifest_path: 测试列表的路径
    :return:
    """
    dataset_writer = DatasetWriter(train_manifest_path)
    with open(train_manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.replace('\n', '')
        dataset_writer.add_data(line)
    dataset_writer.close()
    dataset_writer = DatasetWriter(test_manifest_path)
    with open(test_manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines):
        line = line.replace('\n', '')
        dataset_writer.add_data(line)
    dataset_writer.close()


# 将音频流转换为numpy
def buf_to_float(x, n_bytes=2, dtype=np.float32):
    """Convert an integer buffer to floating point values.
    This is primarily useful when loading integer-valued wav data
    into numpy arrays.

    Parameters
    ----------
    x : np.ndarray [dtype=int]
        The integer-valued data buffer

    n_bytes : int [1, 2, 4]
        The number of bytes per sample in ``x``

    dtype : numeric type
        The target output type (default: 32-bit float)

    Returns
    -------
    x_float : np.ndarray [dtype=float]
        The input data buffer cast to floating point
    """

    # Invert the scale of the data
    scale = 1.0 / float(1 << ((8 * n_bytes) - 1))

    # Construct the format string
    fmt = "<i{:d}".format(n_bytes)

    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)


def opus_to_wav(opus_path, save_wav_path, rate=16000):
    source_wav = AudioSegment.from_file(opus_path)
    target_audio = source_wav.set_frame_rate(rate)
    target_audio.export(save_wav_path, format="wav")


def change_rate(audio_path, target_sr=16000):
    """改变音频采样率"""
    is_change = False
    wav, samplerate = soundfile.read(audio_path, dtype='float32')
    # 多通道转单通道
    if wav.ndim > 1:
        wav = wav.T
        wav = np.mean(wav, axis=tuple(range(wav.ndim - 1)))
        is_change = True
    # 重采样
    if samplerate != target_sr:
        wav = resampy.resample(wav, sr_orig=samplerate, sr_new=target_sr)
        is_change = True
    if is_change:
        soundfile.write(audio_path, wav, samplerate=target_sr)


def is_ustr(in_str):
    """过滤非法的字符"""
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str + in_str[i]
    return out_str


def is_uchar(uchar):
    """判断是否为中文字符或者英文字符"""
    if uchar == ' ':
        return True
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    if u'\u0030' <= uchar <= u'\u0039':
        return False
    if (u'\u0041' <= uchar <= u'\u005a') or (u'\u0061' <= uchar <= u'\u007a'):
        return True
    if uchar in ["'"]:
        return True
    if uchar in ['-', ',', '.', '>', '?']:
        return False
    return False
