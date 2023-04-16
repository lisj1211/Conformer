import numpy as np
import torch


def collate_fn(batch):
    """对一个batch的数据处理"""
    batch_size = len(batch)
    # 找出音频长度最长的
    max_audio_len_sample = sorted(batch, key=lambda data: data[0].shape[0], reverse=True)[0]
    max_audio_length, freq_size = max_audio_len_sample[0].shape
    # 找出标签最长的
    max_label_len_sample = sorted(batch, key=lambda data: len(data[1]), reverse=True)[0]
    max_label_length = len(max_label_len_sample[1])

    inputs = np.zeros((batch_size, max_audio_length, freq_size), dtype=np.float32)
    labels = np.ones((batch_size, max_label_length), dtype=np.int32) * -1
    input_lens = []
    label_lens = []
    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[0]
        label_length = target.shape[0]
        # padding
        inputs[x, :seq_length, :] = tensor[:, :]
        labels[x, :label_length] = target[:]
        input_lens.append(seq_length)
        label_lens.append(label_length)
    input_lens = np.array(input_lens, dtype=np.int64)
    label_lens = np.array(label_lens, dtype=np.int64)

    return torch.from_numpy(inputs), torch.from_numpy(labels), torch.from_numpy(input_lens), torch.from_numpy(label_lens)
