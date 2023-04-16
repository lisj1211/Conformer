"""查找最优的集束搜索方法的alpha参数和beta参数"""
import os
import argparse
import functools

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.decoders.beam_search_decoder import BeamSearchDecoder
from src.data_utils.dataset import MyDataset
from src.data_utils.collate_fn import collate_fn
from src.models.conformer.model import ConformerModel
from src.utils.metrics import cer, wer
from src.utils.utils import labels_to_string, add_arguments, print_arguments, dict_to_object

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',      str,   'configs/conformer.yml',                        '配置文件')
add_arg('resume_model', str,   'models/conformer_streaming_fbank/best_model/', '模型的路径')
add_arg('num_alphas',   int,   30,                                             '用于调优的alpha候选个数')
add_arg('num_betas',    int,   20,                                             '用于调优的beta候选个数')
add_arg('alpha_from',   float, 1.0,                                            'alpha调优开始大小')
add_arg('alpha_to',     float, 3.2,                                            'alpha调优结速大小')
add_arg('beta_from',    float, 0.1,                                            'beta调优开始大小')
add_arg('beta_to',      float, 4.5,                                            'beta调优结速大小')
args = parser.parse_args()
print_arguments(args)


def tune():
    assert os.path.exists(args.configs), f'配置文件{args.configs}不存在'
    with open(args.configs, 'r', encoding='utf-8') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = dict_to_object(configs)
    # 逐步调整alphas参数和betas参数
    if not args.num_alphas >= 0:
        raise ValueError("num_alphas must be non-negative!")
    if not args.num_betas >= 0:
        raise ValueError("num_betas must be non-negative!")

    # 获取测试数据
    test_dataset = MyDataset(preprocess_configs=configs.preprocess_conf,
                             data_manifest=configs.dataset_conf.test_manifest,
                             vocab_filepath=configs.dataset_conf.dataset_vocab,
                             manifest_type=configs.dataset_conf.manifest_type)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=configs.train_conf.batch_size,
                             collate_fn=collate_fn,
                             prefetch_factor=configs.train_conf.prefetch_factor,
                             num_workers=configs.train_conf.num_workers)

    # 获取模型
    model = ConformerModel(input_dim=test_dataset.feature_dim,
                           vocab_size=test_dataset.vocab_size,
                           mean_istd_path=configs.dataset_conf.mean_istd_path,
                           streaming=configs.streaming,
                           encoder_conf=configs.encoder_conf,
                           decoder_conf=configs.decoder_conf,
                           **configs.model_conf)

    assert os.path.exists(os.path.join(args.resume_model, 'model.pt')), "模型不存在！"
    model.cuda()
    model.load_state_dict(torch.load(os.path.join(args.resume_model, 'model.pt')))
    model.eval()

    # 创建用于搜索的alphas参数和betas参数
    cand_alphas = np.linspace(args.alpha_from, args.alpha_to, args.num_alphas)
    cand_betas = np.linspace(args.beta_from, args.beta_to, args.num_betas)
    params_grid = [(round(alpha, 2), round(beta, 2)) for alpha in cand_alphas for beta in cand_betas]
    eos = test_dataset.vocab_size - 1

    predicts = []
    ground_trues = []
    print('开始识别数据...')
    used_sum = 0
    for inputs, labels, input_lens, label_lens in tqdm(test_loader):
        inputs = inputs.cuda()
        input_lens = input_lens.cuda()
        used_sum += inputs.shape[0]
        outs = model.get_encoder_out(inputs, input_lens).cpu().detach().numpy()
        predicts.append(outs)
        ground_trues.append(labels)

    print('开始使用识别结果解码...')
    print(f'解码alpha和beta的排列：{params_grid}')
    # 搜索alphas参数和betas参数
    best_alpha, best_beta, best_cer = 0, 0, 1
    for i, (alpha, beta) in enumerate(params_grid):
        beam_search_decoder = BeamSearchDecoder(
            vocab_list=test_dataset.vocab_list,
            alpha=alpha,
            beta=beta,
            beam_size=configs.ctc_beam_search_decoder_conf.beam_size,
            num_processes=configs.ctc_beam_search_decoder_conf.num_processes,
            cutoff_prob=configs.ctc_beam_search_decoder_conf.cutoff_prob,
            cutoff_top_n=configs.ctc_beam_search_decoder_conf.cutoff_top_n,
            language_model_path=configs.ctc_beam_search_decoder_conf.language_model_path
        )

        c = []
        print('正在解码[%d/%d]: (%.2f, %.2f)' % (i, len(params_grid), alpha, beta))
        for j in tqdm(range(len(ground_trues))):
            outs, label = predicts[j], ground_trues[j]
            out_strings = beam_search_decoder.decode_batch_beam_search_offline(probs_split=outs)
            labels_str = labels_to_string(label, test_dataset.vocab_list, eos=eos)
            for out_string, label in zip(*(out_strings, labels_str)):
                # 计算字错率或者词错率
                if configs.metrics_type == 'wer':
                    c.append(wer(out_string, label))
                else:
                    c.append(cer(out_string, label))
        c = float(sum(c) / len(c))
        if c < best_cer:
            best_alpha = alpha
            best_beta = beta
            best_cer = c
        print('当alpha为：%f, beta为：%f，%s：%f' % (alpha, beta, configs.metrics_type, c))
    print('【最后结果】当alpha为：%f, beta为：%f，%s最低，为：%f' % (best_alpha, best_beta, configs.metrics_type, best_cer))


if __name__ == '__main__':
    tune()
