import json
import os
import platform
import shutil
import time
from datetime import timedelta

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_utils.collate_fn import collate_fn
from src.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from src.data_utils.featurizer.text_featurizer import TextFeaturizer
from src.data_utils.dataset import MyDataset
from src.models.conformer.model import ConformerModel
from src.optimizer.scheduler import WarmupLR, NoamHoldAnnealing, CosineWithWarmup
from src.decoders.ctc_greedy_decoder import greedy_decoder_batch
from src.utils.logger import setup_logger
from src.utils.metrics import cer, wer
from src.utils.utils import dict_to_object, print_arguments
from src.utils.utils import labels_to_string

logger = setup_logger(__name__)


class SRTrainer:
    """Conformer Speech Recognition Framework"""

    def __init__(self, configs, use_gpu=True):
        """
        :param configs: 配置文件路径
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device('cuda:1')
        else:
            self.device = torch.device('cpu')
        # 读取配置文件
        if os.path.exists(configs):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f, Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        else:
            raise ValueError('当前config文件不存在')
        self.configs = dict_to_object(configs)
        self.use_gpu = use_gpu
        self.model = None
        self.test_loader = None
        self.beam_search_decoder = None
        self.train_step = 0

    def __setup_dataloader(self, augment_conf_path=None, is_train=False):
        if augment_conf_path is not None and os.path.exists(augment_conf_path) and is_train:
            augmentation_config = open(augment_conf_path, 'r', encoding='utf8').read()
        else:
            if augment_conf_path is not None and not os.path.exists(augment_conf_path):
                logger.warning(f'数据增强配置文件{augment_conf_path}不存在')
            augmentation_config = '{}'
        if not os.path.exists(self.configs.dataset_conf.mean_istd_path):
            raise FileNotFoundError(f'归一化列表文件 {self.configs.dataset_conf.mean_istd_path} 不存在')
        if is_train:
            self.train_dataset = MyDataset(preprocess_configs=self.configs.preprocess_conf,
                                           data_manifest=self.configs.dataset_conf.train_manifest,
                                           vocab_filepath=self.configs.dataset_conf.dataset_vocab,
                                           min_duration=self.configs.dataset_conf.min_duration,
                                           max_duration=self.configs.dataset_conf.max_duration,
                                           augmentation_config=augmentation_config,
                                           manifest_type=self.configs.dataset_conf.manifest_type,
                                           train=is_train)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.configs.train_conf.batch_size,
                                           collate_fn=collate_fn,
                                           prefetch_factor=self.configs.train_conf.prefetch_factor,
                                           num_workers=self.configs.train_conf.num_workers,
                                           drop_last=True,
                                           shuffle=True)
            self.dev_dataset = MyDataset(preprocess_configs=self.configs.preprocess_conf,
                                         data_manifest=self.configs.dataset_conf.dev_manifest,
                                         vocab_filepath=self.configs.dataset_conf.dataset_vocab,
                                         min_duration=self.configs.dataset_conf.min_duration,
                                         max_duration=self.configs.dataset_conf.max_duration,
                                         manifest_type=self.configs.dataset_conf.manifest_type)
            self.dev_loader = DataLoader(dataset=self.dev_dataset,
                                         batch_size=self.configs.train_conf.batch_size,
                                         collate_fn=collate_fn,
                                         prefetch_factor=self.configs.train_conf.prefetch_factor,
                                         num_workers=self.configs.train_conf.num_workers)
        # 获取测试数据
        self.test_dataset = MyDataset(preprocess_configs=self.configs.preprocess_conf,
                                      data_manifest=self.configs.dataset_conf.test_manifest,
                                      vocab_filepath=self.configs.dataset_conf.dataset_vocab,
                                      manifest_type=self.configs.dataset_conf.manifest_type,
                                      min_duration=self.configs.dataset_conf.min_duration,
                                      max_duration=self.configs.dataset_conf.max_duration)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.configs.train_conf.batch_size,
                                      collate_fn=collate_fn,
                                      prefetch_factor=self.configs.train_conf.prefetch_factor,
                                      num_workers=self.configs.train_conf.num_workers)

    def __setup_model(self, input_dim, vocab_size, is_train=False):
        self.model = ConformerModel(input_dim=input_dim,
                                    vocab_size=vocab_size,
                                    mean_istd_path=self.configs.dataset_conf.mean_istd_path,
                                    streaming=self.configs.streaming,
                                    encoder_conf=self.configs.encoder_conf,
                                    decoder_conf=self.configs.decoder_conf,
                                    **self.configs.model_conf)
        self.model.to(self.device)
        if is_train:
            if self.configs.train_conf.enable_amp:
                self.amp_scaler = torch.cuda.amp.GradScaler(init_scale=1024)
            # 获取优化方法
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=float(self.configs.optimizer_conf.learning_rate),
                                                  weight_decay=float(self.configs.optimizer_conf.weight_decay))
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=float(self.configs.optimizer_conf.learning_rate),
                                                   weight_decay=float(self.configs.optimizer_conf.weight_decay))
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.momentum,
                                                 lr=float(self.configs.optimizer_conf.learning_rate),
                                                 weight_decay=float(self.configs.optimizer_conf.weight_decay))
            else:
                raise ValueError(f'不支持优化方法：{optimizer}')
            # 学习率衰减
            scheduler_conf = self.configs.optimizer_conf.scheduler_conf
            scheduler = self.configs.optimizer_conf.scheduler
            if scheduler == 'WarmupLR':
                self.scheduler = WarmupLR(optimizer=self.optimizer, **scheduler_conf)
            elif scheduler == 'NoamHoldAnnealing':
                self.scheduler = NoamHoldAnnealing(optimizer=self.optimizer, **scheduler_conf)
            elif scheduler == 'CosineWithWarmup':
                self.scheduler = CosineWithWarmup(optimizer=self.optimizer, **scheduler_conf)
            else:
                raise Exception(f'不支持学习率衰减方法：{scheduler}')

    def __load_pretrained(self, pretrained_model):
        if pretrained_model is None:
            return
        if os.path.isdir(pretrained_model):
            pretrained_model = os.path.join(pretrained_model, 'model.pt')
        assert os.path.exists(pretrained_model), f'{pretrained_model} 模型不存在！'
        model_dict = self.model.state_dict()
        model_state_dict = torch.load(pretrained_model)

        for name, weight in model_dict.items():
            if name in model_state_dict.keys():
                if list(weight.shape) != list(model_state_dict[name].shape):
                    logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                   format(name, list(model_state_dict[name].shape), list(weight.shape)))
                    model_state_dict.pop(name, None)
            else:
                logger.warning(f'Lack weight: {name}')
        self.model.load_state_dict(model_state_dict, strict=False)
        logger.info(f'成功加载预训练模型：{pretrained_model}')

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_error_rate = 1.0
        save_model_name = f'{self.configs.use_model}_{"streaming" if self.configs.streaming else "non-streaming"}' \
                          f'_{self.configs.preprocess_conf.feature_method}'
        last_model_dir = os.path.join(save_model_path, save_model_name, 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pt'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pt'))):
            # 判断从指定resume_model恢复训练，还是last_model恢复训练
            if resume_model is None:
                resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pt')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pt')), "优化方法参数文件不存在！"
            self.model.load_state_dict(torch.load(os.path.join(resume_model, 'model.pt')))
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                if 'dev_cer' in json_data.keys():
                    best_error_rate = abs(json_data['dev_cer'])
                if 'dev_wer' in json_data.keys():
                    best_error_rate = abs(json_data['dev_wer'])
            logger.info(f'成功恢复模型参数和优化方法参数：{resume_model}')
        return last_epoch, best_error_rate

    def __save_checkpoint(self, save_model_path, epoch_id, error_rate=1.0, test_loss=1e3, best_model=False):
        save_model_name = f'{self.configs.use_model}_{"streaming" if self.configs.streaming else "non-streaming"}' \
                          f'_{self.configs.preprocess_conf.feature_method}'
        if best_model:
            model_path = os.path.join(save_model_path, save_model_name, 'best_model')
        else:
            model_path = os.path.join(save_model_path, save_model_name, f'epoch_{epoch_id}')
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pt'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            f.write('{{"last_epoch": {}, "test_{}": {}, "test_loss": {}}}'.
                    format(epoch_id, self.configs.metrics_type, error_rate, test_loss))
        if not best_model:
            last_model_path = os.path.join(save_model_path, save_model_name, 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, save_model_name, f'epoch_{epoch_id - 2}')
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info(f'已保存模型：{model_path}')

    def __decoder_result(self, outs, vocabulary):
        if self.configs.decoder == 'ctc_beam_search' and self.beam_search_decoder is None:
            if platform.system() != 'Windows':
                try:
                    from src.decoders.beam_search_decoder import BeamSearchDecoder
                    self.beam_search_decoder = BeamSearchDecoder(vocab_list=vocabulary,
                                                                 **self.configs.ctc_beam_search_decoder_conf)
                except ModuleNotFoundError:
                    logger.warning('==================================================================')
                    logger.warning('缺少 paddlespeech-ctcdecoders 库，请根据文档安装。')
                    logger.warning('【注意】已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                    logger.warning('==================================================================\n')
                    self.configs.decoder = 'ctc_greedy'
            else:
                logger.warning('==================================================================')
                logger.warning(
                    '【注意】Windows不支持ctc_beam_search，已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。'
                )
                logger.warning('==================================================================\n')
                self.configs.decoder = 'ctc_greedy'

        if self.configs.decoder == 'ctc_greedy':
            result = greedy_decoder_batch(outs, vocabulary)
        else:
            result = self.beam_search_decoder.decode_batch_beam_search_offline(probs_split=outs)
        return result

    def __train_epoch(self, epoch_id):
        accum_grad = self.configs.train_conf.accum_grad
        grad_clip = self.configs.train_conf.grad_clip
        reader_times, batch_times = [], []
        start = time.time()
        self.model.train()
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc=f'epoch:{epoch_id}')):
            inputs, labels, input_lens, label_lens = batch
            reader_times.append((time.time() - start) * 1000)
            start_step = time.time()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            input_lens = input_lens.to(self.device)
            label_lens = label_lens.to(self.device)
            num_samples = label_lens.size(0)
            if num_samples == 0:
                continue
            # 执行模型计算，是否开启自动混合精度
            with torch.cuda.amp.autocast(enabled=self.configs.train_conf.enable_amp):
                loss_dict = self.model(inputs, input_lens, labels, label_lens)
            loss = loss_dict['loss'] / accum_grad
            # 是否开启自动混合精度
            if self.configs.train_conf.enable_amp:
                # loss缩放，乘以系数loss_scaling
                scaled = self.amp_scaler.scale(loss)
                scaled.backward()
            else:
                loss.backward()
            # 执行一次梯度计算
            if batch_id % accum_grad == 0:
                # 是否开启自动混合精度
                if self.configs.train_conf.enable_amp:
                    self.amp_scaler.unscale_(self.optimizer)
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    if torch.isfinite(grad_norm):
                        self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.train_step += 1
            batch_times.append((time.time() - start_step) * 1000)
            if batch_id % self.configs.train_conf.log_interval == 0:
                logger.info(f'loss: {loss.cpu().detach().numpy():.5f}, '
                            f'learning_rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'reader_cost: {(sum(reader_times) / len(reader_times) / 1000):.4f}, '
                            f'batch_cost: {(sum(batch_times) / len(batch_times) / 1000):.4f}, ')
            start = time.time()

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              augment_conf_path='configs/augmentation.json'):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        :param augment_conf_path: 数据增强的配置文件，为json格式
        """
        # 训练只能用贪心解码，解码速度快
        self.configs.decoder = 'ctc_greedy'
        # 获取数据
        self.__setup_dataloader(augment_conf_path=augment_conf_path, is_train=True)
        logger.info(f'训练数据大小：{len(self.train_dataset)}， 验证数据大小: {len(self.dev_dataset)}')
        # 获取模型
        self.__setup_model(input_dim=self.test_dataset.feature_dim,
                           vocab_size=self.test_dataset.vocab_size,
                           is_train=True)
        self.__load_pretrained(pretrained_model=pretrained_model)
        # 加载恢复模型
        last_epoch, best_error_rate = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step = 0
        last_epoch += 1
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            self.__train_epoch(epoch_id=epoch_id)
            logger.info('=' * 70)
            loss, error_result = self.evaluate()
            logger.info('Dev result: epoch: {}, time/epoch: {}, loss: {:.5f}, {}: {:.5f}, best {}: {:.5f}'.format(
                epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), loss, self.configs.metrics_type,
                error_result, self.configs.metrics_type,
                error_result if error_result <= best_error_rate else best_error_rate))
            logger.info('=' * 70)
            test_step += 1
            # 保存最优模型
            if error_result <= best_error_rate:
                best_error_rate = error_result
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, error_rate=error_result,
                                       test_loss=loss, best_model=True)
            # 保存模型
            self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, error_rate=error_result,
                                   test_loss=loss)

        # do test
        start_test = time.time()
        loss, error_result = self.evaluate(resume_model='models/conformer_streaming_fbank/best_model/')
        logger.info('Test result: time: {}, loss: {:.5f}, {}: {:.5f}'.format(
            str(timedelta(seconds=(time.time() - start_test))), loss, self.configs.metrics_type, error_result))

    def evaluate(self, resume_model=None, display_result=False):
        """
        评估模型
        :param resume_model: 所使用的模型，为None时表示训练过程中验证集的评估，为str时表示路径模型对测试集的评估
        :param display_result: 是否打印识别结果
        :return: 评估结果
        """
        if resume_model is not None:  # 评估测试集
            self.configs.decoder = 'ctc_beam_search'  # 测试集采用beam_search
            if self.test_loader is None:
                self.__setup_dataloader()
            if self.model is None:
                self.__setup_model(input_dim=self.test_dataset.feature_dim,
                                   vocab_size=self.test_dataset.vocab_size)
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pt')
            assert os.path.exists(resume_model), f'{resume_model} 模型不存在！'
            if self.use_gpu:
                model_state_dict = torch.load(resume_model)
            else:
                model_state_dict = torch.load(resume_model, map_location='cpu')
            self.model.load_state_dict(model_state_dict)
            logger.info(f'成功加载模型：{resume_model}')

        test_loader = self.test_loader if resume_model is not None else self.dev_loader
        error_results, losses = [], []
        eos = self.test_dataset.vocab_size - 1

        self.model.eval()
        with torch.no_grad():
            for batch_id, batch in enumerate(tqdm(test_loader, desc='evaluate')):
                inputs, labels, input_lens, label_lens = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                input_lens = input_lens.to(self.device)

                loss_dict = self.model(inputs, input_lens, labels, label_lens)
                losses.append(loss_dict['loss'].cpu().detach().numpy())

                outputs = self.model.get_encoder_out(inputs, input_lens).cpu().detach().numpy()
                out_strings = self.__decoder_result(outs=outputs, vocabulary=self.test_dataset.vocab_list)
                labels_str = labels_to_string(labels, self.test_dataset.vocab_list, eos=eos)

                for out_string, label in zip(*(out_strings, labels_str)):
                    # 计算字错率或者词错率
                    if self.configs.metrics_type == 'wer':
                        error_rate = wer(out_string, label)
                    else:
                        error_rate = cer(out_string, label)
                    error_results.append(error_rate)
                    if display_result:
                        logger.info(f'预测结果为：{out_string}')
                        logger.info(f'实际标签为：{label}')
                        logger.info(f'这条数据的{self.configs.metrics_type}：{error_rate:.6f}，'
                                    f'当前{self.configs.metrics_type}：{sum(error_results) / len(error_results):.6f)}')
                        logger.info('-' * 70)
        loss = float(sum(losses) / len(losses))
        error_result = float(sum(error_results) / len(error_results))

        return loss, error_result

    def export(self,
               save_model_path='models/',
               resume_model='models/deepspeech2_non-streaming_fbank/best_model/',
               save_quant=False):
        """
        导出预测模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 准备转换的模型路径
        :param save_quant: 是否保存量化模型
        :return:
        """
        audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        text_featurizer = TextFeaturizer(self.configs.dataset_conf.dataset_vocab)
        if not os.path.exists(self.configs.dataset_conf.mean_istd_path):
            raise Exception(f'归一化列表文件 {self.configs.dataset_conf.mean_istd_path} 不存在')

        self.__setup_model(input_dim=audio_featurizer.feature_dim,
                           vocab_size=text_featurizer.vocab_size)
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pt')
        assert os.path.exists(resume_model), f"{resume_model} 模型不存在！"
        if torch.cuda.is_available() and self.use_gpu:
            model_state_dict = torch.load(resume_model)
        else:
            model_state_dict = torch.load(resume_model, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        logger.info(f'成功恢复模型参数和优化方法参数：{resume_model}')

        self.model.eval()
        infer_model = self.model.export()
        save_model_name = f'{self.configs.use_model}_{"streaming" if self.configs.streaming else "non-streaming"}' \
                          f'_{self.configs.preprocess_conf.feature_method}'
        infer_model_path = os.path.join(save_model_path, save_model_name, 'inference.pt')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info(f'预测模型已保存：{infer_model_path}')
        # 保存量化模型
        if save_quant:
            quant_model_path = os.path.join(os.path.dirname(infer_model_path), 'inference_quant.pt')
            quantized_model = torch.quantization.quantize_dynamic(self.model)
            script_quant_model = torch.jit.script(quantized_model)
            torch.jit.save(script_quant_model, quant_model_path)
            logger.info(f'量化模型已保存：{quant_model_path}')
