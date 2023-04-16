import os
import platform

import cn2an
import numpy as np
import yaml

from src.data_utils.audio import AudioSegment
from src.data_utils.featurizer.audio_featurizer import AudioFeaturizer
from src.data_utils.featurizer.text_featurizer import TextFeaturizer
from src.decoders.ctc_greedy_decoder import greedy_decoder, greedy_decoder_chunk
from src.infer_utils.inference_predictor import InferencePredictor
from src.utils.logger import setup_logger
from src.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class SRPredictor:
    def __init__(self,
                 configs=None,
                 model_path=None,
                 use_gpu=True):
        """
        语音识别预测工具
        :param configs: 配置文件路径
        :param model_path: 导出的预测模型文件夹路径
        :param use_gpu: 是否使用GPU预测
        """
        if not isinstance(configs, str) or not os.path.exists(configs):
            raise FileNotFoundError('configs文件不存在')
        with open(configs, 'r', encoding='utf-8') as f:
            configs = yaml.load(f.read(), Loader=yaml.FullLoader)
        print_arguments(configs=configs)

        self.configs = dict_to_object(configs)
        self.running = False
        self.use_gpu = use_gpu
        self.inv_normalizer = None
        self.vad_predictor = None
        self._text_featurizer = TextFeaturizer(vocab_filepath=self.configs.dataset_conf.dataset_vocab)
        self._audio_featurizer = AudioFeaturizer(**self.configs.preprocess_conf)
        # 流式解码参数
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        self.__init_decoder()
        # 创建模型
        if not os.path.exists(model_path):
            raise Exception("模型文件不存在，请检查{}是否存在！".format(model_path))
        # 获取预测器
        self.predictor = InferencePredictor(configs=self.configs,
                                            use_model=self.configs.use_model,
                                            streaming=self.configs.streaming,
                                            model_path=model_path,
                                            use_gpu=self.use_gpu)
        # 预热
        for _ in range(5):
            warmup_audio = np.random.uniform(low=-2.0, high=2.0, size=(134240,))
            self.predict(audio_data=warmup_audio, is_itn=False)
            if 'online' in self.configs.use_model:
                self.predict_stream(audio_data=warmup_audio[:8000], is_itn=False)
        self.reset_stream()

    # 初始化解码器
    def __init_decoder(self):
        # 集束搜索方法的处理
        if self.configs.decoder == "ctc_beam_search":
            if platform.system() != 'Windows':
                try:
                    from src.decoders.beam_search_decoder import BeamSearchDecoder
                    self.beam_search_decoder = BeamSearchDecoder(vocab_list=self._text_featurizer.vocab_list,
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
                    '【注意】Windows不支持ctc_beam_search，已自动切换为ctc_greedy解码器，ctc_greedy解码器准确率相对较低。')
                logger.warning('==================================================================\n')
                self.configs.decoder = 'ctc_greedy'

    # 解码模型输出结果
    def decode(self, output_data, is_itn):
        """
        解码模型输出结果
        :param output_data: 模型输出结果
        :param is_itn: 是否对文本进行反标准化
        :return:
        """
        # 执行解码
        if self.configs.decoder == 'ctc_beam_search':
            # 集束搜索解码策略
            result = self.beam_search_decoder.decode_beam_search_offline(probs_split=output_data)
        else:
            # 贪心解码策略
            result = greedy_decoder(probs_seq=output_data, vocabulary=self._text_featurizer.vocab_list)

        score, text = result[0], result[1]
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)
        return score, text

    # 预测音频
    def predict(self,
                audio_data,
                is_itn=False,
                sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        audio_feature = self._audio_featurizer.featurize(audio_segment)
        input_data = np.array(audio_feature).astype(np.float32)[np.newaxis, :]
        audio_len = np.array([input_data.shape[1]]).astype(np.int64)

        # 运行predictor
        output_data = self.predictor.predict(input_data, audio_len)[0]

        # 解码
        score, text = self.decode(output_data=output_data, is_itn=is_itn)
        result = {'text': text, 'score': score}
        return result

    # 长语音预测
    def predict_long(self,
                     audio_data,
                     is_itn=False,
                     sample_rate=16000):
        """
        预测函数，只预测完整的一句话。
        :param audio_data: 需要识别的数据，支持文件路径，字节，numpy。如果是字节的话，必须是完整的字节文件
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        if self.vad_predictor is None:
            from src.infer_utils.vad_predictor import VADPredictor
            self.vad_predictor = VADPredictor()
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        # 重采样，方便进行语音活动检测
        if audio_segment.sample_rate != self.configs.preprocess_conf.sample_rate:
            audio_segment.resample(self.configs.preprocess_conf.sample_rate)
        # 获取语音活动区域
        speech_timestamps = self.vad_predictor.get_speech_timestamps(audio_segment.samples, audio_segment.sample_rate)
        texts, scores = '', []
        for t in speech_timestamps:
            audio_ndarray = audio_segment.samples[t['start']: t['end']]
            # 执行识别
            result = self.predict(audio_data=audio_ndarray, is_itn=is_itn)
            score, text = result['score'], result['text']
            if text != '':
                texts = texts + '，' + text
            scores.append(score)
            logger.info(f'长语音识别片段结果：{text}')
        if texts[0] == '，':
            texts = texts[1:]

        result = {'text': texts, 'score': round(sum(scores) / len(scores), 2)}
        return result

    # 预测音频
    def predict_stream(self,
                       audio_data,
                       is_end=False,
                       is_itn=False,
                       sample_rate=16000):
        """
        预测函数，流式预测，通过一直输入音频数据，实现实时识别。
        :param audio_data: 需要预测的音频wave读取的字节流或者未预处理的numpy值
        :param is_end: 是否结束语音识别
        :param is_itn: 是否对文本进行反标准化
        :param sample_rate: 如果传入的事numpy数据，需要指定采样率
        :return: 识别的文本结果和解码的得分数
        """
        if not self.configs.streaming:
            raise Exception(
                f"不支持改该模型流式识别，当前模型：{self.configs.use_model}，参数streaming为：{self.configs.streaming}")
        # 加载音频文件，并进行预处理
        if isinstance(audio_data, np.ndarray):
            audio_data = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_data = AudioSegment.from_wave_bytes(audio_data)
        else:
            raise Exception(f'不支持该数据类型，当前数据类型为：{type(audio_data)}')
        if self.remained_wav is None:
            self.remained_wav = audio_data
        else:
            self.remained_wav = AudioSegment(np.concatenate([self.remained_wav.samples, audio_data.samples]),
                                             audio_data.sample_rate)

        # 预处理语音块
        x_chunk = self._audio_featurizer.featurize(self.remained_wav)
        x_chunk = np.array(x_chunk).astype(np.float32)[np.newaxis, :]
        if self.cached_feat is None:
            self.cached_feat = x_chunk
        else:
            self.cached_feat = np.concatenate([self.cached_feat, x_chunk], axis=1)
        self.remained_wav._samples = self.remained_wav.samples[160 * x_chunk.shape[1]:]  # 因为提取mel时的hop_length为10ms, 采样率为16000, 所以重叠窗口为160, 这里为了保证cached_mel_feat

        # 识别的数据块大小
        decoding_chunk_size = 16
        context = 7
        subsampling = 4

        cached_feature_num = context - subsampling
        decoding_window = (decoding_chunk_size - 1) * subsampling + context
        stride = subsampling * decoding_chunk_size

        # 保证每帧数据长度都有效
        num_frames = self.cached_feat.shape[1]
        if num_frames < decoding_window and not is_end:
            return None
        if num_frames < context:
            return None

        # 如果识别结果，要使用最后一帧
        if is_end:
            left_frames = context
        else:
            left_frames = decoding_window

        score, text, end = None, None, None
        for cur in range(0, num_frames - left_frames + 1, stride):
            end = min(cur + decoding_window, num_frames)
            # 获取数据块
            x_chunk = self.cached_feat[:, cur:end, :]

            # 执行识别
            if 'former' in self.configs.use_model:
                num_decoding_left_chunks = -1
                required_cache_size = decoding_chunk_size * num_decoding_left_chunks
                output_chunk_probs = self.predictor.predict_chunk_conformer(x_chunk=x_chunk,
                                                                            required_cache_size=required_cache_size)
                output_lens = np.array([output_chunk_probs.shape[1]])
            else:
                raise Exception(f'当前模型不支持该方法，当前模型为：{self.configs.use_model}')
            # 执行解码
            if self.configs.decoder == 'ctc_beam_search':
                # 集束搜索解码策略
                score, text = self.beam_search_decoder.decode_chunk(probs=output_chunk_probs, logits_lens=output_lens)
            else:
                # 贪心解码策略
                score, text, self.greedy_last_max_prob_list, self.greedy_last_max_index_list = \
                    greedy_decoder_chunk(probs_seq=output_chunk_probs[0], vocabulary=self._text_featurizer.vocab_list,
                                         last_max_index_list=self.greedy_last_max_index_list,
                                         last_max_prob_list=self.greedy_last_max_prob_list)
        # 更新特征缓存
        self.cached_feat = self.cached_feat[:, end - cached_feature_num:, :]
        # 是否对文本进行反标准化
        if is_itn:
            text = self.inverse_text_normalization(text)

        result = {'text': text, 'score': score}
        return result

    # 重置流式识别，每次流式识别完成之后都要执行
    def reset_stream(self):
        self.predictor.reset_stream()
        self.remained_wav = None
        self.cached_feat = None
        self.greedy_last_max_prob_list = None
        self.greedy_last_max_index_list = None
        if self.configs.decoder == 'ctc_beam_search':
            self.beam_search_decoder.reset_decoder()

    # 对文本进行反标准化
    def inverse_text_normalization(self, text):
        if self.configs.decoder == 'ctc_beam_search':
            logger.error("当解码器为ctc_beam_search时，因为包冲突，不能使用文本反标准化")
            text = cn2an.transform(text, "cn2an")
            return text
        if self.inv_normalizer is None:
            # 需要安装WeTextProcessing>=0.1.0
            from itn.chinese.inverse_normalizer import InverseNormalizer
            self.inv_normalizer = InverseNormalizer()
        result_text = self.inv_normalizer.normalize(text)
        return result_text
