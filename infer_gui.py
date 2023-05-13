import _thread
import argparse
import functools
import os
import time
import tkinter.messagebox
import wave
from tkinter import *
from tkinter.filedialog import askopenfilename

import pyaudio

from src.predict import SRPredictor
from src.utils.utils import add_arguments, print_arguments
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs', str, 'configs/conformer.yml', "配置文件")
add_arg('use_gpu', bool, False, "是否使用GPU预测")
add_arg('model_path', str, 'models/conformer_streaming_fbank/inference.pt', "导出的预测模型文件路径")
args = parser.parse_args()
print_arguments(args=args)


class SpeechRecognitionGUI:
    def __init__(self, window: Tk, args):
        self.window = window
        self.predicting = False
        self.recording = False
        self.stream = None
        self.playing = False
        self.is_itn = False
        self.is_streaming = False
        # 录音参数
        interval_time = 0.5
        self.CHUNK = int(16000 * interval_time)
        # 最大录音时长
        self.max_record = 600
        # 录音保存的路径
        self.output_path = './record'
        # 创建一个播放器
        self.p = pyaudio.PyAudio()
        # 指定窗口标题
        self.window.title("语音识别GUI")
        # 固定窗口大小
        self.window.geometry('640x400')
        self.window.resizable(False, False)
        # 识别短语音按钮
        self.short_button = Button(self.window, text="选择短语音识别", width=20, command=self.predict_audio_thread)
        self.short_button.place(x=10, y=10)
        # 录音按钮
        self.record_button = Button(self.window, text="录音识别", width=20, command=self.record_audio_thread)
        self.record_button.place(x=180, y=10)
        # 播放音频按钮
        self.play_button = Button(self.window, text="播放音频", width=20, command=self.play_audio_thread)
        self.play_button.place(x=350, y=10)
        # 输出结果文本框
        self.result_label = Label(self.window, text="输出日志：")
        self.result_label.place(x=10, y=70)
        self.result_text = Text(self.window, width=80, height=20)
        self.result_text.place(x=10, y=100)
        # 对文本进行反标准化
        self.check_var = BooleanVar(value=False)
        self.inverse_norm_check = Checkbutton(self.window, text='是否对文本进行反标准化', variable=self.check_var,
                                              command=self.inverse_norm_state)
        self.inverse_norm_check.place(x=10, y=40)
        # 流式预测
        self.stream_var = BooleanVar(value=False)
        self.stream_check = Checkbutton(window, text='是否使用流式预测', variable=self.stream_var,
                                        command=self.streaming_state)
        self.stream_check.place(x=200, y=40)

        # 获取识别器
        self.predictor = SRPredictor(configs=args.configs,
                                     model_path=args.model_path,
                                     use_gpu=args.use_gpu)

    # 是否对文本进行反标准化
    def inverse_norm_state(self):
        self.is_itn = self.check_var.get()

    # 是否模拟流式预测
    def streaming_state(self):
        self.is_streaming = self.stream_var.get()

    # 预测短语音线程
    def predict_audio_thread(self):
        if not self.predicting:
            wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir='./')
            if wav_path == '':
                return
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, "已选择音频文件：%s\n" % wav_path)
            self.result_text.insert(END, "正在识别中...\n")
            _thread.start_new_thread(self.predict_audio, (wav_path,))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测短语音
    def predict_audio(self, wav_path):
        self.predicting = True
        try:
            if self.is_streaming:
                wf = wave.open(wav_path, 'rb')
                data = wf.readframes(self.CHUNK)
                # 播放
                while data != b'':
                    d = wf.readframes(self.CHUNK)
                    result = self.predictor.predict_stream(audio_data=data, is_itn=self.is_itn, is_end=d == b'')
                    data = d
                    if result is None:
                        continue
                    score, text = result['score'], result['text']
                    self.result_text.delete('1.0', 'end')
                    self.result_text.insert(END, f"{text}\n")
                self.predictor.reset_stream()
            else:
                start = time.time()
                result = self.predictor.predict(audio_data=wav_path, is_itn=self.is_itn)
                score, text = result['score'], result['text']
                self.result_text.insert(END,
                                        f"消耗时间：{int(round((time.time() - start) * 1000))}ms,\n"
                                        f"识别结果: {text},\n"
                                        f"得分: {score}")
        except Exception as e:
            self.result_text.insert(END, str(e))
            logger.error(e)
        self.predicting = False

    # 录音识别线程
    def record_audio_thread(self):
        if not self.playing and not self.recording:
            self.result_text.delete('1.0', 'end')
            _thread.start_new_thread(self.record_audio, ())
        else:
            if self.playing:
                tkinter.messagebox.showwarning('警告', '正在播放音频，无法录音！')
            else:
                # 停止录音
                self.recording = False

    def record_audio(self):
        self.record_button.configure(text='停止录音')
        self.recording = True
        frames = []
        FORMAT = pyaudio.paInt16
        channels = 1
        rate = 16000

        # 打开录音
        self.stream = self.p.open(format=FORMAT,
                                  channels=channels,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=self.CHUNK)
        self.result_text.insert(END, "正在录音...\n")

        while True:
            data = self.stream.read(self.CHUNK)
            frames.append(data)
            result = self.predictor.predict_stream(audio_data=data, is_itn=self.is_itn, is_end=not self.recording)
            if result is None:
                continue
            score, text = result['score'], result['text']
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, f"{text}\n")
            if not self.recording:
                break
        self.predictor.reset_stream()

        # 录音的字节数据，用于后面的预测和保存
        audio_bytes = b''.join(frames)
        # 保存音频数据
        os.makedirs(self.output_path, exist_ok=True)
        wav_path = os.path.join(self.output_path, f'{str(int(time.time()))}.wav')
        wf = wave.open(wav_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(audio_bytes)
        wf.close()
        self.recording = False
        self.result_text.insert(END, "录音已结束，录音文件保存在：%s\n" % wav_path)
        self.record_button.configure(text='录音识别')

    # 播放音频线程
    def play_audio_thread(self):
        if not self.playing and not self.recording:
            wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")], initialdir='./')
            if wav_path == '':
                tkinter.messagebox.showwarning('警告', '音频路径为空！')
            _thread.start_new_thread(self.play_audio, (wav_path,))
        else:
            if self.recording:
                tkinter.messagebox.showwarning('警告', '正在录音，无法播放音频！')
            else:
                # 停止播放
                self.playing = False

    # 播放音频
    def play_audio(self, wav_path):
        self.play_button.configure(text='停止播放')
        self.playing = True
        CHUNK = 1024
        wf = wave.open(wav_path, 'rb')
        # 打开数据流
        self.stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
                                  channels=wf.getnchannels(),
                                  rate=wf.getframerate(),
                                  output=True)
        # 读取数据
        data = wf.readframes(CHUNK)
        # 播放
        while data != b'':
            if not self.playing:
                break
            self.stream.write(data)
            data = wf.readframes(CHUNK)
        # 停止数据流
        self.stream.stop_stream()
        self.stream.close()
        self.playing = False
        self.play_button.configure(text='播放音频')


if __name__ == '__main__':
    tk = Tk()
    myapp = SpeechRecognitionGUI(tk, args)
    tk.mainloop()
