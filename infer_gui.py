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
        self.wav_path = None
        self.predicting = False
        self.recording = False
        self.stream = None
        self.is_itn = False
        # 录音参数
        self.frames = []
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
        # 输出结果文本框
        self.result_label = Label(self.window, text="输出日志：")
        self.result_label.place(x=10, y=70)
        self.result_text = Text(self.window, width=120, height=30)
        self.result_text.place(x=10, y=100)
        # 对文本进行反标准化
        self.an_frame = Frame(window)
        self.check_var = BooleanVar(value=False)
        self.is_itn_check = Checkbutton(self.an_frame, text='是否对文本进行反标准化', variable=self.check_var,
                                        command=self.is_itn_state)
        self.is_itn_check.grid(row=0)
        self.an_frame.grid(row=1)
        self.an_frame.place(x=400, y=10)

        # 获取识别器
        self.predictor = SRPredictor(configs=args.configs,
                                     model_path=args.model_path,
                                     use_gpu=args.use_gpu)

    # 是否对文本进行反标准化
    def is_itn_state(self):
        self.is_itn = self.check_var.get()

    # 预测短语音线程
    def predict_audio_thread(self):
        if not self.predicting:
            self.wav_path = askopenfilename(filetypes=[("音频文件", "*.wav"), ("音频文件", "*.mp3")],
                                            initialdir='./')
            if self.wav_path == '':
                return
            self.result_text.delete('1.0', 'end')
            self.result_text.insert(END, "已选择音频文件：%s\n" % self.wav_path)
            self.result_text.insert(END, "正在识别中...\n")
            _thread.start_new_thread(self.predict_audio, (self.wav_path,))
        else:
            tkinter.messagebox.showwarning('警告', '正在预测，请等待上一轮预测结束！')

    # 预测短语音
    def predict_audio(self, wav_file):
        self.predicting = True
        try:
            start = time.time()
            result = self.predictor.predict(audio_data=wav_file, is_itn=self.is_itn)
            score, text = result['score'], result['text']
            self.result_text.insert(END,
                                    f"消耗时间：{int(round((time.time() - start) * 1000))}ms, 识别结果: {text}, 得分: {score}\n")
        except Exception as e:
            self.result_text.insert(END, str(e))
            logger.error(e)
        self.predicting = False

    # 录音识别线程
    def record_audio_thread(self):
        if not self.recording:
            self.result_text.delete('1.0', 'end')
            _thread.start_new_thread(self.record_audio, ())
        else:
            self.recording = False

    def record_audio(self):
        self.record_button.configure(text='停止录音')
        self.recording = True
        self.frames = []
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
            self.frames.append(data)
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
        audio_bytes = b''.join(self.frames)
        # 保存音频数据
        os.makedirs(self.output_path, exist_ok=True)
        self.wav_path = os.path.join(self.output_path, f'{str(int(time.time()))}.wav')
        wf = wave.open(self.wav_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(audio_bytes)
        wf.close()
        self.recording = False
        self.result_text.insert(END, "录音已结束，录音文件保存在：%s\n" % self.wav_path)
        self.record_button.configure(text='录音识别')


if __name__ == '__main__':
    tk = Tk()
    myapp = SpeechRecognitionGUI(tk, args)
    tk.mainloop()
