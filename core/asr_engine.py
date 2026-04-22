import os
import queue
import time
import sounddevice as sd
import sherpa_onnx
import keyboard
import numpy as np


class SentiaEar:
    def __init__(self, base_dir):
        print(" 正在加载 Sherpa-ONNX 听觉模型...")
        model_dir = os.path.join(base_dir, "models", "asr")
        self.push_to_talk_key = os.getenv("SENTIA_PUSH_TO_TALK_KEY", "space").strip().lower() or "space"
        self.recognizer_sample_rate = 16000
        self.input_gain = max(1.0, float(os.getenv("SENTIA_ASR_INPUT_GAIN", "3.0")))
        self.max_auto_gain = max(self.input_gain, float(os.getenv("SENTIA_ASR_MAX_GAIN", "8.0")))
        self.target_peak = min(0.95, max(0.1, float(os.getenv("SENTIA_ASR_TARGET_PEAK", "0.75"))))
        self.noise_floor = max(0.0, float(os.getenv("SENTIA_ASR_NOISE_FLOOR", "0.01")))
        self.input_device = self._resolve_input_device()

        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=os.path.join(model_dir, "tokens.txt"),
                encoder=os.path.join(model_dir, "encoder-epoch-99-avg-1.onnx"),
                decoder=os.path.join(model_dir, "decoder-epoch-99-avg-1.onnx"),
                joiner=os.path.join(model_dir, "joiner-epoch-99-avg-1.onnx"),
                num_threads=2,
                sample_rate=self.recognizer_sample_rate,
                feature_dim=80,
                rule1_min_trailing_silence=2.4,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=30.0,
                provider="cpu"
            )
            print(" ASR 加载成功！")
        except Exception as e:
            print(f" ASR 加载失败！请检查 models 目录。\n报错: {e}")
            self.recognizer = None

        self.stream_sample_rate = self._select_stream_sample_rate()
        self.audio_queue = queue.Queue()
        if self.input_device is not None:
            try:
                device_name = sd.query_devices(self.input_device)["name"]
                print(f" [ASR] 当前录音设备: {device_name} | 采样率: {self.stream_sample_rate}Hz | 按键: {self.push_to_talk_key.upper()}")
            except Exception:
                pass

    def _resolve_input_device(self):
        requested = os.getenv("SENTIA_AUDIO_INPUT_DEVICE", "").strip()
        devices = sd.query_devices()

        if requested:
            try:
                return int(requested)
            except ValueError:
                lowered = requested.lower()
                for index, device in enumerate(devices):
                    if device.get("max_input_channels", 0) <= 0:
                        continue
                    if lowered in device["name"].lower():
                        return index

        def is_generic_device(device_name):
            lowered = device_name.lower()
            return (
                "microsoft 声音映射器" in lowered
                or "主声音捕获驱动程序" in lowered
                or "sound mapper" in lowered
                or "primary sound capture driver" in lowered
            )

        default_device = sd.default.device
        if isinstance(default_device, (list, tuple)) and len(default_device) > 0 and default_device[0] is not None:
            default_index = int(default_device[0])
            try:
                if not is_generic_device(devices[default_index]["name"]):
                    return default_index
            except Exception:
                pass

        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0 and not is_generic_device(device["name"]):
                return index

        if isinstance(default_device, (list, tuple)) and len(default_device) > 0 and default_device[0] is not None:
            return int(default_device[0])
        return None

    def _select_stream_sample_rate(self):
        try:
            kwargs = {"samplerate": self.recognizer_sample_rate, "channels": 1, "dtype": "float32"}
            if self.input_device is not None:
                kwargs["device"] = self.input_device
            sd.check_input_settings(**kwargs)
            return self.recognizer_sample_rate
        except Exception:
            pass

        if self.input_device is not None:
            try:
                device_info = sd.query_devices(self.input_device)
                fallback_rate = int(device_info.get("default_samplerate", self.recognizer_sample_rate))
                if fallback_rate > 0:
                    print(f" [ASR Warning] 当前麦克风不支持 {self.recognizer_sample_rate}Hz，改用 {fallback_rate}Hz 并在本地重采样。")
                    return fallback_rate
            except Exception:
                pass
        return self.recognizer_sample_rate

    def _resample_chunk_if_needed(self, chunk):
        if self.stream_sample_rate == self.recognizer_sample_rate or chunk.size == 0:
            return chunk

        new_len = max(1, int(round(len(chunk) * self.recognizer_sample_rate / self.stream_sample_rate)))
        if new_len == len(chunk):
            return chunk

        source_positions = np.linspace(0.0, 1.0, len(chunk), dtype=np.float32)
        target_positions = np.linspace(0.0, 1.0, new_len, dtype=np.float32)
        return np.interp(target_positions, source_positions, chunk).astype(np.float32, copy=False)

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = np.array(indata[:, 0], dtype=np.float32, copy=True)
        if chunk.size > 0:
            chunk = self._resample_chunk_if_needed(chunk)
            peak = float(np.max(np.abs(chunk)))
            gain = self.input_gain
            if peak > self.noise_floor:
                gain = min(self.max_auto_gain, max(self.input_gain, self.target_peak / max(peak, 1e-6)))
            chunk *= gain
            np.clip(chunk, -1.0, 1.0, out=chunk)
        self.audio_queue.put(chunk)

    def _clear_audio_queue(self):
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    def _get_stream_text(self, stream):
        text = self.recognizer.get_result(stream)
        return text.strip() if text else ""

    def listen(self, stop_event=None):
        if not self.recognizer:
            return ""

        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return ""
                if keyboard.is_pressed(self.push_to_talk_key):
                    break
                time.sleep(0.05)

            self._clear_audio_queue()
            print(f"\n [录音中... 请保持按住 {self.push_to_talk_key.upper()}，松开发送]")
            stream = self.recognizer.create_stream()
            last_text = ""

            stream_kwargs = {
                "channels": 1,
                "dtype": "float32",
                "samplerate": self.stream_sample_rate,
                "callback": self._audio_callback,
            }
            if self.input_device is not None:
                stream_kwargs["device"] = self.input_device

            with sd.InputStream(**stream_kwargs):
                while keyboard.is_pressed(self.push_to_talk_key):
                    if stop_event is not None and stop_event.is_set():
                        break
                    try:
                        chunk = self.audio_queue.get(timeout=0.05)
                        stream.accept_waveform(self.recognizer_sample_rate, chunk)
                        while self.recognizer.is_ready(stream):
                            self.recognizer.decode_stream(stream)

                        current_text = self._get_stream_text(stream)
                        if current_text and current_text != last_text:
                            print(f"\r 听到: {current_text}", end="", flush=True)
                            last_text = current_text
                    except queue.Empty:
                        time.sleep(0.001)

            if stop_event is not None and stop_event.is_set():
                return ""

            print("\n [松开按键，录音结束]")
            stream.input_finished()
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)

            return self._get_stream_text(stream)
        except Exception as e:
            print(f"\n[ASR Warning] 语音输入失败，已回退为键盘输入: {e}")
            return ""
