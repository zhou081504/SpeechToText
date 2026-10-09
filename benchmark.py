"""可关闭的转写测速；关闭后不读取计时器，也不执行测速用的 GPU 同步。"""

import time
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from contextlib import contextmanager


class Benchmark:
    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self.timings = {}
        self.total_audio_seconds = 0.0
        self.total_process_seconds = 0.0
        self.monitor = None
        self.resources = {}

    def _sync(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)

    def start(self, name):
        if self.enabled:
            self._sync()
            self.timings[name] = time.perf_counter()

    def stop(self, name):
        if not self.enabled:
            return None
        if name not in self.timings:
            return self.timings.get(name + "_seconds")
        self._sync()
        elapsed = time.perf_counter() - self.timings.pop(name)
        self.timings[name + "_seconds"] = elapsed
        return elapsed

    @contextmanager
    def measure(self, name):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    @staticmethod
    def speed(duration, elapsed):
        return duration / elapsed if elapsed > 0 else 0.0

    def record_success(self, duration, elapsed):
        if self.enabled:
            self.total_audio_seconds += duration
            self.total_process_seconds += elapsed

    def record_fields(self, duration=None, elapsed=None):
        return {
            "处理耗时_秒": f"{elapsed:.2f}" if self.enabled and elapsed is not None else "",
            "处理倍速": f"{self.speed(duration, elapsed):.2f}"
            if self.enabled and duration is not None and elapsed is not None else "",
        }

    def print_load(self):
        if self.enabled:
            print(f"模型加载完成，耗时 {self.timings['load_seconds']:.2f} 秒\n")
        else:
            print("模型加载完成\n")

    def print_file(self, duration, elapsed):
        if self.enabled:
            print(f"完成：音频 {duration:.1f} 秒，耗时 {elapsed:.1f} 秒，"
                  f"{self.speed(duration, elapsed):.2f} 倍实时\n")
        else:
            print(f"完成：音频 {duration:.1f} 秒\n")

    def print_single(self, duration):
        if not self.enabled:
            return
        load = self.timings['load_seconds']
        convert = self.timings['convert_seconds']
        transcribe = self.timings['transcribe_seconds']
        print("\n========== 性能统计 ==========")
        print(f"选择的设备：{self.device}")
        print(f"模型加载耗时：{load:.2f} 秒")
        print(f"音频转换耗时：{convert:.2f} 秒")
        print(f"模型转写耗时：{transcribe:.2f} 秒")
        print(f"转换加转写：{convert + transcribe:.2f} 秒")
        print(f"RTF：{transcribe / duration:.4f}")
        print(f"模型处理速度：{self.speed(duration, transcribe):.2f} 倍实时")
        print(f"1 小时音频预计转写耗时：{transcribe / duration * 60:.2f} 分钟")
        print(f"1 小时音频预计处理耗时（转换加转写）：{(convert + transcribe) / duration * 60:.2f} 分钟")

    def print_batch(self):
        if not self.enabled:
            return
        print("\n========== 性能统计 ==========")
        print(f"选择的设备：{self.device}")
        duration = self.total_audio_seconds
        elapsed = self.total_process_seconds
        if duration <= 0 or elapsed <= 0:
            print("没有成功处理的音频，无法估算每小时音频耗时")
            return
        print(f"本次音频总时长：{duration / 3600:.2f} 小时")
        print(f"本次处理总耗时：{elapsed / 60:.2f} 分钟")
        print(f"综合处理速度：{self.speed(duration, elapsed):.2f} 倍实时")
        print(f"综合 RTF：{elapsed / duration:.4f}")
        print(f"1 小时音频预计处理耗时：{elapsed / duration * 60:.2f} 分钟")


    def start_run(self):
        if not self.enabled:
            return
        from resource_monitor import ResourceMonitor
        self.monitor = ResourceMonitor(self.device).start()
        self.start("run")

    def finish_run(self):
        if not self.enabled:
            return
        try:
            self.stop("run")
        finally:
            if self.monitor:
                self.resources = self.monitor.stop()

    def save_summary(self, path, *, batch_size, input_dir, counts):
        if not self.enabled:
            return None
        elapsed = self.timings['run_seconds']
        duration = self.total_audio_seconds
        speed = duration / elapsed if elapsed > 0 and duration > 0 else None
        packages = {}
        for name in ('torch', 'funasr', 'modelscope'):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = None
        summary = {
            'schema_version': 1,
            'device': self.device,
            'model': 'iic/SenseVoiceSmall',
            'vad_model': 'fsmn-vad',
            'merge_vad': True,
            'merge_length_s': 15,
            'batch_size_s': batch_size,
            'input_dir': str(input_dir),
            **counts,
            'audio_seconds': duration,
            'wall_seconds': elapsed,
            'model_load_seconds': self.timings.get('load_seconds'),
            'successful_process_seconds': self.total_process_seconds,
            'realtime_speed': speed,
            'rtf': elapsed / duration if duration > 0 else None,
            'minutes_per_audio_hour': 60 / speed if speed else None,
            'valid_for_comparison': counts['success'] == counts['files'] and counts['files'] > 0,
            'resources': self.resources,
            'python': platform.python_version(),
            'packages': packages,
            'timing_scope': '包含模型加载、转换、VAD/ASR、文本与CSV写入、失败任务及循环开销；不包含评测子进程启动及模块导入',
        }
        path = Path(path)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"端到端总耗时：{elapsed / 60:.2f} 分钟（包含模型加载）")
        if speed:
            print(f"端到端倍速：{speed:.2f}；每小时音频预计耗时：{60 / speed:.2f} 分钟")
        for key, label in [('system_cpu_peak_percent', '系统 CPU 峰值（%）'),
                           ('process_rss_peak_gib', '转写进程内存峰值（GiB）'),
                           ('gpu_reserved_peak_gib', 'PyTorch 预留显存峰值（GiB）')]:
            value = self.resources.get(key)
            print(f"{label}：{value:.2f}" if value is not None else f"{label}：不适用或不可用")
        print(f"性能报告：{path}")
        return summary
