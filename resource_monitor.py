"""无额外依赖的 Linux 资源采样；不可用指标返回 None，不伪造为零。"""
import os
import platform
import shutil
import subprocess
import threading
from uuid import UUID
from pathlib import Path


GIB = 1024 ** 3


def nvidia_smi_uuid(value):
    """PyTorch 可返回不带 GPU- 前缀的 UUID；保留完整 GPU/MIG 标识。"""
    identifier = str(value).strip()
    if identifier.startswith(("GPU-", "MIG-")):
        return identifier
    return "GPU-" + str(UUID(identifier))


class ResourceMonitor:
    def __init__(self, device, interval=0.5):
        self.device = device
        self.interval = interval
        self.event = threading.Event()
        self.thread = None
        self.previous_cpu = None
        self.gpu_id = None
        self.data = {
            'cpu_model': platform.processor() or platform.machine(),
            'cpu_logical_count': os.cpu_count(),
            'system_cpu_peak_percent': None,
            'process_rss_peak_gib': None,
            'system_memory_total_gib': None,
            'system_memory_used_peak_gib': None,
            'gpu_name': None,
            'gpu_total_gib': None,
            'gpu_allocated_peak_gib': None,
            'gpu_reserved_peak_gib': None,
            'gpu_device_used_peak_gib': None,
            'gpu_utilization_peak_percent': None,
            'sample_interval_seconds': interval,
            'notes': [],
        }

    def note(self, message):
        if message not in self.data['notes']:
            self.data['notes'].append(message)

    def peak(self, key, value):
        previous = self.data[key]
        self.data[key] = value if previous is None else max(previous, value)

    def sample(self):
        try:
            values = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
            total, idle = sum(values), values[3] + values[4]
            if self.previous_cpu:
                dt = total - self.previous_cpu[0]
                if dt > 0:
                    self.peak('system_cpu_peak_percent', 100 * (1 - (idle - self.previous_cpu[1]) / dt))
            self.previous_cpu = total, idle
            status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
            self.peak('process_rss_peak_gib', int(status['VmHWM'].split()[0]) * 1024 / GIB)
            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            total_memory = int(memory['MemTotal'].split()[0]) * 1024
            available = int(memory['MemAvailable'].split()[0]) * 1024
            self.data['system_memory_total_gib'] = total_memory / GIB
            self.peak('system_memory_used_peak_gib', (total_memory - available) / GIB)
        except (OSError, KeyError, ValueError, IndexError) as error:
            self.note(f'Linux CPU/内存采样不可用：{error}')
        if self.gpu_id:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '-i', self.gpu_id,
                     '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=2, check=True,
                )
                utilization, used = map(float, result.stdout.strip().split(','))
                self.peak('gpu_utilization_peak_percent', utilization)
                self.peak('gpu_device_used_peak_gib', used / 1024)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                detail = (getattr(error, 'stderr', None) or getattr(error, 'stdout', None) or '').strip()
                self.note(f'整卡 GPU 采样不可用：{error}' + (f'；{detail}' if detail else ''))
                self.gpu_id = None

    def start(self):
        try:
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    self.data['cpu_model'] = line.split(':', 1)[1].strip()
                    break
        except OSError:
            pass
        if self.device.startswith('cuda'):
            import torch
            props = torch.cuda.get_device_properties(self.device)
            self.data['gpu_name'] = props.name
            self.data['gpu_total_gib'] = props.total_memory / GIB
            torch.cuda.reset_peak_memory_stats(self.device)
            # UUID 避免 CUDA_VISIBLE_DEVICES 导致采错物理卡。
            uuid = getattr(props, 'uuid', None)
            if uuid and shutil.which('nvidia-smi'):
                try:
                    self.gpu_id = nvidia_smi_uuid(uuid)
                except ValueError:
                    self.note(f'无法识别 GPU UUID：{uuid}')
            else:
                self.note('无法按 UUID 采样整卡利用率；仍记录 PyTorch 显存峰值')
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def _loop(self):
        while not self.event.wait(self.interval):
            self.sample()

    def stop(self):
        self.event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        if self.device.startswith('cuda'):
            import torch
            self.data['gpu_allocated_peak_gib'] = torch.cuda.max_memory_allocated(self.device) / GIB
            self.data['gpu_reserved_peak_gib'] = torch.cuda.max_memory_reserved(self.device) / GIB
        return self.data
