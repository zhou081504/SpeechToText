import subprocess
import unittest
from unittest.mock import Mock, patch
from resource_monitor import ResourceMonitor, nvidia_smi_uuid


class GpuMonitorTests(unittest.TestCase):
    def test_uuid_prefix(self):
        raw = '7457666a-7a26-7651-ebe5-305eac6d1715'
        self.assertEqual(nvidia_smi_uuid(raw), 'GPU-' + raw)
        self.assertEqual(nvidia_smi_uuid('GPU-' + raw), 'GPU-' + raw)
        self.assertEqual(nvidia_smi_uuid('MIG-' + raw), 'MIG-' + raw)
        with self.assertRaises(ValueError):
            nvidia_smi_uuid('invalid')

    def test_start_uses_correct_uuid_and_samples(self):
        raw = '7457666a-7a26-7651-ebe5-305eac6d1715'
        torch = Mock()
        torch.cuda.get_device_properties.return_value = Mock(uuid=raw, total_memory=6 * 1024**3)
        monitor = ResourceMonitor('cuda:0')
        with patch.dict('sys.modules', {'torch': torch}), patch('resource_monitor.shutil.which', return_value='/usr/bin/nvidia-smi'), patch('resource_monitor.threading.Thread'), patch('resource_monitor.subprocess.run', return_value=Mock(stdout='80, 2048\n')) as run:
            monitor.start()
            self.assertEqual(run.call_args.args[0][2], 'GPU-' + raw)
            self.assertEqual(monitor.data['gpu_utilization_peak_percent'], 80)
            self.assertEqual(monitor.data['gpu_device_used_peak_gib'], 2)

    def test_sampling_error_preserves_diagnostic(self):
        monitor = ResourceMonitor('cuda:0')
        monitor.gpu_id = 'GPU-test'
        error = subprocess.CalledProcessError(6, 'nvidia-smi', stderr='GPU not found')
        with patch('resource_monitor.subprocess.run', side_effect=error):
            monitor.sample()
        self.assertIsNone(monitor.data['gpu_device_used_peak_gib'])
        self.assertIn('GPU not found', monitor.data['notes'][-1])
