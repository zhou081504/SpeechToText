import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from benchmark import Benchmark
from evaluate import estimate_capacity, rank_results, main as evaluate_main
from resource_monitor import ResourceMonitor
import transcribe


class PipelineTests(unittest.TestCase):
    def test_capacity(self):
        plan = estimate_capacity(30, history_hours=100000, deadline_days=30)
        self.assertEqual(plan['units_for_history'], 7)
        self.assertEqual(plan['audio_hours_per_unit_per_day'], 480)
        self.assertEqual(plan['all_history_pcm_tb'], 11.52)
        plan = estimate_capacity(30, daily_audio_hours=500)
        self.assertEqual(plan['units_for_daily'], 2)
        self.assertEqual(plan['units_for_history'], 0)
        for kwargs in ({'speed': 0}, {'speed': 30, 'utilization': 2},
                       {'speed': 30, 'hours_per_day': 25}, {'speed': float('nan')}):
            with self.assertRaises(ValueError):
                estimate_capacity(**kwargs)

    def test_cli_defaults_validation(self):
        self.assertEqual(transcribe.parse_args(['data']).batch_size, 1)
        self.assertEqual(transcribe.parse_args(['data', '--batch-size', '120']).batch_size, 120)
        with contextlib.redirect_stderr(io.StringIO()):
            for value in ('0', '-1', 'nan'):
                with self.assertRaises(SystemExit):
                    transcribe.parse_args(['data', '--batch-size', value])

    def test_disabled_does_not_monitor_time_or_sync(self):
        benchmark = Benchmark('cuda:0', enabled=False)
        with patch('benchmark.time.perf_counter', side_effect=AssertionError), patch(
            'resource_monitor.ResourceMonitor', side_effect=AssertionError
        ), patch.object(benchmark, '_sync', side_effect=AssertionError):
            benchmark.start_run()
            with benchmark.measure('process'):
                pass
            benchmark.finish_run()
            self.assertIsNone(benchmark.save_summary('/unused', batch_size=1, input_dir='data', counts={}))

    def test_rank_excludes_failed_group(self):
        base = dict(requested_device='cpu', device='cpu', batch_size_s=60,
                    valid_for_comparison=True, realtime_speed=30, files=2,
                    audio_seconds=100, resources={})
        rows = [base, {**base, 'realtime_speed': 10},
                {**base, 'batch_size_s': 120, 'valid_for_comparison': False}]
        ranked = rank_results(rows)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]['median_realtime_speed'], 20)
        self.assertEqual(rank_results(rows, expected_repeats=3), [])

    def test_transcription_success_skip_failure_and_cleanup(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / 'audio'
                source.mkdir()
                (source / 'test.wav').touch()
                output = root / 'output'
                args = transcribe.parse_args([str(source), '--output-dir', str(output), '--batch-size', '120'])
                model = Mock()
                model.generate.return_value = [{'text': '<|zh|>测试文字'}]
                factory = Mock(return_value=model)
                converted = []
                def convert(src, dest):
                    dest.touch()
                    converted.append(dest)
                with patch.dict('sys.modules', {'funasr': types.SimpleNamespace(AutoModel=factory)}), patch.object(
                    transcribe, 'check_command'
                ), patch.object(transcribe, 'convert_to_wav', side_effect=convert), patch.object(
                    transcribe, 'get_duration', return_value=60
                ), contextlib.redirect_stdout(io.StringIO()):
                    benchmark = Benchmark('cpu', enabled)
                    benchmark.start_run()
                    counts = transcribe.run_transcription(args, 'cpu', benchmark)
                    benchmark.finish_run()
                    summary = benchmark.save_summary(output / '性能统计.json', batch_size=120, input_dir=source, counts=counts)
                    self.assertEqual(counts['success'], 1)
                    self.assertEqual(model.generate.call_args.kwargs['batch_size_s'], 120)
                    self.assertEqual((output / 'test.wav.txt').read_text(), '测试文字')
                    self.assertFalse(converted[-1].exists())
                    if enabled:
                        self.assertTrue(summary['valid_for_comparison'])
                        self.assertGreater(summary['wall_seconds'], 0)
                    else:
                        self.assertFalse((output / '性能统计.json').exists())
                    self.assertEqual(transcribe.run_transcription(args, 'cpu', benchmark)['skipped'], 1)
                    args.overwrite = True
                    model.generate.side_effect = RuntimeError('模拟显存不足')
                    counts = transcribe.run_transcription(args, 'cpu', benchmark)
                    self.assertEqual(counts['failure'], 1)
                    with (output / '转写报告.csv').open(encoding='utf-8-sig') as file:
                        row = next(csv.DictReader(file))
                    self.assertEqual(row['状态'], '失败')
                    self.assertEqual(row['处理耗时_秒'] == '', not enabled)
                    self.assertFalse(converted[-1].exists())

    def test_monitor_reports_unavailable_gpu_as_null(self):
        monitor = ResourceMonitor('cpu').start()
        result = monitor.stop()
        self.assertIsNone(result['gpu_reserved_peak_gib'])
        self.assertFalse(monitor.thread.is_alive())
        self.assertGreater(result['process_rss_peak_gib'], 0)

    def test_evaluation_orchestration_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'audio'
            source.mkdir()
            (source / 'test.wav').touch()
            def launch(command, **kwargs):
                directory = Path(command[command.index('--output-dir') + 1])
                batch = int(command[command.index('--batch-size') + 1])
                summary = dict(device='cpu', batch_size_s=batch, files=1, success=1,
                               failure=0, skipped=0, audio_seconds=60, wall_seconds=2,
                               resources={'gpu_name': None, 'cpu_model': 'Test CPU'}, valid_for_comparison=True)
                (directory / '性能统计.json').write_text(json.dumps(summary))
                return Mock(wait=Mock(return_value=0))
            with patch('evaluate.subprocess.Popen', side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_main([str(source), '--devices', 'cpu', '--batch-sizes', '60', '120',
                                        '--output-dir', str(root / 'reports'), '--history-hours', '100000'])
            report = json.loads((result / '评测汇总.json').read_text())
            self.assertEqual(len(report['runs']), 2)
            self.assertEqual(len(report['capacity_plans']), 1)
            self.assertTrue((result / '评测对比.csv').is_file())
            self.assertTrue((result / '算力评估.md').is_file())


if __name__ == '__main__':
    unittest.main()
