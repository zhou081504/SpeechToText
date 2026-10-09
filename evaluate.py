"""独立进程比较 CPU/GPU 与 batch 参数，并以实测端到端吞吐估算资源。"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

from transcribe import positive_batch_size


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('必须为有限正数')
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('必须为有限非负数')
    return number


def estimate_capacity(speed, *, history_hours=0, deadline_days=30,
                      daily_audio_hours=0, hours_per_day=20, utilization=0.8):
    for value in (speed, deadline_days, hours_per_day, utilization):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('速度、期限、每日运行时间和利用率必须为有限正数')
    if hours_per_day > 24 or utilization > 1:
        raise ValueError('每日运行时间不能超过24小时，利用率不能超过1')
    if any(not math.isfinite(x) or x < 0 for x in (history_hours, daily_audio_hours)):
        raise ValueError('音频时长不能为负数或非有限值')
    capacity = speed * hours_per_day * utilization
    history_rate = history_hours / deadline_days
    return {
        'measured_realtime_speed': speed,
        'history_hours': history_hours,
        'deadline_days': deadline_days,
        'daily_audio_hours': daily_audio_hours,
        'hours_per_day': hours_per_day,
        'utilization': utilization,
        'audio_hours_per_unit_per_day': capacity,
        'units_for_history': math.ceil(history_rate / capacity),
        'units_for_daily': math.ceil(daily_audio_hours / capacity),
        'units_for_history_plus_daily': math.ceil((history_rate + daily_audio_hours) / capacity),
        'history_days_on_one_unit': history_hours / capacity,
        'all_history_pcm_tb': history_hours * 16000 * 2 * 3600 / 10**12,
        'scope': '同等硬件的独立转写节点；GPU配置为每节点一张同型号卡，CPU配置为同等CPU节点。未验证多节点线性扩展。',
        'storage_scope': '全部历史音频展开为16kHz/16bit/单声道PCM的十进制TB；当前程序逐文件清理，不代表必须配备同等临时盘。',
    }


def rank_results(results, expected_repeats=None):
    """仅完整成功、相同文件集的重复实验参与比较，采用倍速中位数。"""
    groups = {}
    for result in results:
        groups.setdefault((result['requested_device'], result['batch_size_s']), []).append(result)
    ranked = []
    for (device, batch), runs in groups.items():
        if expected_repeats is not None and len(runs) != expected_repeats:
            continue
        if not all(r.get('valid_for_comparison') and r.get('realtime_speed', 0) > 0 for r in runs):
            continue
        if len({r['files'] for r in runs}) != 1:
            continue
        audio = [r['audio_seconds'] for r in runs]
        if max(audio) - min(audio) > 0.01:
            continue
        ranked.append({
            'requested_device': device, 'device': runs[0]['device'], 'batch_size_s': batch,
            'median_realtime_speed': statistics.median(r['realtime_speed'] for r in runs),
            'repeat_count': len(runs),
            'gpu_name': runs[0]['resources'].get('gpu_name'),
            'cpu_model': runs[0]['resources'].get('cpu_model'),
        })
    return sorted(ranked, key=lambda x: x['median_realtime_speed'], reverse=True)


def write_reports(root, results, args):
    ranked = rank_results(results, args.repeats)
    plans = []
    for device in dict.fromkeys(r['requested_device'] for r in ranked):
        best = next(r for r in ranked if r['requested_device'] == device)
        plans.append({**best, **estimate_capacity(
            best['median_realtime_speed'], history_hours=args.history_hours,
            deadline_days=args.deadline_days, daily_audio_hours=args.daily_audio_hours,
            hours_per_day=args.hours_per_day, utilization=args.utilization,
        )})
    (root / '评测汇总.json').write_text(json.dumps(
        {'runs': results, 'ranked_configurations': ranked, 'capacity_plans': plans},
        ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for result in results:
        row = {k: v for k, v in result.items() if k not in ('resources', 'packages')}
        row.update(result.get('resources', {}))
        row['notes'] = '; '.join(row.get('notes', []))
        rows.append(row)
    if rows:
        with (root / '评测对比.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            fields = list(dict.fromkeys(key for row in rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 性能与算力评估', '',
             '速度包含每次独立进程的启动、模型加载、音频转换、VAD/ASR、保存及监控开销。每组重复实验按倍速中位数比较。', '',
             'CPU 利用率及系统内存属于整机指标；进程 RSS 不包含解码子进程。PyTorch 显存与整卡显存分别记录，采样峰值可能遗漏瞬时尖峰。', '',
             '仅全部文件成功且没有跳过的配置参与排名；失败、显存不足或设备不可用见各运行日志。', '',
             '| 设备 | batch_size_s | 重复次数 | 中位倍速 | 每小时音频耗时（分钟） |',
             '|---|---:|---:|---:|---:|']
    for row in ranked:
        speed = row['median_realtime_speed']
        lines.append(f"| {row['device']} | {row['batch_size_s']} | {row['repeat_count']} | {speed:.2f} | {60 / speed:.2f} |")
    if not ranked:
        lines += ['', '暂无可用于算力估算的完整成功实验。']
    for plan in plans:
        lines += ['', f"## {plan['device']} 最佳实测配置", '',
                  f"硬件：{plan['gpu_name'] or plan['cpu_model']}；batch_size_s={plan['batch_size_s']}。",
                  f"单节点每日处理能力：{plan['audio_hours_per_unit_per_day']:.2f} 小时音频（每天运行 {args.hours_per_day:g} 小时，利用率 {args.utilization:g}）。"]
        if args.history_hours or args.daily_audio_hours:
            lines += [f"历史音频 {args.history_hours:g} 小时 / {args.deadline_days:g} 天，新增音频 {args.daily_audio_hours:g} 小时/天。",
                      f"仅历史任务需要 {plan['units_for_history']} 个同等节点；仅日增需要 {plan['units_for_daily']} 个；同时处理需要 {plan['units_for_history_plus_daily']} 个。",
                      f"全部历史音频展开为 PCM 约 {plan['all_history_pcm_tb']:.3f} TB（不是当前流水处理的临时空间需求）。"]
        else:
            lines += ['未提供业务音频规模，仅报告单节点容量，不给出设备采购数量。']
    lines += ['', '公式：节点数 = ceil((历史小时数 / 完成天数 + 日增小时数) / (实测倍速 × 每日运行小时数 × 利用率))。', '',
              '结果只适用于实测同等硬件和代表性数据，不将当前GPU速度直接套用到其他型号。硬件扩展吞吐需另行验证；此评测覆盖解码、VAD、ASR和保存，不包含尚未实现的文本脱敏。']
    (root / '算力评估.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--devices', nargs='+', default=['auto'], help='例如 cpu cuda:0')
    parser.add_argument('--batch-sizes', nargs='+', type=positive_batch_size, default=[60, 120, 300, 600])
    parser.add_argument('--repeats', type=positive_batch_size, default=1)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / '评测结果')
    parser.add_argument('--history-hours', type=nonnegative_float, default=0)
    parser.add_argument('--deadline-days', type=positive_float, default=30)
    parser.add_argument('--daily-audio-hours', type=nonnegative_float, default=0)
    parser.add_argument('--hours-per-day', type=positive_float, default=20)
    parser.add_argument('--utilization', type=positive_float, default=0.8)
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error('输入必须为音频文件夹')
    if args.hours_per_day > 24 or args.utilization > 1:
        parser.error('每日运行时间必须 ≤24，利用率必须 ≤1')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='run_', dir=args.output_dir.resolve()))
    print(f'评测目录：{root}', flush=True)
    results = []
    input_dir = args.input_dir.resolve()
    # 每一轮确认路径、大小和修改时间相同，避免比较不同数据集。
    from transcribe import SUPPORTED_EXTENSIONS
    def signature():
        return [(str(p.relative_to(input_dir)), p.stat().st_size, p.stat().st_mtime_ns)
                for p in sorted(input_dir.rglob('*'))
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                and args.output_dir.resolve() not in p.parents]
    initial = signature()
    if not initial:
        parser.error('没有找到支持的音频文件')
    (root / '输入清单.json').write_text(json.dumps(initial, ensure_ascii=False, indent=2), encoding='utf-8')
    script = Path(__file__).resolve().with_name('transcribe.py')
    for device in dict.fromkeys(args.devices):
        for batch in dict.fromkeys(args.batch_sizes):
            for repeat in range(1, args.repeats + 1):
                if signature() != initial:
                    raise RuntimeError('输入音频在评测期间变化，停止比较；已有结果已保存')
                directory = root / f'case_{len(results) + 1:03d}'
                directory.mkdir()
                print(f'评测 {device} / batch_size_s={batch} / 第{repeat}次；日志：{directory / "运行.log"}', flush=True)
                command = [sys.executable, str(script), str(input_dir), '--device', device,
                           '--batch-size', str(batch), '--output-dir', str(directory), '--overwrite']
                started = time.perf_counter()
                with (directory / '运行.log').open('w', encoding='utf-8') as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        returncode = child.wait()
                    except KeyboardInterrupt:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise
                elapsed = time.perf_counter() - started
                result = {'requested_device': device, 'batch_size_s': batch, 'repeat': repeat,
                          'returncode': returncode, 'run_dir': str(directory), 'valid_for_comparison': False}
                summary_path = directory / '性能统计.json'
                if returncode == 0 and summary_path.is_file():
                    result.update(json.loads(summary_path.read_text(encoding='utf-8')))
                    result['transcription_wall_seconds'] = result['wall_seconds']
                    result['wall_seconds'] = elapsed
                    duration = result['audio_seconds']
                    result['realtime_speed'] = duration / elapsed if duration else None
                    result['rtf'] = elapsed / duration if duration else None
                    result['minutes_per_audio_hour'] = elapsed / duration * 60 if duration else None
                    result['timing_scope'] = '完整独立进程墙钟时间，含启动、导入、模型加载、转写及报告'
                    result['valid_for_comparison'] &= result['files'] == len(initial) and signature() == initial
                results.append(result)
                write_reports(root, results, args)
    print(f'评测完成：{root / "算力评估.md"}', flush=True)
    return root


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('评测已取消，已完成结果保留')
        raise SystemExit(130)
