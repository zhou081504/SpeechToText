import csv
import math
import re
import shutil
import subprocess
import argparse
import tempfile
from pathlib import Path


from benchmark import Benchmark


SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".amr",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
}


def check_command(command):
    if shutil.which(command) is None:
        raise RuntimeError(
            f"找不到 {command}，请先安装：\n"
            "conda install -c conda-forge ffmpeg -y"
        )


def convert_to_wav(source, destination):
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "未知音频解码错误")


def get_duration(audio):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "无法读取音频时长")
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError("音频时长为零")
    return duration


def clean_text(results):
    parts = []
    for item in results:
        text = item.get("text", "")
        text = re.sub(r"<\|[^>]+?\|>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


# FunASR 的 VAD 批处理按音频时长（秒）设置预算。
DEFAULT_BATCH_SIZE = 1


def positive_batch_size(value):
    try:
        size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("batch_size 必须为正整数（秒）") from error
    if size < 1:
        raise argparse.ArgumentTypeError("batch_size 必须为正整数（秒）")
    return size


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="音频转写")
    parser.add_argument("input_dir", help="输入音频文件夹路径")
    parser.add_argument("--no-benchmark", action="store_true", help="关闭测速统计")
    parser.add_argument(
        "--batch-size", "--batch_size", type=positive_batch_size,
        default=DEFAULT_BATCH_SIZE,
        help="模型批处理时长预算（秒），默认 1，对应 FunASR batch_size_s",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda:0 等 CUDA 设备")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "转写结果")
    parser.add_argument("--overwrite", action="store_true", help="重新处理已有结果，用于完整测速")
    return parser.parse_args(argv)


def resolve_device(requested):
    import torch
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return requested
    if requested == "cuda":
        requested = "cuda:0"
    if not re.fullmatch(r"cuda:\d+", requested):
        raise ValueError("设备必须为 auto、cpu 或 cuda:0 等")
    if not torch.cuda.is_available() or int(requested.split(":")[1]) >= torch.cuda.device_count():
        raise ValueError(f"指定 CUDA 设备不可用：{requested}")
    return requested


def run_transcription(args, device, benchmark):
    from funasr import AutoModel
    input_dir = Path(args.input_dir).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"找不到文件夹：{input_dir}")

    check_command("ffmpeg")
    check_command("ffprobe")

    output_dir = args.output_dir.expanduser().resolve()
    audio_files = sorted(
        file
        for file in input_dir.rglob("*")
        if file.is_file()
        and output_dir not in file.parents
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not audio_files:
        raise SystemExit("文件夹中没有找到支持的音频文件")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"找到 {len(audio_files)} 个文件")
    print(f"使用设备：{device}")
    print(f"模型 batch_size_s：{args.batch_size} 秒")
    print("正在加载模型……")

    benchmark.start("load")
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        hub="ms",
        device=device,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
    )
    benchmark.stop("load")
    benchmark.print_load()

    records = []
    success_count = 0
    failure_count = 0
    skipped_count = 0

    with tempfile.TemporaryDirectory(prefix="sensevoice_batch_") as temp_name:
        temp_dir = Path(temp_name)

        for index, audio in enumerate(audio_files, start=1):
            relative = audio.relative_to(input_dir)
            relative_output = relative.with_suffix(relative.suffix + ".txt")
            output_file = output_dir / relative_output

            print(f"[{index}/{len(audio_files)}] 检查文件：{relative}")

            if not args.overwrite and output_file.is_file() and output_file.stat().st_size > 0:
                skipped_count += 1
                records.append(
                    {
                        "文件": str(relative),
                        "状态": "跳过",
                        "音频时长_秒": "",
                        "处理耗时_秒": "0" if benchmark.enabled else "",
                        "处理倍速": "",
                        "输出文件": str(output_file),
                        "错误": "",
                    }
                )
                print(f"已有转写结果，跳过：{relative_output}\n")
                continue

            benchmark.start("process")
            temporary_output = output_file.with_suffix(output_file.suffix + ".tmp")

            temp_wav = temp_dir / f"audio_{index}.wav"
            try:
                convert_to_wav(audio, temp_wav)
                duration = get_duration(temp_wav)

                results = model.generate(
                    input=str(temp_wav),
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=args.batch_size,
                    merge_vad=True,
                    merge_length_s=15,
                )
                text = clean_text(results)
                if not text:
                    raise RuntimeError("没有识别到有效文字")

                output_file.parent.mkdir(parents=True, exist_ok=True)
                temporary_output.write_text(text, encoding="utf-8")
                temporary_output.replace(output_file)

                elapsed = benchmark.stop("process")
                success_count += 1
                benchmark.record_success(duration, elapsed)
                records.append(
                    {
                        "文件": str(relative),
                        "状态": "成功",
                        "音频时长_秒": f"{duration:.2f}",
                        **benchmark.record_fields(duration, elapsed),
                        "输出文件": str(output_file),
                        "错误": "",
                    }
                )
                benchmark.print_file(duration, elapsed)
            except Exception as error:
                elapsed = benchmark.stop("process")
                failure_count += 1
                if temporary_output.exists():
                    temporary_output.unlink()
                records.append(
                    {
                        "文件": str(relative),
                        "状态": "失败",
                        "音频时长_秒": "",
                        **benchmark.record_fields(elapsed=elapsed),
                        "输出文件": "",
                        "错误": str(error),
                    }
                )
                print(f"失败：{error}\n")
            finally:
                # 逐文件释放 PCM 临时文件，避免历史数据全部展开占满磁盘。
                temp_wav.unlink(missing_ok=True)

    report_file = output_dir / "转写报告.csv"
    with report_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "文件",
                "状态",
                "音频时长_秒",
                "处理耗时_秒",
                "处理倍速",
                "输出文件",
                "错误",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print("========== 批量任务完成 ==========")
    print(f"本次成功：{success_count}")
    print(f"已有结果跳过：{skipped_count}")
    print(f"失败：{failure_count}")
    print(f"结果目录：{output_dir}")
    print(f"处理报告：{report_file}")

    benchmark.print_batch()
    return {"files": len(audio_files), "success": success_count,
            "failure": failure_count, "skipped": skipped_count}


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    benchmark = Benchmark(device, enabled=not args.no_benchmark)
    benchmark.start_run()
    try:
        counts = run_transcription(args, device, benchmark)
    finally:
        benchmark.finish_run()
    benchmark.save_summary(
        args.output_dir.expanduser().resolve() / "性能统计.json",
        batch_size=args.batch_size,
        input_dir=Path(args.input_dir).expanduser().resolve(), counts=counts,
    )
    return counts


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n任务已手动停止")
        raise SystemExit(130)
    except Exception as error:
        print(f"\n程序失败：{error}")
        raise SystemExit(1)
