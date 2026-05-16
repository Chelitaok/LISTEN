"""
Transcribe audio with WhisperX (large-v3) and diarize speakers with pyannote.

Usage:
    conda activate listen
    python src/asr/transcribe.py -i "data/audio/Photovoice categorización-1.m4a" --hf-token YOUR_TOKEN

    # Or store the token in a .env file (HF_TOKEN=xxx) and omit --hf-token.

Outputs (in data/transcripts/):
    <stem>.json   — full structured data
    <stem>.txt    — human-readable transcript, easy to relabel speakers
    <stem>.csv    — tabular with empty 'speaker_label' column for manual annotation
"""

import csv
import json
import logging
import warnings
from datetime import timedelta
from pathlib import Path

import click
import torch
import yaml


def _suppress_noise():
    warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")


def format_ts(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--input", "-i", "audio_path", required=True, type=click.Path(exists=True),
              help="Path to audio file (.m4a, .mp3, .wav, etc.)")
@click.option("--output-dir", "-o", default="data/transcripts", type=click.Path(),
              help="Directory for output files", show_default=True)
@click.option("--hf-token", envvar="HF_TOKEN", required=True,
              help="HuggingFace token for pyannote models (or set HF_TOKEN in .env)")
@click.option("--num-speakers", default=7, show_default=True,
              help="Total number of distinct speakers in the recording")
@click.option("--diarize-model", default="pyannote/speaker-diarization-3.1", show_default=True,
              help="Pyannote diarization model. Use 'pyannote/speaker-diarization-community-1' if 3.1 terms not yet accepted.")
@click.option("--config", "config_path", default="config/asr.yaml",
              type=click.Path(exists=True), show_default=True)
def transcribe(audio_path, output_dir, hf_token, num_speakers, config_path, diarize_model):
    """Transcribe audio and segment by speaker using WhisperX + pyannote."""
    _suppress_noise()

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    cfg = load_config(Path(config_path))
    input_path = Path(audio_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CTranslate2 (faster-whisper) runs on cpu or cuda, not mps.
    # PyTorch-based pyannote can use mps for faster diarization on M1.
    whisper_device = "cpu"
    diarize_device = "mps" if torch.backends.mps.is_available() else "cpu"
    compute_type = "int8"

    log.info(f"Audio       : {input_path.name}")
    log.info(f"Whisper     : large-v3 on {whisper_device} ({compute_type})")
    log.info(f"Diarization : pyannote on {diarize_device}")
    log.info(f"Speakers    : {num_speakers}")

    import whisperx
    from whisperx.diarize import DiarizationPipeline

    # ── Step 1: Transcribe ─────────────────────────────────────────────────
    log.info("Step 1/3  Transcribing... (this takes a while for long audio)")
    model = whisperx.load_model(
        "large-v3", whisper_device, compute_type=compute_type, language="es"
    )
    audio = whisperx.load_audio(str(input_path))
    result = model.transcribe(audio, language="es", batch_size=8)
    del model
    log.info(f"          → {len(result['segments'])} raw segments")

    # ── Step 2: Word-level alignment ───────────────────────────────────────
    log.info("Step 2/3  Aligning word timestamps...")
    align_model, metadata = whisperx.load_align_model(
        language_code="es", device=whisper_device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio,
        whisper_device, return_char_alignments=False
    )
    del align_model

    # ── Step 3: Speaker diarization ────────────────────────────────────────
    log.info("Step 3/3  Running speaker diarization...")
    log.info(f"          Using model: {diarize_model}")
    diarize_pipeline = DiarizationPipeline(
        model_name=diarize_model,
        token=hf_token,
        device=torch.device(diarize_device),
    )
    diarize_segments = diarize_pipeline(audio, num_speakers=num_speakers)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    # ── Build clean segment list ───────────────────────────────────────────
    segments = [
        {
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg["text"].strip(),
        }
        for seg in result["segments"]
        if seg.get("text", "").strip()
    ]

    stem = input_path.stem
    speakers_found = sorted({s["speaker"] for s in segments})

    # ── JSON ───────────────────────────────────────────────────────────────
    payload = {
        "audio_file": input_path.name,
        "language": "es",
        "model": "large-v3",
        "diarize_model": diarize_model,
        "num_speakers_requested": num_speakers,
        "speakers_found": speakers_found,
        "total_segments": len(segments),
        "segments": segments,
    }
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"JSON  → {json_path}")

    # ── Human-readable TXT ─────────────────────────────────────────────────
    txt_path = out_dir / f"{stem}.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write(f"Transcript: {input_path.name}\n")
        f.write(f"Speakers detected: {', '.join(speakers_found)}\n")
        f.write("Relabel by replacing SPEAKER_0X with a name (e.g. Moderadora, Joven1...)\n")
        f.write("=" * 70 + "\n\n")
        for seg in segments:
            f.write(f"[{format_ts(seg['start'])}] {seg['speaker']}: {seg['text']}\n")
    log.info(f"TXT   → {txt_path}")

    # ── CSV (for manual annotation in spreadsheet) ─────────────────────────
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig for Excel
        writer = csv.DictWriter(
            f, fieldnames=["start", "end", "speaker", "speaker_label", "text"]
        )
        writer.writeheader()
        for seg in segments:
            writer.writerow({**seg, "speaker_label": ""})
    log.info(f"CSV   → {csv_path}")

    log.info(f"Done — {len(segments)} segments across {len(speakers_found)} speakers.")


if __name__ == "__main__":
    transcribe()
