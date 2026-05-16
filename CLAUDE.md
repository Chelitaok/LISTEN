# LISTEN — Audio Transcription & NLP Extraction

## Project Overview

LISTEN is a pipeline that uses ASR (Whisper) to transcribe audio files and NLP (spaCy) to extract structured information from the resulting transcripts.

## Stack

- **ASR**: [OpenAI Whisper](https://github.com/openai/whisper) — speech-to-text transcription
- **NLP**: [spaCy](https://spacy.io/) — named entity recognition, dependency parsing, information extraction
- **Language**: Python 3.10+

## Repository Structure

```
LISTEN/
├── data/
│   ├── audio/          # Raw audio input files (.mp3, .wav, .m4a, etc.)
│   ├── transcripts/    # Raw Whisper transcription output (JSON + txt)
│   └── processed/      # NLP-extracted structured data (JSON)
├── src/
│   ├── asr/            # Whisper transcription pipeline
│   ├── nlp/            # spaCy extraction pipeline
│   └── utils/          # Shared helpers (file I/O, logging, etc.)
├── models/             # Custom/fine-tuned model artifacts (not committed)
├── notebooks/          # Jupyter notebooks for exploration and prototyping
├── tests/              # Unit tests
├── config/             # YAML/JSON pipeline configuration files
└── outputs/            # Final structured pipeline outputs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Pipeline

```bash
# Transcribe audio
python src/asr/transcribe.py --input data/audio/example.mp3

# Extract information from transcripts
python src/nlp/extract.py --input data/transcripts/example.json

# Run end-to-end
python src/pipeline.py --input data/audio/example.mp3
```

## Development Guidelines

- All audio files go in `data/audio/` — never committed to git (see `.gitignore`)
- Transcripts and processed outputs are also gitignored — only code and config are committed
- Use `config/` YAML files for model names, thresholds, and pipeline parameters — no hardcoded values in `src/`
- Each module in `src/` should be independently runnable with a `--input`/`--output` CLI
- Write tests for any extraction logic in `tests/` — NLP pipelines are easy to silently regress
- Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) is set in `config/asr.yaml`
- spaCy model is set in `config/nlp.yaml`

## Key Conventions

- Input audio formats: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`
- Transcript format: JSON with keys `text`, `segments`, `language` (Whisper native output)
- Extracted output format: JSON with entity lists, key phrases, and any custom fields
- Logging via Python `logging` module — set level in `config/pipeline.yaml`

## Notes

- Large Whisper models require significant VRAM; `medium` works well on CPU for short clips
- For GPU inference, ensure `torch` CUDA version matches your driver
- spaCy transformer models (`en_core_web_trf`) are more accurate but slower than `en_core_web_sm`
