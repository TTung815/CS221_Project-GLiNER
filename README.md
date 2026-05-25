# GLiNER NER Evaluation & Demo

Group project for **CS221 - Xử lý ngôn ngữ tự nhiên**.

This repository contains evaluation scripts and an interactive demo for
**GLiNER (Generalist and Lightweight Named Entity Recognition)**. The project
tests zero-shot NER performance across CrossNER domains and several public NER
benchmarks, then exposes a small Gradio playground for trying GLiNER on custom
text and custom entity labels.

## Team Credits

**Instructor:** TS. Nguyễn Thị Quý

| Name                | Student ID | Email                  |
| ------------------- | ---------- | ---------------------- |
| Nguyễn Thanh Tùng   | 23521745   | 23521745@gm.uit.edu.vn |
| Mai Lê Bá Vương     | 23521821   | 23521821@gm.uit.edu.vn |
| Chướng Hồng Văn     | 23521769   | 23521769@gm.uit.edu.vn |

## Tech Stack

- Python 3.8+
- GLiNER models from Hugging Face (`urchade/gliner_*`)
- PyTorch for model execution on CUDA, Apple MPS, or CPU
- pandas and tqdm for evaluation reporting
- Hugging Face `datasets` for public benchmark loading
- requests for downloading HarveyNER test data
- Gradio for the web demo

## Repository Structure

```text
.
├── data/crossner/              # CrossNER test files used by the evaluation script
├── demo/app.py                 # Gradio NER playground
├── demo/requirements.txt       # Demo dependencies
├── scripts/test_crossner.py    # CrossNER evaluation
└── scripts/test_20nerbenchmark.py
                                # Multi-benchmark NER evaluation
```

## Methodology / Workflow

1. Load a selected GLiNER checkpoint: small, medium, or large.
2. Convert benchmark labels into token-level entity spans.
3. Run zero-shot entity extraction with each dataset's entity label set.
4. Align GLiNER character-level predictions back to token spans.
5. Compute exact-span micro F1 from true positives, false positives, and false negatives.
6. Print tabular results and optionally save CSV outputs for later comparison.
7. Use the Gradio demo to inspect predictions interactively on free-form text.

The scripts fix random seeds where applicable and choose the best available
runtime device in this order: CUDA, Apple MPS, then CPU.

## Setup

Use Python 3.8+ and install the required dependencies. A virtual environment is
recommended for local development, but the exact setup can depend on your OS and
Python distribution.

```bash
pip install gliner torch pandas tqdm numpy datasets requests
```

Install demo dependencies:

```bash
cd demo
pip install -r requirements.txt
```

## CrossNER Evaluation

CrossNER test files are included under `data/crossner`. If you move the dataset,
update `CROSSNER_ROOT` in `scripts/test_crossner.py`.

Run the default evaluation with GLiNER-L across all CrossNER domains:

```bash
cd scripts
python test_crossner.py --model l --subset all
```

Available model sizes:

```bash
python test_crossner.py --model s
python test_crossner.py --model m
python test_crossner.py --model l
```

Available domains:

- `ai`
- `literature`
- `music`
- `politics`
- `science`

Save results to CSV:

```bash
python test_crossner.py --model m --subset ai literature --save-csv
```

The generated file is named `crossner_results_{model_size}.csv`.

## Multi-Benchmark Evaluation

This script evaluates GLiNER on public NER datasets from Hugging Face plus
HarveyNER. The first run requires internet access for dataset/model downloads.

Run all supported datasets with GLiNER-L:

```bash
cd scripts
python test_20nerbenchmark.py --model l --dataset all
```

Supported datasets:

- `WikiANN`
- `BC5CDR`
- `NCBI`
- `GENIA`
- `BC2GM`
- `TweetNER7`
- `HarveyNER`

Run a subset and save results:

```bash
python test_20nerbenchmark.py --model s --dataset WikiANN BC5CDR GENIA --save-csv
```

The generated file is named `7ner_results_{model_size}.csv`.

## Web Demo

The demo lets users select one or more GLiNER models, provide custom text, and
enter comma-separated entity labels.

```bash
cd demo
python app.py
```

The Gradio app starts locally, typically at `http://localhost:7860`.

To create a temporary public Gradio link:

```bash
python app.py --share
```

## Results

The repository does not currently include committed result CSV files. When the
evaluation scripts are run, they print F1 tables to the terminal and can save
CSV outputs with `--save-csv`.

Expected output formats:

```text
CrossNER:
Model      Ai  Literature  Music  Politics  Science  Average
GLiNER-L  ...         ...    ...       ...      ...      ...
```

```text
Multi-benchmark:
Dataset      F1 Score  Label Count
WikiANN      ...       ...
BC5CDR       ...       ...
```
