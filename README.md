# HuggingFace Dataset Downloader

A fast, multithreaded CLI tool for downloading datasets from [HuggingFace](https://huggingface.co/datasets).

## Features

- **Multithreaded Downloads** - Configurable parallel downloads for maximum speed
- **Preserves Folder Structure** - Maintains original dataset organization (train/test splits, etc.)
- **Flexible Filtering** - Download only specific file types or limit file count
- **Private Dataset Support** - Authenticate with HuggingFace tokens
- **Progress Tracking** - Optional progress bar with tqdm
- **Dry Run Mode** - Preview files before downloading

## Supported File Types

- **Images**: PNG, JPG, JPEG, WEBP, GIF, BMP, etc.
- **Labels/Metadata**: Parquet, JSON, CSV, JSONL
- **Other**: Any file type in the dataset repository

## Installation

### Prerequisites

- Python 3.10+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/HuggingFace_scraper.git
cd HuggingFace_scraper

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `requests` - HTTP library for downloads
- `huggingface_hub` - Official HuggingFace API client
- `tqdm` - Progress bar (optional but recommended)

## Usage

### Basic Usage

```bash
# Download a dataset (saves to ./edc505_pokemon/)
python hf_downloader.py edc505/pokemon

# Download with progress bar
python hf_downloader.py edc505/pokemon --progress

# Download dataset with labels
python hf_downloader.py jise/simworld-20k-balanced --progress
```

### Advanced Usage

```bash
# Increase download speed with more threads
python hf_downloader.py edc505/pokemon -t 16 --progress

# Download only images
python hf_downloader.py edc505/pokemon --filter .png,.jpg,.jpeg,.webp

# Download only metadata files
python hf_downloader.py jise/simworld-20k-balanced --filter .json,.csv,.parquet

# Limit number of files
python hf_downloader.py edc505/pokemon --limit 100 --progress

# Custom output directory
python hf_downloader.py edc505/pokemon --output ./datasets/pokemon

# Preview files without downloading
python hf_downloader.py edc505/pokemon --dry-run
```

### Private Datasets

```bash
# Using command line token
python hf_downloader.py username/private-dataset --token hf_xxxxxxxxxx

# Using environment variable (recommended)
export HF_TOKEN=hf_xxxxxxxxxx  # Linux/macOS
set HF_TOKEN=hf_xxxxxxxxxx     # Windows CMD
$env:HF_TOKEN="hf_xxxxxxxxxx"  # Windows PowerShell

python hf_downloader.py username/private-dataset --progress
```

Get your token at: https://huggingface.co/settings/tokens

## CLI Reference

```
usage: hf_downloader [-h] [--base-url BASE_URL] [-o OUTPUT] [-t THREADS]
                     [-f FILTER] [-l LIMIT] [--token TOKEN] [-p] [--dry-run] [-v]
                     dataset

Download datasets from HuggingFace with multithreading support.

positional arguments:
  dataset               Dataset path (e.g., 'edc505/pokemon')

options:
  -h, --help            Show help message and exit
  --base-url BASE_URL   Base URL for HuggingFace datasets
  -o, --output OUTPUT   Output directory
  -t, --threads THREADS Number of parallel threads (default: 8)
  -f, --filter FILTER   File extensions to download (e.g., '.png,.jpg')
  -l, --limit LIMIT     Maximum number of files to download
  --token TOKEN         HuggingFace API token for private datasets
  -p, --progress        Show progress bar
  --dry-run             List files without downloading
  -v, --verbose         Verbose output
```

## Options Reference

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--base-url` | | `https://huggingface.co/datasets` | Base URL for datasets |
| `--output` | `-o` | `<user>_<dataset>` | Output directory path |
| `--threads` | `-t` | `8` | Number of parallel download threads |
| `--filter` | `-f` | `None` | Comma-separated file extensions |
| `--limit` | `-l` | `None` | Maximum files to download |
| `--token` | | `$HF_TOKEN` | HuggingFace authentication token |
| `--progress` | `-p` | `False` | Display progress bar |
| `--dry-run` | | `False` | List files only, no download |
| `--verbose` | `-v` | `False` | Show detailed output |

## Examples

### Download Pokemon Dataset

```bash
python hf_downloader.py edc505/pokemon -t 16 --progress
```

Output:
```
Fetching file list from 'edc505/pokemon'...
Found 833 file(s) to download.
Downloading: 100%|████████████████████| 833/833 [01:23<00:00, 10.2file/s]

Download complete!
  Success: 833
  Failed:  0
  Output:  D:\datasets\edc505_pokemon
```

### Download Only Training Images

```bash
python hf_downloader.py jise/simworld-20k-balanced \
  --filter .png,.jpg \
  --limit 1000 \
  --output ./training_data \
  --progress
```

### List Dataset Contents

```bash
python hf_downloader.py edc505/pokemon --dry-run
```

Output:
```
Fetching file list from 'edc505/pokemon'...
Found 833 file(s) to download.

Files (dry run):
  data/train/image_001.png
  data/train/image_002.png
  ...
```

## Output Structure

The tool preserves the original dataset folder structure:

```
edc505_pokemon/
├── data/
│   ├── train/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── test/
│       └── ...
├── metadata.json
└── README.md
```

## Troubleshooting

### "Dataset not found"

- Verify the dataset path is correct
- Check if the dataset is private (requires `--token`)

### "Gated repository"

Some datasets require accepting terms on HuggingFace:
1. Visit the dataset page on huggingface.co
2. Accept the terms/license
3. Use your token with `--token`

### Slow downloads

- Increase threads: `-t 16` or `-t 32`
- Check your internet connection
- HuggingFace servers may be under load

### Missing tqdm

```bash
pip install tqdm
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [HuggingFace](https://huggingface.co/) for their excellent datasets platform
- [huggingface_hub](https://github.com/huggingface/huggingface_hub) library
