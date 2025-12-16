# Installation

## Requirements

- Python 3.11 or later
- A terminal that supports ANSI escape codes

## pip (recommended)

Install from PyPI:

```bash
pip install snakesee
```

### With logo support

To enable the Fulcrum Genomics logo easter egg (press `fg` in the TUI):

```bash
pip install snakesee[logo]
```

This installs the optional `rich-pixels` and `pillow` dependencies.

## conda / mamba

Install from Bioconda:

```bash
conda install -c bioconda snakesee
```

Or with mamba:

```bash
mamba install -c bioconda snakesee
```

## From source

Clone and install in development mode:

```bash
git clone https://github.com/fulcrumgenomics/snakesee.git
cd snakesee
pip install -e '.[logo]'
```

## Verify installation

```bash
snakesee --help
```

You should see:

```
usage: snakesee [-h] {watch,status} ...

positional arguments:
  {watch,status}
    watch         Watch a Snakemake workflow in real-time with a TUI dashboard.
    status        Show a one-time status snapshot (non-interactive).

options:
  -h, --help      show this help message and exit
```
