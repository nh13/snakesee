# Usage

## CLI Commands

### watch

Watch a Snakemake workflow in real-time with a TUI dashboard.

```bash
# In a workflow directory
snakesee watch

# Specify a path
snakesee watch /path/to/workflow

# Custom refresh rate (default: 2.0 seconds)
snakesee watch --refresh 5.0

# Disable time estimation
snakesee watch --no-estimate
```

Press `q` to quit the TUI, or `?` to see all keyboard shortcuts.

### status

Show a one-time status snapshot (non-interactive).

```bash
# In a workflow directory
snakesee status

# Specify a path
snakesee status /path/to/workflow

# Without time estimation
snakesee status --no-estimate
```

Output example:

```
Status: RUNNING
Progress: 25/100 (25.0%)
Elapsed: 5m 30s
Running: 4 jobs
ETA: ~15m
Log: .snakemake/log/2024-01-15T120000.snakemake.log
```

## TUI Keyboard Shortcuts

### General

| Key | Action |
|-----|--------|
| `q` | Quit |
| `?` | Show help overlay |
| `p` | Pause/resume auto-refresh |
| `e` | Toggle time estimation |
| `w` | Toggle wildcard conditioning (see below) |
| `r` | Force refresh |
| `Ctrl+r` | Hard refresh (reload historical data) |

### Refresh Rate

| Key | Action |
|-----|--------|
| `+` / `-` | Fine adjust (±0.5s) |
| `<` / `>` | Coarse adjust (±5s) |
| `0` | Reset to default (1s) |
| `G` | Set to minimum (0.5s, fastest) |

### Layout

| Key | Action |
|-----|--------|
| `Tab` | Cycle layout mode (full/compact/minimal) |

Layout modes:

- **Full**: Header, progress, running jobs, pending jobs, completions, stats, footer
- **Compact**: Header, progress, running jobs, footer
- **Minimal**: Header, progress, footer

### Filtering

| Key | Action |
|-----|--------|
| `/` | Enter filter mode (filter rules by name) |
| `n` | Next filter match |
| `N` | Previous filter match |
| `Esc` | Clear filter, return to latest log |

### Log History Navigation

Browse through historical workflow executions:

| Key | Action |
|-----|--------|
| `[` | View older log (1 step back) |
| `]` | View newer log (1 step forward) |
| `{` | View older log (5 steps back) |
| `}` | View newer log (5 steps forward) |
| `Esc` | Return to latest log |

### Table Sorting

| Key | Action |
|-----|--------|
| `s` / `S` | Cycle sort table forward/backward |
| `1-4` | Sort by column (press again to reverse) |

### Modal Navigation (vim-style)

snakesee uses a two-mode navigation system for exploring jobs and their logs:

**Enter Table Mode:** Press `Enter` from the main view

| Key | Action |
|-----|--------|
| `j` / `k` | Move down/up one row |
| `g` / `G` | Jump to first/last row |
| `Ctrl+d` / `Ctrl+u` | Half-page down/up |
| `Ctrl+f` / `Ctrl+b` | Full-page down/up |
| `h` / `l` | Switch to running/completions table |
| `Tab` / `Shift+Tab` | Cycle between tables |
| `Enter` | View selected job's log |
| `Esc` | Exit table mode |

**Log Viewing Mode:** Press `Enter` on a selected job

| Key | Action |
|-----|--------|
| `j` / `k` | Scroll down/up one line |
| `g` / `G` | Jump to start/end of log |
| `Ctrl+d` / `Ctrl+u` | Half-page down/up |
| `Ctrl+f` / `Ctrl+b` | Full-page down/up |
| `Esc` | Return to table mode |

## Time Estimation

snakesee estimates remaining workflow time using historical data from `.snakemake/metadata/`. The estimation methods are:

- **weighted**: Uses per-rule timing with exponential weighting (favors recent runs)
- **simple**: Linear extrapolation based on average time per job
- **bootstrap**: Initial estimate when no jobs have completed

### Weighting Strategies

snakesee supports two strategies for weighting historical timing data:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `index` (default) | Weights by run index (number of runs ago) | Active development, frequent changes |
| `time` | Weights by wall-clock time since run | Stable pipelines, infrequent runs |

**Index-based weighting** treats each run as a potential improvement, discounting older runs regardless of actual time elapsed. Use `--weighting-strategy index --half-life-logs 10`.

**Time-based weighting** discounts data based on calendar time, naturally aging out old runs. Use `--weighting-strategy time --half-life-days 7`.

The ETA display shows confidence levels:

- `~5m` - High confidence estimate
- `3m - 7m` - Medium confidence range
- `~5m (rough)` - Low confidence estimate
- `unknown` - No data available

Toggle estimation with `e` or disable at startup with `--no-estimate`.

### Wildcard Conditioning

When enabled (default), wildcard conditioning improves time estimates by considering the specific wildcard values of each job. For example, if a rule processes different samples and some samples take longer than others, wildcard conditioning will use sample-specific timing data.

Without wildcard conditioning, snakesee uses only rule-level averages. With conditioning enabled, it looks for historical jobs with matching wildcard values and uses those timings for more accurate per-job estimates.

Toggle wildcard conditioning with `w` in the TUI or use `--no-wildcard-timing` at startup.

## How It Works

snakesee reads from the `.snakemake/` directory that Snakemake creates:

- `.snakemake/log/*.snakemake.log` - Workflow logs (progress, job status)
- `.snakemake/metadata/` - Completed job timing data
- `.snakemake/locks/` - Lock files (indicates running workflow)
- `.snakemake/incomplete/` - In-progress job markers

No special flags are needed when running Snakemake - snakesee works with any existing workflow.

## Enhanced Monitoring with Logger Plugin

For more accurate real-time monitoring, you can optionally use the Snakemake logger plugin. This provides direct event streaming from Snakemake instead of log parsing.

### Installation

```bash
pip install snakemake-logger-plugin-snakesee
```

### Usage

Run Snakemake with the logger plugin enabled:

```bash
snakemake --logger snakesee --cores 4
```

Then monitor with snakesee as usual:

```bash
snakesee watch
```

### Benefits

| Feature | Without Plugin | With Plugin |
|---------|---------------|-------------|
| Job detection | Log parsing (polling) | Real-time events |
| Start times | Approximate (log modification time) | Exact timestamp |
| Job durations | Calculated from log patterns | Precise from events |
| Failed jobs | Pattern matching in logs | Direct notification |

### How It Works

When the logger plugin is active, Snakemake writes events to `.snakesee_events.jsonl` in the workflow directory. Snakesee automatically detects this file and uses the events to enhance its monitoring accuracy.

The plugin is completely optional - snakesee works without it using log parsing, and automatically switches to using events when available.
