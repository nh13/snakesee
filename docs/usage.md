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
| `r` | Force refresh |
| `Ctrl+r` | Hard refresh (reload historical data) |

### Refresh Rate (vim-style)

| Key | Action |
|-----|--------|
| `h` | Decrease by 5s (faster) |
| `j` | Decrease by 0.5s (faster) |
| `k` | Increase by 0.5s (slower) |
| `l` | Increase by 5s (slower) |
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

### Log Navigation

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
| `s` | Cycle sort table (Running → Completions → Stats → none) |
| `1` | Sort by column 1 (press again to reverse) |
| `2` | Sort by column 2 |
| `3` | Sort by column 3 |
| `4` | Sort by column 4 (Running/Stats tables only) |

## Time Estimation

snakesee estimates remaining workflow time using historical data from `.snakemake/metadata/`. The estimation methods are:

- **weighted**: Uses per-rule timing with exponential weighting (favors recent runs)
- **simple**: Linear extrapolation based on average time per job
- **bootstrap**: Initial estimate when no jobs have completed

The ETA display shows confidence levels:

- `~5m` - High confidence estimate
- `3m - 7m` - Medium confidence range
- `~5m (rough)` - Low confidence estimate
- `unknown` - No data available

Toggle estimation with `e` or disable at startup with `--no-estimate`.

## How It Works

snakesee reads from the `.snakemake/` directory that Snakemake creates:

- `.snakemake/log/*.snakemake.log` - Workflow logs (progress, job status)
- `.snakemake/metadata/` - Completed job timing data
- `.snakemake/locks/` - Lock files (indicates running workflow)
- `.snakemake/incomplete/` - In-progress job markers

No special flags are needed when running Snakemake - snakesee works with any existing workflow.
