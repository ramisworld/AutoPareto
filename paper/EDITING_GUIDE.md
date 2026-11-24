# Editing the AutoPareto paper

The paper is editable LaTeX. The scientific figures are editable Python and Matplotlib, so arrows, coordinates, colors, labels, line widths, and font sizes can all be changed without image-editing software.

## Recommended editor

Open `/Users/Rami/Documents/dev/autopareto` from **File > Open Folder** in VS Code. The `code` shell command is not currently installed. If you later install it, the equivalent command is:

```sh
cd /Users/Rami/Documents/dev/autopareto
code .
```

Install the VS Code extension **LaTeX Workshop** if you want source and PDF side by side. Tectonic is already installed at `/opt/homebrew/bin/tectonic`.

The most useful files are:

- `paper/main.tex`: title, introduction, sections, captions, figure placement, figure size, and narrative text.
- `analysis/generate_reports.py`: abstract and Primary Result sentences. These are generated from the claim ledger.
- `analysis/generate_figures.py`: all graphs and diagrams.
- `analysis/generate_tables.py`: tables and numerical LaTeX macros.
- `paper/references.bib`: references.
- `paper/neurips_2026.sty`: workshop formatting. Avoid editing this unless the official template itself changes.

Files under `paper/generated/` are build outputs. A full refresh overwrites them. Edit their generator instead.

## Fast editing loop for prose

Edit `paper/main.tex`, save it, then run:

```sh
cd /Users/Rami/Documents/dev/autopareto/paper
tectonic --keep-logs --keep-intermediates main.tex
open main.pdf
```

This recompiles only the paper. It does not rerun experiments or regenerate data.

For the abstract, edit the tuples beginning with `('abstract', ...)` in `analysis/generate_reports.py`, then run:

```sh
cd /Users/Rami/Documents/dev/autopareto
paper/.venv/bin/python analysis/generate_reports.py
cd paper
tectonic --keep-logs --keep-intermediates main.tex
open main.pdf
```

## Change a figure's size inside the paper

Find its `\includegraphics` line in `paper/main.tex`.

```tex
\includegraphics[width=\linewidth]{figures/autopareto_lineage.pdf}
```

Examples:

```tex
% 90 percent of the text width
\includegraphics[width=.90\linewidth]{figures/autopareto_lineage.pdf}

% Fixed height while preserving the aspect ratio
\includegraphics[height=7cm,keepaspectratio]{figures/autopareto_lineage.pdf}

% Set both limits and never distort the figure
\includegraphics[width=\linewidth,height=7cm,keepaspectratio]{figures/autopareto_lineage.pdf}
```

Figure captions are the `\caption{...}` lines immediately below each image. Placement codes mean `[t]` for top of page, `[b]` for bottom, `[h]` for approximately here, and `[H]` for exactly here. Using `[H]` can create awkward blank space, so recheck the complete PDF after changing it.

## Edit the lineage tree

Open `analysis/generate_figures.py` and search for:

```python
# The complete seed-0 lineage
```

The important controls are:

- `figsize=(6.5,3.6)`: overall width and height in inches.
- `width_ratios=[2.35,1.05,1.05]`: relative width of the tree, BPB panel, and throughput panel.
- `main_path={...}`: experiment database IDs highlighted in green.
- `edit={...}`: visible edit label for every attempt.
- `green`, `gray`, `dark`, and `bad`: hexadecimal colors.
- `branch_x`: computed horizontal tree positions.
- `order`: vertical row for every attempt.
- `arrowprops`: arrow style.
- `fontsize=4.8`: source-edit label size.
- `fontsize=4.5`: metric-number size.
- `s=22` and `s=18`: node-marker areas.
- `lw=1.25` and `lw=.55`: highlighted and ordinary edge widths.
- `mutation_scale=5`: arrowhead size.
- `ax.set_xlim(...)`, `axb.set_xlim(...)`, and `axs.set_xlim(...)`: visible coordinate ranges.

The lineage arrows are created by this call:

```python
ax.annotate(
    '',
    xy=(branch_x[child_id], child_row),
    xytext=(branch_x[parent_id], parent_row),
    arrowprops={
        'arrowstyle': '-|>',
        'mutation_scale': 5,
        'lw': 1.25,
        'color': green,
        'shrinkA': 2,
        'shrinkB': 2,
    },
)
```

`xytext` is the start at the parent and `xy` is the arrowhead at the child. Increase `mutation_scale` for a larger arrowhead. Increase `lw` for a thicker line. Increase `shrinkA` or `shrinkB` to leave more space between the arrow and a node.

To move one tree node manually after `branch_x` is computed, add a line such as:

```python
branch_x[137] += 0.08
```

Here `137` is the database ID for attempt 25. Positive values move it right and negative values move it left. The displayed attempt number and database ID are different, so confirm the ID in `main_path` or `paper/generated/historical_experiments.json`.

## Edit the research-loop diagram

In `analysis/generate_figures.py`, search for:

```python
fig,ax=plt.subplots(figsize=(5.5,3.35))
```

Every box uses:

```python
box(x, y, width, height, title, body, color, title_size, body_size)
```

For example:

```python
box(6.75, 3.35, 4.7, .84,
    '5  Train on one A40',
    'discard warm-up; reconstruct initial state\nrun T=300 s or N=counted tokens',
    amber, 5.6, 4.45)
```

- First number: horizontal position.
- Second number: vertical position.
- Third and fourth numbers: box width and height.
- `5.6`: title size.
- `4.45`: body size.
- `\n`: manual line break.

Arrows use:

```python
arrow((start_x, start_y), (end_x, end_y), curve)
```

For example:

```python
arrow((5.30, 3.77), (6.69, 3.77))
arrow((9.10, 3.31), (2.90, 2.81), .15)
```

The third value curves the arrow. Use `0` for straight, a positive value for one direction, and a negative value for the opposite direction. The global arrowhead and line width are in the `def arrow(...)` function as `mutation_scale=7` and `lw=.85`.

The dashed trust boundary is a `FancyBboxPatch`. Change its `(x, y)` origin, width, height, color, or `linestyle` there.

## Edit the Pareto and budget plots

In `analysis/generate_figures.py`:

- Search for `budget_trajectory` to edit the four fixed-time versus fixed-token panels.
- Search for `pareto_front` to edit the three equal-token Pareto panels.
- `figsize=(...)` controls canvas size.
- `ms=4` controls marker size.
- `lw=1.3` controls line width.
- `capsize=2` controls error-bar caps.
- `alpha=.15` controls grid transparency.
- `COLORS=[...]` near the top controls the model palette.
- `set_xlabel`, `set_ylabel`, and `set_title` control axis text.
- `set_xlim`, `set_ylim`, and `set_xticks` control the axis ranges and tick locations.

Do not type experimental values into the plotting code. The plots read normalized records from `paper/generated/*.json`. If a value is wrong, fix the ingestion or analysis script so the figure, table, and claim ledger stay consistent.

## Preview figure changes quickly

Run:

```sh
cd /Users/Rami/Documents/dev/autopareto
./paper/preview_figures.sh
open paper/main.pdf
```

This regenerates all PDF and PNG figures and recompiles the manuscript. It does not rebuild the evidence dataset.

You can inspect an individual PNG without recompiling:

```sh
open paper/figures/autopareto_lineage.png
open paper/figures/research_loop.png
open paper/figures/pareto_front.png
open paper/figures/budget_trajectory.png
```

## Full evidence-safe rebuild

After changing analysis logic, data ingestion, numerical macros, the abstract generator, or tables, run:

```sh
cd /Users/Rami/Documents/dev/autopareto
./paper/refresh_paper.sh --cutoff 2026-09-06T23:59:00+12:00
```

This rebuilds the normalized data, analyses, tables, claims, figures, PDF, and source bundle. It runs the tests and page-limit checker. It never launches experiments.

After a full refresh, visually inspect every PDF page because the refresh marks visual review as pending. Also check that Part 1 remains at most four pages, Part 2 at most four pages, and Part 3 at most one page.

## Common LaTeX formatting changes

```tex
\textbf{bold text}
\emph{italic text}
\small smaller text
\footnotesize still smaller text
\vspace{-2mm} % reduce vertical space carefully
\hspace{2mm}  % add horizontal space
```

For references, add an entry to `paper/references.bib` and cite its key with `\citep{key}` or `\citet{key}`.

For the author name, keep `Anonymous Author(s)` in the review PDF. The private author details are in `paper/AUTHOR_METADATA_PRIVATE.md`. Add the name and email only when the submission stage explicitly requests an identified or camera-ready paper.

## Undo a bad manual change

Before editing, create a Git checkpoint:

```sh
cd /Users/Rami/Documents/dev/autopareto
git add paper analysis
git commit -m "Checkpoint paper before manual edits"
```

Then inspect changes with:

```sh
git diff -- paper/main.tex analysis/generate_figures.py
```

Restore one file to the last commit only when you are certain you do not need its uncommitted edits:

```sh
git restore paper/main.tex
```
