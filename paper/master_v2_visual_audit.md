# Master v2 figure and table audit

Rendered at 115 dpi and inspected page by page. PDF: `paper/master_v2/main.pdf`, 11 pages.

## Figures

| Figure | Page | Source | Status | Notes |
| --- | --- | --- | --- | --- |
| 1. Research loop and trust boundary | 3 | `analysis/generate_figures.py` | Kept, unchanged | Box text legible at 100%. Trust boundary is a labeled dashed enclosure; the agent box sits outside it and no arrow runs from the agent into protected evaluation. Referenced on page 2. |
| 2. Seed-0 autonomous lineage | 4 | `analysis/generate_figures.py` | Rebuilt for legibility | Root cause of the workshop version's tiny text: `bbox_inches='tight'` expanded the canvas to 10 inches because right-hand labels overflowed the axes, so `width=\linewidth` scaled fonts to about 55%. Fixed by saving at the declared 6.6 x 4.6 inch canvas, widening the label panel (width ratio 3.1) and the x limit to 2.05, and raising label fonts to 7.0 pt (about 5.8 pt on the page). The clipped in-figure legend line was removed and its content moved into the caption. A22 is bold, green, and annotated "exp 134 = AP292 recipe". A24 to A25 is amber dashed. Crashed A17 retained. |
| 3. Discovery versus verification | 5 | `analysis/master_v2/build_master.py` | New | Replaces the single held-out scatter. Both panels place each recipe at its verified decode throughput so only the quality axis changes; panel A markers are sized and annotated by discovery token exposure. Label collisions between TPE (114M) and MOTPE (153M) resolved by vertical offsets. Panel B carries seed SD error bars. |
| Removed | n/a | `budget_trajectory.pdf` | Not included | The trajectory plot implied sequential ancestry that the records contradict (N3 returns to the N1 parent) and its BPB axis is dominated by the two T outliers near 3.216. Table 4 plus prose carries the same content honestly. |

## Tables

| Table | Page | Status | Notes |
| --- | --- | --- | --- |
| 1. Discovery exposure and matched-exposure verification | 5 | Rebuilt | Adds two discovery columns (tokens consumed, validation BPB) that the workshop version lacked, plus held-out SD and incremental KV. Ten columns at `\small`, no overflow. |
| 2. Complete downstream suite | 6 | Restored | Was compressed into prose in the workshop version. Now all 7 verified recipes against 8 distinct metrics. No "Favored" column, no win counts, no bolding that would imply untested significance. BLiMP macro/micro collapsed to one column with the reason stated. |
| 3. Executed budget controls | 7 | Kept | Moved inside its own section rather than floating ahead of it. |
| 4. Completed budget attempts | 7 | Kept | Caption states four per arm and that the eight attempts are not independent samples. |
| 5. Artifact status | 8 | New | Separates "preserved" from "released" so the reproducibility claim cannot be read as stronger than it is. |
| 6. Downstream seed SDs | 10 (appendix) | New | Supports the claim that most task-level differences are within seed variation. |

## Checks performed on the rendered PDF

- Body font is uniform throughout; no `\small`, `\footnotesize`, or `\scriptsize` applies to any prose paragraph. The two size directives that remain apply only to table bodies, which is conventional.
- No page is sparse. Text volume per page ranges from 3,248 to 4,857 extracted characters; the near-empty conclusion page present in an intermediate build was removed by tightening two sentences rather than by shrinking type.
- No table or figure appears before the section that introduces it.
- No orphan section headings, no clipped axis labels, no table overflow past the text block.
- Zero unresolved references or citations; all 12 bibliography entries are cited and resolve.
- Zero em dash characters.
- No "Importantly", "Notably", "This highlights", "This underscores", "TODO", "provisional", or placeholder author fields.
- No author name, email, private host, or local filesystem path anywhere in the text or metadata.
- Terminology is consistent: A22 is the attempt, experiment 134 is its archive row, AP292 is the frozen recipe under verification. Every occurrence checked.
- Headline numbers are identical in the abstract, Section 4, and the conclusion.
