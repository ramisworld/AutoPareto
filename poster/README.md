# AutoPareto poster (NeurIPS 2026 AutoMLR workshop)

## Version 7: A1 portrait, cut material and larger figures (current)

| File | What it is |
|---|---|
| `out_v7/autopareto_poster_v7_A1.pdf` | Print file: 594 × 841 mm portrait, vector, fonts embedded, no raster images |
| `out_v7/autopareto_poster_v7_A1.svg` | Editable vector copy |
| `out_v7/autopareto_poster_v7_preview.png` | Full-page preview, 150 dpi |
| `out_v7/autopareto_poster_v7_view_2-3m.png` | Low-resolution render approximating a viewer at 2 to 3 m |
| `out_v7/crops/*.png` | The header (with the QR code) and each section at full preview resolution |
| `build_poster_v7.py` | Layout, charts and diagrams (drawing helpers copied from v6) |

```bash
/Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python poster/build_poster_v7.py
```

**Why.** The workshop organiser asked for one more iteration that cuts material, and Arjun pointed out repetition.
v7 keeps six figures (v6 had thirteen charts and diagrams) and about 17% fewer words, and the figures that remain are
much larger.

**Kept and enlarged:**
- **Research loop** (section 1): now an actual loop, with a return path labelled with the paper's description of
  how each attempt reads all earlier ones and picks a parent.
- **Whole-generation latency** (section 1): 880 ms against 192 ms, ~32× and 4.6×.
- **Opus discovery lineage** (section 2): now the full page width.
- **CPU/GPU execution diagram** (section 3): now the full page width.
- **Both Pareto scatterplots** (section 4): each about twice the area of v6's.

**Removed, with what survives as annotation:**
- **Section 1 BPB strip:** removed. The same 300 s results are in the equal-time Pareto plot.
- **Section 2 verified-throughput bars:** replaced by one line under the lineage: 145, 266 (1.8×) and 4,583 tok/s
  (31.6×).
- **Section 3 ablation bars:** replaced by a three-line box beside the faster lanes. It gives the results the lanes
  do not show: fusion without the graph (2.7×), without custom attention (23.8×), without the ring cache (29.3×), and
  the 13-recipe test's 31.4 to 32.1×.
- **Section 4 BPB dot plots:** removed. Their means are in the Pareto plots, with a small number block in each.
- **Section 4 token and memory bars:**
  - Tokens seen in 300 s are now in the equal-time number block (88M, 158M, 198M).
  - Memory is one line under the plots (337, 243, 96 MiB).
  - The matched-exposure finding is the equal-tokens subtitle and block ("AR-ref learns the most per token", "best
    in every seed").
- **Section 5 lessons:** removed. The limits, AI-use disclosure and references sat in the footer until 2026-11-24, when the author removed them; the footer is now one line of contact, code and venue. They remain in the paper.

**Execution diagram: one time axis.**
- **Lane length:** the time per decoded token in the ablation session, drawn on one millisecond axis: 6.71, 3.40,
  0.58 and 0.22 ms.
- **Speed-up:** a separate column, defined as 6.71 ms ÷ time, so latency and speed-up are never confused.
- **Marks:** each mark stands for 10 profiler-counted CPU launches or GPU kernels, at the same scale in every lane.
  Their positions are schematic.
- **Graph replay stage:** says "same 290 kernels", so graph replay is not mistaken for fusion.

**QR code and contact (added 2026-11-24).**
- **QR code:** top right of the header, linking to the paper's OpenReview page,
  `https://openreview.net/forum?id=U9ICjFvtN8`, as given by the author.
  - It is generated locally with macOS CoreImage: `qr/qr_matrix.swift`, error correction M, version 3, 29 × 29
    modules.
  - The matrix is stored in `qr/openreview_forum_M.txt`, so the build needs no network access or extra package.
  - It is drawn as one vector path, 60 mm square with a 4-module quiet zone (1.6 mm per module).
- **Scan test:** `qr/qr_decode.swift` (CoreImage `CIDetector`) decodes exactly that URL from the PDF rendered at
  300, 150, 72, 50 and 36 dpi, from the full page at 100 dpi, from the matplotlib PNG and from a blurred, rotated,
  low-contrast photo-like copy.
- **Not verified:** whether the OpenReview page is public. openreview.net and its API answer automated requests
  with a bot challenge, so open the link in a private window before printing.
- **Header:** the headline is now two lines at 30 pt, so it clears the QR.
- **Footer:** the contact line adds rami.co.nz and @vicilah on X; the venue moves to the right, and "Records and code: release pending" is removed.

To regenerate the matrix (run from the repository root; this command reproduces the stored file exactly):

```bash
swift poster/qr/qr_matrix.swift 'https://openreview.net/forum?id=U9ICjFvtN8' M > poster/qr/openreview_forum_M.txt
```

## Version 6: A1 portrait, final redesign

| File | What it is |
|---|---|
| `out_v6/autopareto_poster_v6_A1.pdf` | Print file: 594 × 841 mm portrait (meets the 61 × 85 cm portrait limit), vector, fonts embedded |
| `out_v6/autopareto_poster_v6_A1.svg` | Editable vector copy |
| `out_v6/autopareto_poster_v6_preview.png` | Full-page preview, 150 dpi |
| `out_v6/autopareto_poster_v6_view_2-3m.png` | Low-resolution render approximating a viewer at 2 to 3 m |
| `out_v6/crops/*.png` | Each section at full preview resolution |
| `build_poster_v6.py` | Layout, charts and diagrams |

```bash
/Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python poster/build_poster_v6.py
```

**Reading sequence:** five numbered questions.
1. **What does AutoPareto test?** A three-step flow: agent → edits train.py → trains 300 s, scored on three
   objectives. Beside it, the headline box: one generation to scale (880 ms against 192 ms, with the unscored prefill
   hatched), ~32× and 4.6×, and quality at 300 s on one axis (starting model 1.240, Opus A27 1.159, AR-ref 1.157, not
   resolved).
2. **What did the agents discover?**
   - Opus's lineage from its search records, labelled as historical. Architecture milestone (A4) and engine milestones
     (A10 to A22) are distinguished by colour and phase band, with the A10 hypothesis next to A10.
   - A separate "independently verified" bar chart: starting model 145, GPT A22 266, Opus A27 4,583 decode tok/s.
3. **How did Opus make decoding 32× faster?**
   - Four to-scale CPU/GPU stages: original decoder; Opus's smaller model, eager; plus graph replay; plus fusion.
     Profiler counts are shown on each stage.
   - An ablation bar chart ("take one part out") replaces the 2×2 grid.
   - The independently verified 31.4 to 32.1× is shaded on the same axis.
4. **What did independent verification reveal?**
   - A new budget figure in three aligned panels: data seen in 300 s; held-out BPB after 300 s; held-out BPB after
     201.3M tokens. Per-seed points are shown, with nothing connecting the separate runs.
   - Two Pareto panels with direct labels.
   - Memory bars.
5. **What this teaches:** three lessons.

**Changes from v5:**
- **Format:** portrait A1, recomposed rather than rotated.
- **Header:** a display headline built around the discovery; the paper's full title is kept under it.
- **Contract figure:** the arrow chart is replaced by aligned panels that never link independent runs.
- **Verified comparison:** the tiny "others / GPT / Opus" scatter is replaced by labelled bars.
- **Decoder diagram:** the original decoder is now its own stage, separate from the ablated Opus model.
- **Interaction grid:** replaced by an ablation bar chart with the verified band shown on the same axis.
- **Legends:** every colour is directly labelled.
- **Corrections:**
  - no "every agent shrank the model";
  - discovery records and verification are labelled separately;
  - CUDA graphs are described as standard, cutting launch overhead without fusing kernels;
  - "fusion may reduce memory traffic" is marked as not measured.

**QR code: not included.** The OpenReview forum (U9ICjFvtN8) could not be verified: openreview.net showed a
bot-verification challenge. Once a public URL is confirmed, a QR code can be generated and scan-tested locally with
macOS CoreImage (Swift), with no downloads.

## Version 5: A1 landscape, three columns like Arjun's poster

| File | What it is |
|---|---|
| `out_v5/autopareto_poster_v5_A1_landscape.pdf` | Print file: 841 × 594 mm, vector, fonts embedded |
| `out_v5/autopareto_poster_v5_A1_landscape.svg` | Editable vector copy |
| `out_v5/autopareto_poster_v5_preview.png` | Full-size preview, 150 dpi |
| `out_v5/autopareto_poster_v4_v5_reference.png` | v4, v5 and the reference side by side at the same physical scale |
| `build_poster_v5.py` | Layout, charts and diagrams |

**Size: check with the organizers.** The NeurIPS workshop guideline the author cited for v4 is portrait, at most
61 × 85 cm, and an 84.1 cm wide landscape sheet exceeds it. The file is vector, so it scales without loss.

**Layout, following the reference:**
- a navy header band (title, author, venue) and a footer band;
- three equal columns with centred section titles over thin rules, a one-line finding under each title, numbered
  figures with captions and lavender callouts.

**Content (v4's visuals, re-flowed):**
1. **Column 1:** the question; Figure 1, the 880 vs 192 ms bars with 32× and 4.6×; Figure 2, the contract chart.
2. **Column 2:** Figure 3, how Opus found 32× (search staircase on the recorded parent path, beside the verified
   strip); Figure 4, inside one decode step (CPU/GPU lanes to time scale, the 2×2 interaction grid and the A10
   hypothesis).
3. **Column 3:** Figure 5, the verified frontier (equal-token and equal-time panels, with the exact dominated region
   and the memory strip); take-home points; limits and references.

**Checks:**
- Every number passes the `poster_data.py` assertions; the visuals are v4's, whose encodings the v4 fact-check verified.
- New wording checked against the paper. Prefill is "recorded, not scored". At 300 s "the leaders are not
  separated", not tied. The contract caption counts its grey arrows from the data (8; Baseline and GPT A24 lie off
  scale).

## Version 4: A1 portrait, visual-first redesign

| File | What it is |
|---|---|
| `out_v4/autopareto_poster_v4_A1.pdf` | Print file: A1 portrait, 594 × 841 mm (within the 61 × 85 cm limit), vector, fonts embedded |
| `out_v4/autopareto_poster_v4_A1.svg` | Editable vector copy |
| `out_v4/autopareto_poster_v4_preview.png` | Full-size preview, 150 dpi (3507 × 4966 px) |
| `out_v4/autopareto_poster_v3_vs_v4.png` | v3 and v4 side by side at the same physical scale |
| `build_poster_v4.py` | Layout and every chart and diagram (matplotlib, page coordinates in mm) |
| `poster_data.py` | All numbers, read from the paper's records and asserted against the paper |

```bash
/Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python poster/build_poster_v4.py
```

**Reading path.**
- **15 seconds:** the hero row. One tinted box asks the question and answers it, next to proportional latency bars:
  880 ms against 192 ms, with the untimed 164 ms prefill hatched, and the two numbers 32× (decode steps) and 4.6×
  (whole generation).
- **One minute:** three full-width visuals:
  1. how Opus found 32×;
  2. inside one decode step;
  3. the verified frontier, next to "the contract decides".

**Removed from v3.**
- the BPB, dominance, contract and speed-up-estimator equations, and the κ and Amdahl-bound derivations;
- the methodology paragraphs and the search-loop figure;
- both tables (the agents table and the results table with SDs);
- the key-findings box and the separate exactness callout;
- the long captions.

Words fell from 1,021 to 506. The paper keeps all of it.

**Redesigned.**
- **Composition:** portrait bands sized by importance, replacing three equal columns; one tinted box instead of five.
- **How Opus found 32×:**
  - a staircase of Opus's search records (decode speed-up per attempt, log scale) with six labelled milestones and
    phase bands;
  - beside it, on the same axis, the final verification (13-recipe test) for every recipe, so the search history and
    the verified results are visibly separate;
  - GPT-5.6's recipes reach 1.9 to 2.5× with no inference code.
- **Inside one decode step** (new):
  - CPU and GPU lanes for eager decoding, CUDA graph replay, and replay plus fusion. Each row's length is its time per
    decoded token, to scale (1/decode speed-up, ablation medians); the blocks are schematic.
  - In eager decoding every kernel is launched by the CPU and the GPU idles between kernels.
  - With replay, one CPU call replays the recorded step and the same 290 kernels run back to back.
  - With fusion, the small kernels merge, leaving 78.
  - Two insets sit in the time saved: "record once, replay every token" and "fuse the small kernels".
  - A measured 2×2 grid shows the interaction: fusion gains ×1.39 without the graph and ×2.66 with it.
  - Opus's own A10 diagnosis is quoted.
  - Nothing depicts memory transfers or offloading.
- **Frontier:**
  - two large panels, equal tokens and equal time;
  - the region dominated in quality and speed is shaded, and the dashed line is its exact boundary;
  - memory, the third objective, is a strip below;
  - the claim is only that no recipe leads every objective.
- **The contract decides** (new): arrows run from each recipe's 300 s point (tokens it trained on in 300 s, its BPB)
  to its equal-token point (201.3M tokens). At 300 s the recipes are not separated, but saw 88M to 198M tokens; at
  equal tokens AR-ref learns most per token.

**Type at A1.**
- Title 46 pt, section titles 30 pt, section lines 20 pt.
- Chart labels 14 to 17 pt.
- The smallest text (inset block labels, notes) is 12.5 to 13.5 pt.

## Version 3: landscape, academic layout, Opus-focused

Built after feedback on version 2, with Arjun's ICML poster (120 × 72 cm landscape) as the reference for layout and
proportions.

- **Size:** 850 × 610 mm, landscape.
  - **Check with the organizers:** the workshop guideline is 61 cm wide × 85 cm high, portrait, and a landscape sheet
    85 cm wide exceeds that width.
  - The file is vector, so it scales without loss; at 61 cm wide it would be 61 × 43.8 cm.
- **Layout (as in the reference):**
  - a header band with the paper title, author and venue, and a footer band;
  - three equal columns with centred section titles over thin rules;
  - justified text and numbered figures and tables with captions;
  - lavender callouts.
- **Type:** scaled from the reference by 0.71, its width ratio. Title 44 pt, section titles 25 pt, body 17.5 pt,
  captions 14.5 pt.
- **Math:** Computer Modern (matplotlib mathtext, embedded):
  - BPB;
  - the objective vector;
  - Pareto dominance;
  - the two training contracts;
  - the bracketed speed-up estimator;
  - the graph and fusion interaction ratio κ;
  - the whole-generation latency model, with its Amdahl-style bound (derived, labelled).
- **Content:** Opus-focused. The version 2 E1 chart (GPT A22 against AR-ref) is gone.
  1. Column 1: setting, method and verification.
  2. Column 2, the discovery:
     - Figure 2, all 27 Opus attempts on all three objectives: quality plateaus after A7, the scored decode speed-up
       climbs to 32× while whole generation stays near 5×, and memory falls below 100 MiB;
     - Figure 3, a to-scale timeline of one decoded token under each graph and fusion configuration, with κ.
  3. Column 3, verified results:
     - Table 2 (means ± SD);
     - the 13-recipe Pareto projections;
     - the 32× against 4.6× decomposition;
     - the exactness check, limitations and references.

| File | What it is |
|---|---|
| `out_v3/autopareto_poster_v3_landscape.pdf` | Print file: 850 × 610 mm, one page, vector, fonts embedded |
| `out_v3/autopareto_poster_v3_landscape.svg` | Editable vector copy |
| `out_v3/autopareto_poster_v3_preview.png`, `..._small.png` | Previews |
| `build_poster_v3.py` | Version 3 layout; same `poster_data.py` checks, plus checks for κ, the Opus records and Table 2 of the paper |

```bash
/Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python poster/build_poster_v3.py
```

## Version 2: 61 × 85 cm, less text

The workshop's guideline is 61 cm × 85 cm portrait, no larger. Feedback on version 1 was that it carried far too much
information and text. Version 2 is built for that.

- **Size:** exactly 610 × 850 mm, the maximum allowed.
- **Text:** about 455 words including labels and footer, against 1,076 in version 1. Each section has one sentence.
- **Type:** larger. Section headings are 44 pt, ledes 26 pt, figure labels 19 to 22 pt; the minimum is 15.5 pt.
- **Content:** only the strongest visuals:
  1. the E1 chart (budget story);
  2. two agent timelines, with Opus's search-record speed-ups as bars;
  3. the 13-recipe scatter with seed means only, just the key recipes labelled, plus a five-bar memory panel;
  4. the CUDA graph × fusion grid, and the 880 vs 192 ms whole-generation bars.
- **Removed:** the search-loop diagram, the quotes, the fusion schematic, per-seed points and timing-flag rings, and most
  captions. Their facts are in the paper; the scatter note points there.

| File | What it is |
|---|---|
| `out_v2/autopareto_poster_v2_61x85cm.pdf` | Print file: 610 × 850 mm, one page, all vector, fonts embedded |
| `out_v2/autopareto_poster_v2_61x85cm.svg` | Editable vector copy |
| `out_v2/autopareto_poster_v2_preview.png`, `..._small.png` | Previews (110 dpi; reduced) |
| `build_poster_v2.py` | Version 2 layout; same `poster_data.py` checks as version 1 |

```bash
/Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python poster/build_poster_v2.py
```

Every number in version 2 is one of version 1's checked values (see `CLAIMS_CHECKLIST.md`, which also lists version
2's new wording).

## Version 1: A1, detailed (kept as is)

| File | What it is |
|---|---|
| `out/autopareto_poster_A1.pdf` | Print file: 594 × 841 mm, one page, all vector, fonts embedded |
| `out/autopareto_poster_A1.svg` | Editable vector copy (text kept as text; needs Avenir Next, which ships with macOS) |
| `out/autopareto_poster_preview.png` | Full-page preview, 110 dpi |
| `out/autopareto_poster_preview_small.png` | Reduced preview: roughly what a passer-by sees from 3 m |
| `build_poster.py` | Layout, figures and diagrams (matplotlib; page coordinates in mm) |
| `poster_data.py` | Every number on the poster, read from the paper's frozen records and asserted against the paper's macros |
| `CLAIMS_CHECKLIST.md` | Each printed number and claim, its source and its paper location |

## Build

```bash
cd /Users/Rami/Documents/dev/autopareto-camera-ready
/Users/Rami/Documents/dev/autopareto/paper/.venv/bin/python poster/build_poster.py
```

It takes about 15 s. No GPU, no network and no new data. `poster_data.py` stops the build if any printed value differs
from the paper (for example the 4.6×, the 625 kernels or a quote that is not verbatim in its attempt record). It reads
the Opus campaign history through `analysis/cr_r9.py`, so `AUTOPARETO_EVIDENCE_ROOT` must point at the evidence
checkout (default: `../autopareto`).

## Reading order and design

A reader at 3 m sees four things, top to bottom: the name, the two numbers **32×** and **4.6×**, the one-line takeaway,
and four numbered sections. Each section opens with a heading and a one-sentence finding, so the poster can be read in
about 30 seconds from headings and ledes alone. The figures reward a closer look.

1. **The budget shapes the scientist.** The search loop (a four-step cycle), the E1 learning curves and the agents'
   own words. The E1 chart is the clearest evidence for the budget story. At every token budget GPT A22 learns less
   per token than AR-ref. Its 300 s point lands on 158.1M tokens against AR-ref's 88.0M, so at 300 s the two are not
   separated. The x-axis ticks are E1's budgets, which match the 300 s token counts.
2. **One benchmark for all 13 recipes.** The paper's Figure 3, redrawn at poster scale: the two panels (equal tokens,
   equal time), per-seed points, timing-flag rings and the frontier of the seed means. A third panel shows inference
   memory, the third objective, which the scatter cannot show. The benchmark, flag and correctness facts sit in one
   note underneath.
3. **Two agents, two discoveries.** The paper's Figure 2, simplified into two rails aligned by phase (shrink, quality
   limit, next step) rather than by attempt number, so the divergence is visible at a glance. Opus's engine steps
   carry log-scale bars of their decode speed-up from the search records (this is the paper's Figure 4a, merged in).
   GPT-5.6's rail has no per-attempt speed bars: its campaign was timed on the earlier 256/256 workload, so both
   agents are compared only by their re-timed results in the 13-recipe test, in the band at the bottom.
4. **Why 32× decoding is 4.6× generation.** New diagrams:
   - a 2×2 grid (CUDA graph on/off × kernel fusion on/off) built from the profiler census and the ablation session.
     It keeps graph replay (CPU launch calls 284 to 2) separate from fusion (GPU kernels 290 to 78), and its four
     speed-ups (2.0×, 2.7×, 11.6×, 30.8×) show how the two interact;
   - a schematic of fusion inside one layer's decode step, following the compiled regions described in Opus A23's
     source: the matrix-vector multiplies stay as they are, and the small kernels between them are fused;
   - the whole-generation bars, to scale, from the 13-recipe test: 880 ms against 192 ms, of which 164 ms is the
     unscored prefill where Opus records its CUDA graph.

**Design system.**
- **Colour:** the paper's method colours: Baseline grey, AutoResearch orange, classical search green, GPT-5.6 blue,
  Opus violet. Everything else is neutral. Systems diagrams use slate for matrix-vector multiplies, grey for small
  kernels and Opus violet for fused or graph work. Hatching marks unscored prefill and speed-uncertain records.
- **Typography:** Avenir Next in five weights. The title is 128 pt, the hero numbers 100 pt, section headings 40 pt
  and ledes 23 pt. Figure labels are 15 to 19 pt; the smallest text, on a few diagram labels, is 13.5 pt (4.8 mm).
- **Margins:** 18 mm grid; the closest text is 16.3 mm from the bottom edge and 17.4 mm from the left (measured in the PDF).

**Print notes.**
- Size and colour: exactly A1, 594 × 841 mm. The file is RGB; let the print shop convert to its profile. The dark
  header is an inset card, not full bleed, so no bleed is needed.
- Format: all vector, 0.12 MB, with no raster images and no Type 3 fonts.
- Fonts: Avenir Next is embedded as TrueType subsets. Its licence flag permits print embedding (fsType 4). The font
  is extracted from the macOS system collection at build time and never stored in the repository.

## Not on the poster, and why

- **QR code:** no verified public URL exists yet. The paper's Part 2 says the release is pending, and the OpenReview
  forum was not confirmed for this poster. The footer has room for one: once a URL is confirmed, a QR code can be added
  next to "References and AI use".
- **Historical 256/256 decode results:** excluded. Every comparison between recipes uses the unified 1,024/128 test.
- **Downstream benchmarks and the statistics tables:** left to the paper, to keep the poster readable at distance. The
  poster states only the paper's headline statistics (per-seed consistency, intervals at 300 s).

## For the author to confirm

- The footer's AI-use line is new wording, adapted from the paper's disclosure statement.
- Two small inaccuracies in the paper surfaced in the poster's fact-check (see `CLAIMS_CHECKLIST.md`):
  - §1.3's A7 quote describes A6.
  - Appendix E says "launches 625 kernels from the CPU"; the census records 618 launch calls for 625 kernels.

  The poster states both correctly; whether to correct the paper is your call.
- The header shows the author name and email exactly as printed in the paper.
- Nothing has been committed. Commit `poster/` (excluding `poster/.mplcache/`, which is ignored) once approved.
