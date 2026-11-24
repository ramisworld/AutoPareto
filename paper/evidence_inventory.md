# Internal audit: not an anonymized submission asset

# Evidence inventory

Snapshot generated 2026-09-06T12:38:02.850468+00:00. Cutoff 2026-09-06T23:59:00+12:00. This is an as-of snapshot before the final cutoff, not a claim to include future results.

3331 hashed artifacts; 318 experiment rows in the most complete recovered database; 21 equal-token run bundles; 28 budget-history rows across all namespaces. Duplicate database snapshots are inventoried, never counted as new experiments. Full raw tables are in generated/db_*.json; normalized records preserve source paths and raw field names.

## Regime and eligibility

Historical experiments use valid_result and exclude_from_analysis flags, retained alongside failures. The equal-token panel requires token arithmetic, source/checkpoint pairing, inference correctness, full task coverage, and cutoff checks. New study quality evidence is distinct from frontier eligibility. Earlier launches are separate strata. No absent value is imputed. All available numerical and textual fields, complete code diffs, prompts and raw records are retained in generated JSON.

| Namespace | History rows | Completed labels | Analysis use |
| --- | --- | --- | --- |
| runs/ABANDONED_NONSTUDY/pair-a-official-20260905 | 1 | 0 | Separate diagnostic/earlier launch; not pooled |
| runs/pair-a | 4 | 1 | Separate diagnostic/earlier launch; not pooled |
| runs/pair-a-official-20260906-fixed-warmup-v2 | 8 | 8 | Current descriptive study |
| runs/pair-a-official-20260906-fresh | 4 | 4 | Separate diagnostic/earlier launch; not pooled |
| runs/prestudy_debug/pair-a-interrupted-20260905 | 4 | 1 | Separate diagnostic/earlier launch; not pooled |
| runs/prestudy_validation/pair-a-api-repair-20260905-r10/state | 2 | 2 | Separate diagnostic/earlier launch; not pooled |
| runs/prestudy_validation/pair-a-api-repair-20260905-r7/state | 2 | 1 | Separate diagnostic/earlier launch; not pooled |
| runs/prestudy_validation/pair-a-api-repair-20260905-r9/state | 1 | 1 | Separate diagnostic/earlier launch; not pooled |
| runs/prestudy_validation/pair-a-fixed-warmup-t-20260906 | 1 | 1 | Separate diagnostic/earlier launch; not pooled |
| runs/prestudy_validation/sol-high-bracketed-final-smoke-20260906 | 1 | 1 | Separate diagnostic/earlier launch; not pooled |

## Historical experiment ledger

| ID | Campaign / seed | Method / phase | Status / eligible flags | Parent | Params | Tokens | BPB | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | baseline-first / 0 | setup_oom / setup | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 2 | baseline-a40-async / 0 | setup_oom / setup | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 3 | baseline-a40-seed0 / 0 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 93323264 | 1.2067114536967076 | results/candidates/baseline-a40-seed0/0001-train.py |
| 4 | baseline-a40-seed0 / 1 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 92798976 | 1.2127299916819494 | results/candidates/baseline-a40-seed0/0002-train.py |
| 5 | baseline-a40-seed0 / 2 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 92274688 | 1.212275397895648 | results/candidates/baseline-a40-seed0/0003-train.py |
| 6 | baseline-a40-seed0 / 3 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 92798976 | 1.1931848475547713 | results/candidates/baseline-a40-seed0/0004-train.py |
| 7 | baseline-a40-seed0 / 4 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 92274688 | 1.2094987883539934 | results/candidates/baseline-a40-seed0/0005-train.py |
| 8 | baseline-a40-cached-fresh-20260722 / 0 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 90177536 | 1.2137349928434058 | results/candidates/baseline-a40-cached-fresh-20260722/0001-train.py |
| 9 | baseline-a40-cached-fresh-20260722 / 1 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 90177536 | 1.2189262480600007 | results/candidates/baseline-a40-cached-fresh-20260722/0002-train.py |
| 10 | baseline-a40-cached-fresh-20260722 / 2 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 90177536 | 1.2169696961309011 | results/candidates/baseline-a40-cached-fresh-20260722/0003-train.py |
| 11 | baseline-a40-cached-fresh-20260722 / 3 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 90701824 | 1.1966832149933828 | results/candidates/baseline-a40-cached-fresh-20260722/0004-train.py |
| 12 | baseline-a40-cached-fresh-20260722 / 4 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 90701824 | 1.213109342994092 | results/candidates/baseline-a40-cached-fresh-20260722/0005-train.py |
| 13 | discovery-reference-seed42-20260722 / 42 | baseline / baseline | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 94371840 | 1.1892983259238852 | results/candidates/discovery-reference-seed42-20260722/0001-train.py |
| 14 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | interrupted / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 15 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 13 | 50332176 | 93847552 | 1.2445483608068502 | results/candidates/autoresearch-seed42-20260723/0002-train.py |
| 16 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 13 | 50332176 | 93847552 | 1.1908194994994996 | results/candidates/autoresearch-seed42-20260723/0003-train.py |
| 17 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 13 | 50332176 | 94371840 | 1.1890557083878242 | results/candidates/autoresearch-seed42-20260723/0004-train.py |
| 18 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 17 | 50332176 | 90963968 | 1.1716653252934937 | results/candidates/autoresearch-seed42-20260723/0005-train.py |
| 19 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 18 | 50332176 | 88604672 | 1.099018855735464 | results/candidates/autoresearch-seed42-20260723/0006-train.py |
| 20 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 19 | 50332176 | 88735744 | 1.0998658906719363 | results/candidates/autoresearch-seed42-20260723/0007-train.py |
| 21 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 19 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 22 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 19 | 50332176 | 88735744 | 1.0986702394247787 | results/candidates/autoresearch-seed42-20260723/0009-train.py |
| 23 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 22 | 50332176 | 82182144 | 1.1058997335920697 | results/candidates/autoresearch-seed42-20260723/0010-train.py |
| 24 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 22 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 25 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 22 | 50332176 | 88604672 | 1.0988647520856358 | results/candidates/autoresearch-seed42-20260723/0012-train.py |
| 26 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 22 | 50332176 | 88735744 | 1.0985479186876161 | results/candidates/autoresearch-seed42-20260723/0013-train.py |
| 27 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 26 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 28 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 26 | 50332176 | 88473600 | 1.098233192293993 | results/candidates/autoresearch-seed42-20260723/0015-train.py |
| 29 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 28 | 50332176 | 88473600 | 1.0976744953448063 | results/candidates/autoresearch-seed42-20260723/0016-train.py |
| 30 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 29 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 31 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 29 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 32 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 29 | 50332176 | 88735744 | 1.0968251425437556 | results/candidates/autoresearch-seed42-20260723/0019-train.py |
| 33 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 32 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 34 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 32 | 50332176 | 88735744 | 1.0967681235011013 | results/candidates/autoresearch-seed42-20260723/0021-train.py |
| 35 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 36 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | interrupted / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 37 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 38 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 39 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 40 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 34 | 50332176 | 88604672 | 1.1037094687098055 | results/candidates/autoresearch-seed42-20260723/0027-train.py |
| 41 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 42 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 43 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 44 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 45 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 46 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 47 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 48 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 34 | 50332176 | 88080384 | 1.0981045739725503 | results/candidates/autoresearch-seed42-20260723/0035-train.py |
| 49 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 50 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 51 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 34 | 50332176 | 87949312 | 1.0980312867473538 | results/candidates/autoresearch-seed42-20260723/0038-train.py |
| 52 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 34 | 50332176 | 88735744 | 1.0967847138311442 | results/candidates/autoresearch-seed42-20260723/0039-train.py |
| 53 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 54 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 55 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 56 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 57 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 58 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 59 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 60 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 34 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 61 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 34 | 50332176 | 88735744 | 1.096754145494427 | results/candidates/autoresearch-seed42-20260723/0048-train.py |
| 62 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 61 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 63 | autoresearch-seed42-20260723 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 61 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 64 | autoresearch-diagnostic5-seed42-20260724 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 61 | 50332176 | 88735744 | 1.095215 | results/candidates/autoresearch-diagnostic5-seed42-20260724/0001-train.py |
| 65 | autoresearch-diagnostic5-seed42-20260724 / 42 | autoresearch / campaign | crash / valid=0, excluded=1 | 64 | 50332176 | 88735744 | 1.0948561434742032 | results/candidates/autoresearch-diagnostic5-seed42-20260724/0002-train.py |
| 66 | autoresearch-diagnostic5-seed42-20260724 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 65 | 50332176 | 88604672 | 1.0952247206829955 | results/candidates/autoresearch-diagnostic5-seed42-20260724/0003-train.py |
| 67 | autoresearch-diagnostic5-seed42-20260724 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 65 | 50332176 | 88866816 | 1.0953718755313648 | results/candidates/autoresearch-diagnostic5-seed42-20260724/0004-train.py |
| 68 | autoresearch-diagnostic5-seed42-20260724 / 42 | autoresearch / campaign | completed / valid=1, excluded=0 | 65 | 50332176 | 88866816 | 1.0949760675404565 | results/candidates/autoresearch-diagnostic5-seed42-20260724/0005-train.py |
| 69 | setup-checkpoint-mount-20260727 / 0 | setup_checkpoint_mount / setup | crash / valid=0, excluded=1 | 65 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 70 | setup-checkpoint-mount-20260727 / 1 | setup_checkpoint_mount / setup | crash / valid=0, excluded=1 | 65 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 71 | setup-checkpoint-mount-20260727 / 2 | setup_checkpoint_mount / setup | interrupted / valid=0, excluded=1 | 65 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 72 | confirmation-exp65-seeds012-20260725 / 0 | confirmation / confirmation | crash / valid=0, excluded=1 | 65 | 50332176 | 88473600 | 1.103682765078145 | results/candidates/confirmation-exp65-seeds012-20260725/0001-train.py |
| 73 | confirmation-exp65-seeds012-20260725 / 1 | confirmation / confirmation | crash / valid=0, excluded=1 | 65 | 50332176 | 88473600 | 1.0989055050756487 | results/candidates/confirmation-exp65-seeds012-20260725/0002-train.py |
| 74 | confirmation-exp65-seeds012-20260725 / 2 | confirmation / confirmation | crash / valid=0, excluded=1 | 65 | 50332176 | 88604672 | 1.1025230117120575 | results/candidates/confirmation-exp65-seeds012-20260725/0003-train.py |
| 75 | confirmation-exp68-seeds012-20260725 / 0 | confirmation / confirmation | completed / valid=1, excluded=0 | 68 | 50332176 | 88473600 | 1.102901075499148 | results/candidates/confirmation-exp68-seeds012-20260725/0001-train.py |
| 76 | confirmation-exp68-seeds012-20260725 / 1 | confirmation / confirmation | crash / valid=0, excluded=1 | 68 | 50332176 | 88473600 | 1.0989922495467666 | results/candidates/confirmation-exp68-seeds012-20260725/0002-train.py |
| 77 | confirmation-exp68-seeds012-20260725 / 2 | confirmation / confirmation | completed / valid=1, excluded=0 | 68 | 50332176 | 88473600 | 1.1023826571045643 | results/candidates/confirmation-exp68-seeds012-20260725/0003-train.py |
| 78 | ablation-batch-only-seeds012-20260725 / 0 | ablation / ablation | completed / valid=1, excluded=0 | 68 | 50332176 | 88080384 | 1.105105785901319 | results/candidates/ablation-batch-only-seeds012-20260725/0001-train.py |
| 79 | ablation-batch-only-seeds012-20260725 / 1 | ablation / ablation | completed / valid=1, excluded=0 | 68 | 50332176 | 88080384 | 1.1018302497525296 | results/candidates/ablation-batch-only-seeds012-20260725/0002-train.py |
| 80 | ablation-batch-only-seeds012-20260725 / 2 | ablation / ablation | completed / valid=1, excluded=0 | 68 | 50332176 | 88080384 | 1.1049674786609989 | results/candidates/ablation-batch-only-seeds012-20260725/0003-train.py |
| 81 | ablation-nonbatch-only-seeds012-20260725 / 0 | ablation / ablation | completed / valid=1, excluded=0 | 68 | 50332176 | 94371840 | 1.2175053097473636 | results/candidates/ablation-nonbatch-only-seeds012-20260725/0001-train.py |
| 82 | ablation-nonbatch-only-seeds012-20260725 / 1 | ablation / ablation | completed / valid=1, excluded=0 | 68 | 50332176 | 94371840 | 1.2031014086353828 | results/candidates/ablation-nonbatch-only-seeds012-20260725/0002-train.py |
| 83 | ablation-nonbatch-only-seeds012-20260725 / 2 | ablation / ablation | completed / valid=1, excluded=0 | 68 | 50332176 | 94371840 | 1.2357992936844187 | results/candidates/ablation-nonbatch-only-seeds012-20260725/0003-train.py |
| 84 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 13 | 39846160 | 95420416 | 1.197972779425378 | results/candidates/autopareto-seed42-20260727/0001-train.py |
| 85 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 84 | 34603152 | 98566144 | 1.1984678093789436 | results/candidates/autopareto-seed42-20260727/0002-train.py |
| 86 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 13 | 50332176 | 90701824 | 1.1727378714675079 | results/candidates/autopareto-seed42-20260727/0003-train.py |
| 87 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 86 | 50332176 | 88342528 | 1.0992980186424477 | results/candidates/autopareto-seed42-20260727/0004-train.py |
| 91 | setup-autopareto-seed42-20260730-tokenizer-cache / 42 | setup_tokenizer_cache / setup | crash / valid=0, excluded=1 | 87 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 92 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 87 | 39846160 | 89522176 | 1.1040101100692374 | results/candidates/autopareto-seed42-20260727/0005-train.py |
| 93 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 87 | 50332176 | 84344832 | 1.0968816573323883 | results/candidates/autopareto-seed42-20260727/0006-train.py |
| 94 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 92 | 39846160 | 86114304 | 1.098776118658719 | results/candidates/autopareto-seed42-20260727/0007-train.py |
| 95 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 85 | 34603152 | 88866816 | 1.10565624705647 | results/candidates/autopareto-seed42-20260727/0008-train.py |
| 96 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 93 | 50332176 | 84148224 | 1.094998394758386 | results/candidates/autopareto-seed42-20260727/0009-train.py |
| 97 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 96 | 50332176 | 84148224 | 1.0941956306093856 | results/candidates/autopareto-seed42-20260727/0010-train.py |
| 98 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 97 | 50332176 | 83755008 | 1.1002052757261853 | results/candidates/autopareto-seed42-20260727/0011-train.py |
| 99 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 94 | 39846160 | 86114304 | 1.096379801845374 | results/candidates/autopareto-seed42-20260727/0012-train.py |
| 100 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 95 | 34603152 | 88932352 | 1.1038952378615163 | results/candidates/autopareto-seed42-20260727/0013-train.py |
| 101 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 99 | 36962574 | 96468992 | 1.0970767703590418 | results/candidates/autopareto-seed42-20260727/0014-train.py |
| 102 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 101 | 18874476 | 141492224 | 1.1120012735464326 | results/candidates/autopareto-seed42-20260727/0015-train.py |
| 103 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 102 | 17301610 | 142868480 | 1.1228413979653016 | results/candidates/autopareto-seed42-20260727/0016-train.py |
| 104 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 103 | 9175112 | 145489920 | 1.1810906497739186 | results/candidates/autopareto-seed42-20260727/0017-train.py |
| 105 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 104 | 9175112 | 143327232 | 1.187811462122034 | results/candidates/autopareto-seed42-20260727/0018-train.py |
| 106 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 101 | 23888014 | 117047296 | 1.1111777433968115 | results/candidates/autopareto-seed42-20260727/0019-train.py |
| 107 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 97 | 50332176 | 77037568 | 1.1014960792279884 | results/candidates/autopareto-seed42-20260727/0020-train.py |
| 108 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 101 | 31850638 | 100007936 | 1.1003845377646218 | results/candidates/autopareto-seed42-20260727/0021-train.py |
| 109 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 97 | 50332176 | 84082688 | 1.0939814996982178 | results/candidates/autopareto-seed42-20260727/0022-train.py |
| 110 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 109 | 50332176 | 84213760 | 1.0956503885523574 | results/candidates/autopareto-seed42-20260727/0023-train.py |
| 111 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 109 | 50332176 | 84475904 | 1.094030816486994 | results/candidates/autopareto-seed42-20260727/0024-train.py |
| 112 | autopareto-seed42-20260727 / 42 | autopareto / campaign | completed / valid=1, excluded=0 | 109 | 50332176 | 84017152 | 1.0943073777201133 | results/candidates/autopareto-seed42-20260727/0025-train.py |
| 113 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 8 | 50332176 | 86769664 | 1.1062543668726008 | results/candidates/autopareto-seed0-20260731/0001-train.py |
| 114 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 113 | 50332176 | 86638592 | 1.1061395795946751 | results/candidates/autopareto-seed0-20260731/0002-train.py |
| 115 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 113 | 39846160 | 87949312 | 1.1070624539100138 | results/candidates/autopareto-seed0-20260731/0003-train.py |
| 116 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 113 | 47186446 | 96993280 | 1.1004748069553763 | results/candidates/autopareto-seed0-20260731/0004-train.py |
| 117 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 116 | 47186446 | 96993280 | 1.1003487738010664 | results/candidates/autopareto-seed0-20260731/0005-train.py |
| 118 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 116 | 26345772 | 154533888 | 1.0977683792435076 | results/candidates/autopareto-seed0-20260731/0006-train.py |
| 119 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 118 | 24576298 | 176816128 | 1.096585475985295 | results/candidates/autopareto-seed0-20260731/0007-train.py |
| 120 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 119 | 11534472 | 203685888 | 1.1573142984371514 | results/candidates/autopareto-seed0-20260731/0008-train.py |
| 121 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 120 | 19661000 | 202506240 | 1.1067982125557727 | results/candidates/autopareto-seed0-20260731/0009-train.py |
| 122 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 121 | 19661000 | 204341248 | 1.1053898614245676 | results/candidates/autopareto-seed0-20260731/0010-train.py |
| 123 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 119 | 24576298 | 176029696 | 1.0963848158981664 | results/candidates/autopareto-seed0-20260731/0011-train.py |
| 124 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 119 | 24576298 | 176685056 | 1.0963036991575175 | results/candidates/autopareto-seed0-20260731/0012-train.py |
| 125 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 124 | 24576298 | 176816128 | 1.0954397474837 | results/candidates/autopareto-seed0-20260731/0013-train.py |
| 126 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 125 | 24576298 | 177078272 | 1.0972554884740537 | results/candidates/autopareto-seed0-20260731/0014-train.py |
| 127 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 125 | 24576298 | 176816128 | 1.0958214810899216 | results/candidates/autopareto-seed0-20260731/0015-train.py |
| 128 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 125 | 24576298 | 176947200 | 1.0991642512857476 | results/candidates/autopareto-seed0-20260731/0016-train.py |
| 129 | autopareto-seed0-20260731 / 0 | autopareto / campaign | crash / valid=0, excluded=1 | 120 | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 130 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 120 | 29360392 | 159514624 | 1.0956682214132973 | results/candidates/autopareto-seed0-20260731/0018-train.py |
| 131 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 130 | 29360392 | 159776768 | 1.0953582319191209 | results/candidates/autopareto-seed0-20260731/0019-train.py |
| 132 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 131 | 29360392 | 159645696 | 1.0933824480504735 | results/candidates/autopareto-seed0-20260731/0020-train.py |
| 133 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 132 | 29360392 | 159776768 | 1.0938047291112776 | results/candidates/autopareto-seed0-20260731/0021-train.py |
| 134 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 132 | 29360392 | 159776768 | 1.093294337780448 | results/candidates/autopareto-seed0-20260731/0022-train.py |
| 135 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 134 | 24117384 | 161873920 | 1.0994866245558268 | results/candidates/autopareto-seed0-20260731/0023-train.py |
| 136 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 134 | 17891526 | 251789312 | 1.116323646635026 | results/candidates/autopareto-seed0-20260731/0024-train.py |
| 137 | autopareto-seed0-20260731 / 0 | autopareto / campaign | completed / valid=1, excluded=0 | 136 | 26214662 | 197656576 | 1.0974344667032045 | results/candidates/autopareto-seed0-20260731/0025-train.py |
| 138 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 9 | 50332176 | 88473600 | 1.1016356487007881 | results/candidates/autopareto-seed1-20260731/0001-train.py |
| 139 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 138 | 47186446 | 98959360 | 1.101113915648037 | results/candidates/autopareto-seed1-20260731/0002-train.py |
| 140 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 139 | 26345772 | 157286400 | 1.0991319295081716 | results/candidates/autopareto-seed1-20260731/0003-train.py |
| 141 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 140 | 24576298 | 180486144 | 1.0977507601713636 | results/candidates/autopareto-seed1-20260731/0004-train.py |
| 142 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 141 | 11534472 | 215482368 | 1.1532424110045243 | results/candidates/autopareto-seed1-20260731/0005-train.py |
| 143 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 141 | 19661000 | 211025920 | 1.1110562368142414 | results/candidates/autopareto-seed1-20260731/0006-train.py |
| 144 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 141 | 24576298 | 180224000 | 1.0977430424757049 | results/candidates/autopareto-seed1-20260731/0007-train.py |
| 145 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 144 | 24576298 | 179830784 | 1.0980728735935417 | results/candidates/autopareto-seed1-20260731/0008-train.py |
| 146 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 144 | 24576298 | 180355072 | 1.098524940784983 | results/candidates/autopareto-seed1-20260731/0009-train.py |
| 147 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 144 | 17301610 | 186384384 | 1.106084217614867 | results/candidates/autopareto-seed1-20260731/0010-train.py |
| 148 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 143 | 14680136 | 206045184 | 1.1217505455598882 | results/candidates/autopareto-seed1-20260731/0011-train.py |
| 149 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 148 | 14680136 | 205520896 | 1.121629977731109 | results/candidates/autopareto-seed1-20260731/0012-train.py |
| 150 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 142 | 9175112 | 286916608 | 1.1534318103059629 | results/candidates/autopareto-seed1-20260731/0013-train.py |
| 151 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 142 | 10748038 | 227016704 | 1.1771338483378084 | results/candidates/autopareto-seed1-20260731/0014-train.py |
| 152 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 151 | 3538980 | 250740736 | 1.342846933180371 | results/candidates/autopareto-seed1-20260731/0015-train.py |
| 153 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 152 | 3342370 | 245366784 | 1.5014775232212512 | results/candidates/autopareto-seed1-20260731/0016-train.py |
| 154 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 144 | 24576298 | 180486144 | 1.0970906197729535 | results/candidates/autopareto-seed1-20260731/0017-train.py |
| 155 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 154 | 24576298 | 180486144 | 1.100262387153288 | results/candidates/autopareto-seed1-20260731/0018-train.py |
| 156 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 154 | 24576298 | 180748288 | 1.0972010162603174 | results/candidates/autopareto-seed1-20260731/0019-train.py |
| 157 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 154 | 24576298 | 180355072 | 1.0976329053037672 | results/candidates/autopareto-seed1-20260731/0020-train.py |
| 158 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 154 | 24576298 | 180748288 | 1.0967395498165118 | results/candidates/autopareto-seed1-20260731/0021-train.py |
| 159 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 158 | 24576298 | 180224000 | 1.0968207272593342 | results/candidates/autopareto-seed1-20260731/0022-train.py |
| 160 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 158 | 24576298 | 180224000 | 1.0967957035869536 | results/candidates/autopareto-seed1-20260731/0023-train.py |
| 161 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 158 | 24576298 | 180748288 | 1.096086669185082 | results/candidates/autopareto-seed1-20260731/0024-train.py |
| 162 | autopareto-seed1-20260731 / 1 | autopareto / campaign | completed / valid=1, excluded=0 | 161 | 24576298 | 181010432 | 1.0957550904836137 | results/candidates/autopareto-seed1-20260731/0025-train.py |
| 163 | tpe-seed42-20260805 / 42 | tpe / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 164 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 82313216 | 1.2935241860063231 | results/candidates/tpe-seed42-20260805/0002-train.py |
| 165 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 57016976 | 65667072 | 1.1362065184873693 | results/candidates/tpe-seed42-20260805/0003-train.py |
| 166 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 79692114 | 50331648 | 1.2842428848875718 | results/candidates/tpe-seed42-20260805/0004-train.py |
| 167 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 85852980 | 52428800 | 1.1437594194591176 | results/candidates/tpe-seed42-20260805/0005-train.py |
| 168 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261070 | 145227776 | 1.1684478742027242 | results/candidates/tpe-seed42-20260805/0006-train.py |
| 169 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981772 | 121110528 | 1.2275472559599574 | results/candidates/tpe-seed42-20260805/0007-train.py |
| 170 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 70780176 | 64225280 | 1.1407665296701197 | results/candidates/tpe-seed42-20260805/0008-train.py |
| 171 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 33030928 | 128450560 | 1.1634825333074035 | results/candidates/tpe-seed42-20260805/0009-train.py |
| 172 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981964 | 119144448 | 1.1033275615166664 | results/candidates/tpe-seed42-20260805/0010-train.py |
| 173 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 86540468 | 45088768 | 1.6865693238700088 | results/candidates/tpe-seed42-20260805/0011-train.py |
| 174 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981964 | 119013376 | 1.1683734059475173 | results/candidates/tpe-seed42-20260805/0012-train.py |
| 175 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 57016976 | 65929216 | 1.1453235834855673 | results/candidates/tpe-seed42-20260805/0013-train.py |
| 176 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 20742444 | 164626432 | 1.1081249975372536 | results/candidates/tpe-seed42-20260805/0014-train.py |
| 177 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 13205612 | 204341248 | 1.1368731119528201 | results/candidates/tpe-seed42-20260805/0015-train.py |
| 178 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846668 | 114556928 | 1.1029292610093113 | results/candidates/tpe-seed42-20260805/0016-train.py |
| 179 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981964 | 117833728 | 1.1097715605046519 | results/candidates/tpe-seed42-20260805/0017-train.py |
| 180 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846668 | 114425856 | 1.1020947690588292 | results/candidates/tpe-seed42-20260805/0018-train.py |
| 181 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 56181902 | 81264640 | 1.1158639197877234 | results/candidates/tpe-seed42-20260805/0019-train.py |
| 182 | tpe-seed42-20260805 / 42 | tpe / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 183 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846284 | 113770496 | 1.0977022287835185 | results/candidates/tpe-seed42-20260805/0021-train.py |
| 184 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846668 | 112197632 | 1.1048724142647626 | results/candidates/tpe-seed42-20260805/0022-train.py |
| 185 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 26345772 | 157941760 | 1.09995565108263 | results/candidates/tpe-seed42-20260805/0023-train.py |
| 186 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 26345772 | 157941760 | 1.104989843152193 | results/candidates/tpe-seed42-20260805/0024-train.py |
| 187 | tpe-seed42-20260805 / 42 | tpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 28049516 | 120324096 | 1.1685654484558863 | results/candidates/tpe-seed42-20260805/0025-train.py |
| 188 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 42582194 | 74711040 | 1.2282757209146293 | results/candidates/cmaes-seed42-20260805/0001-train.py |
| 189 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846416 | 94896128 | 1.178317228693807 | results/candidates/cmaes-seed42-20260805/0002-train.py |
| 190 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 54067378 | 63438848 | 1.2377822350848064 | results/candidates/cmaes-seed42-20260805/0003-train.py |
| 191 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 92012544 | 1.1831505450237823 | results/candidates/cmaes-seed42-20260805/0004-train.py |
| 192 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 94896128 | 1.2620644597002666 | results/candidates/cmaes-seed42-20260805/0005-train.py |
| 193 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 20742444 | 165412864 | 1.1397614619492624 | results/candidates/cmaes-seed42-20260805/0006-train.py |
| 194 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31850638 | 106168320 | 1.173507077697738 | results/candidates/cmaes-seed42-20260805/0007-train.py |
| 195 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 92012544 | 1.1849964033443523 | results/candidates/cmaes-seed42-20260805/0008-train.py |
| 196 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846416 | 89915392 | 1.2214305907810852 | results/candidates/cmaes-seed42-20260805/0009-train.py |
| 197 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846416 | 94896128 | 1.1951229050224856 | results/candidates/cmaes-seed42-20260805/0010-train.py |
| 198 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 92012544 | 1.192263998242647 | results/candidates/cmaes-seed42-20260805/0011-train.py |
| 199 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 25559440 | 131334144 | 1.151852784363994 | results/candidates/cmaes-seed42-20260805/0012-train.py |
| 200 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 92274688 | 1.1785545347896724 | results/candidates/cmaes-seed42-20260805/0013-train.py |
| 201 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 36962574 | 103022592 | 1.1729989121132762 | results/candidates/cmaes-seed42-20260805/0014-train.py |
| 202 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21496078 | 146669568 | 1.108287424839527 | results/candidates/cmaes-seed42-20260805/0015-train.py |
| 203 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23888014 | 125042688 | 1.1731783892716967 | results/candidates/cmaes-seed42-20260805/0016-train.py |
| 204 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846416 | 94896128 | 1.192848515447125 | results/candidates/cmaes-seed42-20260805/0017-train.py |
| 205 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23068944 | 134742016 | 1.1646363176264138 | results/candidates/cmaes-seed42-20260805/0018-train.py |
| 206 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23888014 | 122683392 | 1.1217358103734685 | results/candidates/cmaes-seed42-20260805/0019-train.py |
| 207 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 92274688 | 1.190959913211741 | results/candidates/cmaes-seed42-20260805/0020-train.py |
| 208 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 36962574 | 103284736 | 1.183028697124604 | results/candidates/cmaes-seed42-20260805/0021-train.py |
| 209 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 18874572 | 169345024 | 1.1313971539571819 | results/candidates/cmaes-seed42-20260805/0022-train.py |
| 210 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 20742444 | 167247872 | 1.1294081465992294 | results/candidates/cmaes-seed42-20260805/0023-train.py |
| 211 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 39846160 | 92274688 | 1.1818071958035437 | results/candidates/cmaes-seed42-20260805/0024-train.py |
| 212 | cmaes-seed42-20260805 / 42 | cmaes / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23888014 | 125042688 | 1.1824100141809064 | results/candidates/cmaes-seed42-20260805/0025-train.py |
| 213 | motpe-seed42-20260805 / 42 | motpe / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 214 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 82313216 | 1.29599812599048 | results/candidates/motpe-seed42-20260805/0002-train.py |
| 215 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 57016976 | 65798144 | 1.1361773047675803 | results/candidates/motpe-seed42-20260805/0003-train.py |
| 216 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 79692114 | 50069504 | 1.284770156214957 | results/candidates/motpe-seed42-20260805/0004-train.py |
| 217 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 85852980 | 52297728 | 1.143309147241601 | results/candidates/motpe-seed42-20260805/0005-train.py |
| 218 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261070 | 144703488 | 1.1726147168741703 | results/candidates/motpe-seed42-20260805/0006-train.py |
| 219 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981772 | 121110528 | 1.21603683790542 | results/candidates/motpe-seed42-20260805/0007-train.py |
| 220 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 70780176 | 64094208 | 1.1408779974976013 | results/candidates/motpe-seed42-20260805/0008-train.py |
| 221 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 33030928 | 128188416 | 1.1398895083923322 | results/candidates/motpe-seed42-20260805/0009-train.py |
| 222 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981964 | 118751232 | 1.1026693492109831 | results/candidates/motpe-seed42-20260805/0010-train.py |
| 223 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 86540468 | 45088768 | 1.6891547742796635 | results/candidates/motpe-seed42-20260805/0011-train.py |
| 224 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261070 | 140771328 | 1.128737740078817 | results/candidates/motpe-seed42-20260805/0012-train.py |
| 225 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21495950 | 141819904 | 1.1129154331062863 | results/candidates/motpe-seed42-20260805/0013-train.py |
| 226 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31850638 | 109576192 | 1.2063371443923057 | results/candidates/motpe-seed42-20260805/0014-train.py |
| 227 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21496078 | 146014208 | 1.1100793727077045 | results/candidates/motpe-seed42-20260805/0015-train.py |
| 228 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981964 | 118882304 | 1.1008891722397443 | results/candidates/motpe-seed42-20260805/0016-train.py |
| 229 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23068816 | 126353408 | 1.1109125643537296 | results/candidates/motpe-seed42-20260805/0017-train.py |
| 230 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21495950 | 141819904 | 1.1120889660896578 | results/candidates/motpe-seed42-20260805/0018-train.py |
| 231 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21495950 | 141688832 | 1.1125416887580797 | results/candidates/motpe-seed42-20260805/0019-train.py |
| 232 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 20742444 | 154140672 | 1.1143218790411986 | results/candidates/motpe-seed42-20260805/0020-train.py |
| 233 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 20742444 | 154009600 | 1.115321017118144 | results/candidates/motpe-seed42-20260805/0021-train.py |
| 234 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 26345772 | 152961024 | 1.0981116329963774 | results/candidates/motpe-seed42-20260805/0022-train.py |
| 235 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 18874476 | 160825344 | 1.1095704233197372 | results/candidates/motpe-seed42-20260805/0023-train.py |
| 236 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 18874476 | 160169984 | 1.1085651159850365 | results/candidates/motpe-seed42-20260805/0024-train.py |
| 237 | motpe-seed42-20260805 / 42 | motpe / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 26345772 | 153354240 | 1.100810149724087 | results/candidates/motpe-seed42-20260805/0025-train.py |
| 238 | random-search-seed42-20260805 / 42 | random_search / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 239 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 82313216 | 1.2950466518992665 | results/candidates/random-search-seed42-20260805/0002-train.py |
| 240 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 57016976 | 65536000 | 1.137309997743618 | results/candidates/random-search-seed42-20260805/0003-train.py |
| 241 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 79692114 | 50069504 | 1.2847773531498627 | results/candidates/random-search-seed42-20260805/0004-train.py |
| 242 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 85852980 | 52297728 | 1.1430666740063211 | results/candidates/random-search-seed42-20260805/0005-train.py |
| 243 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261070 | 144703488 | 1.1616147822686733 | results/candidates/random-search-seed42-20260805/0006-train.py |
| 244 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981772 | 121110528 | 1.2272249833276923 | results/candidates/random-search-seed42-20260805/0007-train.py |
| 245 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 70780176 | 64225280 | 1.1404770969138074 | results/candidates/random-search-seed42-20260805/0008-train.py |
| 246 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 33030928 | 128450560 | 1.161306939714534 | results/candidates/random-search-seed42-20260805/0009-train.py |
| 247 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981964 | 118882304 | 1.103808660853851 | results/candidates/random-search-seed42-20260805/0010-train.py |
| 248 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 34603280 | 98828288 | 1.1997185861869173 | results/candidates/random-search-seed42-20260805/0011-train.py |
| 249 | random-search-seed42-20260805 / 42 | random_search / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 250 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 82313216 | 1.2894404904034926 | results/candidates/random-search-seed42-20260805/0013-train.py |
| 251 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23068816 | 133169152 | 1.2157640993118162 | results/candidates/random-search-seed42-20260805/0014-train.py |
| 252 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 70780176 | 66584576 | 1.398183766846925 | results/candidates/random-search-seed42-20260805/0015-train.py |
| 253 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 54067378 | 62914560 | 1.2673899433970914 | results/candidates/random-search-seed42-20260805/0016-train.py |
| 254 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 48758928 | 70516736 | 1.2209721929919009 | results/candidates/random-search-seed42-20260805/0017-train.py |
| 255 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 23068816 | 133693440 | 1.2215359202343736 | results/candidates/random-search-seed42-20260805/0018-train.py |
| 256 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 34603152 | 92798976 | 1.1100453773605512 | results/candidates/random-search-seed42-20260805/0019-train.py |
| 257 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 56181902 | 82575360 | 1.218617689326707 | results/candidates/random-search-seed42-20260805/0020-train.py |
| 258 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 38404274 | 88604672 | 1.4414257701520798 | results/candidates/random-search-seed42-20260805/0021-train.py |
| 259 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261454 | 144703488 | 1.20355691508419 | results/candidates/random-search-seed42-20260805/0022-train.py |
| 260 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 38880142 | 112197632 | 1.1046358839852746 | results/candidates/random-search-seed42-20260805/0023-train.py |
| 261 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 38880142 | 111673344 | 1.1845142747325912 | results/candidates/random-search-seed42-20260805/0024-train.py |
| 262 | random-search-seed42-20260805 / 42 | random_search / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21496078 | 154140672 | 1.2109539903812885 | results/candidates/random-search-seed42-20260805/0025-train.py |
| 263 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 264 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 82313216 | 1.2953163591115135 | results/candidates/nsgaii-seed42-20260805/0002-train.py |
| 265 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 57016976 | 65536000 | 1.136848199567758 | results/candidates/nsgaii-seed42-20260805/0003-train.py |
| 266 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 79692114 | 50069504 | 1.285759442968087 | results/candidates/nsgaii-seed42-20260805/0004-train.py |
| 267 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 85852980 | 52297728 | 1.1432321816776705 | results/candidates/nsgaii-seed42-20260805/0005-train.py |
| 268 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261070 | 144703488 | 1.1620090908093972 | results/candidates/nsgaii-seed42-20260805/0006-train.py |
| 269 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981772 | 120586240 | 1.2175226364906184 | results/candidates/nsgaii-seed42-20260805/0007-train.py |
| 270 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 70780176 | 63963136 | 1.1412371152843306 | results/candidates/nsgaii-seed42-20260805/0008-train.py |
| 271 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21495950 | 147849216 | 1.1662681193161075 | results/candidates/nsgaii-seed42-20260805/0009-train.py |
| 272 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | oom / valid=0, excluded=1 | UNRESOLVED | UNRESOLVED | UNRESOLVED | UNRESOLVED | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 273 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 65864334 | 77070336 | 1.2300154739655336 | results/candidates/nsgaii-seed42-20260805/0011-train.py |
| 274 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 21495950 | 143917056 | 1.1432421203274201 | results/candidates/nsgaii-seed42-20260805/0012-train.py |
| 275 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 87556096 | 1.1077436616900123 | results/candidates/nsgaii-seed42-20260805/0013-train.py |
| 276 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 56181902 | 80478208 | 1.114167215759927 | results/candidates/nsgaii-seed42-20260805/0014-train.py |
| 277 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31261070 | 144179200 | 1.1591214101342908 | results/candidates/nsgaii-seed42-20260805/0015-train.py |
| 278 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 40501646 | 86769664 | 1.1211046579253596 | results/candidates/nsgaii-seed42-20260805/0016-train.py |
| 279 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 56181902 | 80609280 | 1.1152985466643839 | results/candidates/nsgaii-seed42-20260805/0017-train.py |
| 280 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 68649780 | 55312384 | 1.163953921361752 | results/candidates/nsgaii-seed42-20260805/0018-train.py |
| 281 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 40501646 | 86900736 | 1.1210601398031539 | results/candidates/nsgaii-seed42-20260805/0019-train.py |
| 282 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 31981772 | 117178368 | 1.1625953689093864 | results/candidates/nsgaii-seed42-20260805/0020-train.py |
| 283 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 48759056 | 68943872 | 1.1475188035821131 | results/candidates/nsgaii-seed42-20260805/0021-train.py |
| 284 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 78643200 | 1.1966604591329113 | results/candidates/nsgaii-seed42-20260805/0022-train.py |
| 285 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 50332176 | 94371840 | 1.2036043133647936 | results/candidates/nsgaii-seed42-20260805/0023-train.py |
| 286 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 75890688 | 1.1199297173014011 | results/candidates/nsgaii-seed42-20260805/0024-train.py |
| 287 | nsgaii-seed42-20260805 / 42 | nsgaii / campaign | completed / valid=1, excluded=0 | UNRESOLVED | 44499086 | 75759616 | 1.1201375056341338 | results/candidates/nsgaii-seed42-20260805/0025-train.py |
| 288 | autopareto-primary-lowest-memory-within-1pct-quality-exp124-seed0-20260810 / 0 | confirmation / confirmation | completed / valid=1, excluded=0 | 124 | 24576298 | 168951808 | 1.0997528025223982 | results/candidates/autopareto-primary-lowest-memory-within-1pct-quality-exp124-seed0-20260810/0001-train.py |
| 289 | autopareto-primary-lowest-memory-within-1pct-quality-exp124-seed1-20260810 / 1 | confirmation / confirmation | completed / valid=1, excluded=0 | 124 | 24576298 | 179568640 | 1.0987113236143706 | results/candidates/autopareto-primary-lowest-memory-within-1pct-quality-exp124-seed1-20260810/0001-train.py |
| 290 | autopareto-primary-lowest-memory-within-1pct-quality-exp124-seed2-20260810 / 2 | confirmation / confirmation | completed / valid=1, excluded=0 | 124 | 24576298 | 179437568 | 1.0947538544543503 | results/candidates/autopareto-primary-lowest-memory-within-1pct-quality-exp124-seed2-20260810/0001-train.py |
| 291 | autopareto-primary-best-quality-exp134-seed0-20260810 / 0 | confirmation / confirmation | completed / valid=1, excluded=0 | 134 | 29360392 | 152174592 | 1.0975360832068706 | results/candidates/autopareto-primary-best-quality-exp134-seed0-20260810/0001-train.py |
| 292 | autopareto-primary-best-quality-exp134-seed1-20260810 / 1 | confirmation / confirmation | completed / valid=1, excluded=0 | 134 | 29360392 | 159252480 | 1.0939395395869094 | results/candidates/autopareto-primary-best-quality-exp134-seed1-20260810/0001-train.py |
| 293 | autopareto-primary-best-quality-exp134-seed2-20260810 / 2 | confirmation / confirmation | completed / valid=1, excluded=0 | 134 | 29360392 | 159252480 | 1.0927597550359875 | results/candidates/autopareto-primary-best-quality-exp134-seed2-20260810/0001-train.py |
| 294 | autopareto-primary-largest-exclusive-hypervolume-contribution-exp136-seed0-20260810 / 0 | confirmation / confirmation | completed / valid=1, excluded=0 | 136 | 17891526 | 209715200 | 1.1253738742802166 | results/candidates/autopareto-primary-largest-exclusive-hypervolume-contribution-exp136-seed0-20260810/0001-train.py |
| 295 | autopareto-primary-largest-exclusive-hypervolume-contribution-exp136-seed1-20260810 / 1 | confirmation / confirmation | completed / valid=1, excluded=0 | 136 | 17891526 | 215744512 | 1.1267202298554773 | results/candidates/autopareto-primary-largest-exclusive-hypervolume-contribution-exp136-seed1-20260810/0001-train.py |
| 296 | autopareto-primary-largest-exclusive-hypervolume-contribution-exp136-seed2-20260810 / 2 | confirmation / confirmation | completed / valid=1, excluded=0 | 136 | 17891526 | 220463104 | 1.1230857787930126 | results/candidates/autopareto-primary-largest-exclusive-hypervolume-contribution-exp136-seed2-20260810/0001-train.py |
| 297 | autopareto-primary-fastest-within-1pct-quality-exp137-seed0-20260810 / 0 | confirmation / confirmation | completed / valid=1, excluded=0 | 137 | 26214662 | 195297280 | 1.098160698456856 | results/candidates/autopareto-primary-fastest-within-1pct-quality-exp137-seed0-20260810/0001-train.py |
| 298 | autopareto-primary-fastest-within-1pct-quality-exp137-seed1-20260810 / 1 | confirmation / confirmation | completed / valid=1, excluded=0 | 137 | 26214662 | 196083712 | 1.0996460401783261 | results/candidates/autopareto-primary-fastest-within-1pct-quality-exp137-seed1-20260810/0001-train.py |
| 299 | autopareto-primary-fastest-within-1pct-quality-exp137-seed2-20260810 / 2 | confirmation / confirmation | completed / valid=1, excluded=0 | 137 | 26214662 | 196345856 | 1.098675096131147 | results/candidates/autopareto-primary-fastest-within-1pct-quality-exp137-seed2-20260810/0001-train.py |
| 300 | downstream-recovery-motpe-exp234-seed42-20260812 / 42 | confirmation / campaign | completed / valid=1, excluded=0 | 234 | 26345772 | 154927104 | 1.0975075398733676 | results/candidates/downstream-recovery-motpe-exp234-seed42-20260812/0001-train.py |
| 301 | phase1-200m-downstream / 3 | autopareto_292 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 29360392 | 201326592 | 1.0765807647533765 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 302 | phase1-200m-downstream / 4 | autopareto_292 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 29360392 | 201326592 | 1.079902066613783 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 303 | phase1-200m-downstream / 5 | autopareto_292 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 29360392 | 201326592 | 1.0766959194391954 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 304 | phase1-200m-downstream / 3 | autopareto_294 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 17891526 | 201326592 | 1.1276068282323484 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 305 | phase1-200m-downstream / 4 | autopareto_294 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 17891526 | 201326592 | 1.1283790152587907 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 306 | phase1-200m-downstream / 5 | autopareto_294 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 17891526 | 201326592 | 1.129124072140931 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 307 | phase1-200m-downstream / 3 | autopareto_299 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 26214662 | 201326592 | 1.097274793362705 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 308 | phase1-200m-downstream / 4 | autopareto_299 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 26214662 | 201326592 | 1.0963662170716404 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 309 | phase1-200m-downstream / 5 | autopareto_299 / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 26214662 | 201326592 | 1.0977769148210852 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 310 | phase1-200m-downstream / 3 | autoresearch_compatible / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 50332176 | 201326592 | 1.0325736749786727 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 311 | phase1-200m-downstream / 4 | autoresearch_compatible / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 50332176 | 201326592 | 1.0332000569813968 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 312 | phase1-200m-downstream / 5 | autoresearch_compatible / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 50332176 | 201326592 | 1.0321554657557812 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 313 | phase1-200m-downstream / 3 | baseline / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 50332176 | 201326592 | 1.0896786388973692 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 314 | phase1-200m-downstream / 4 | baseline / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 50332176 | 201326592 | 1.0973407526640946 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 315 | phase1-200m-downstream / 5 | baseline / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 50332176 | 201326592 | 1.093257123620665 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 316 | phase1-200m-downstream / 3 | motpe_quality / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 26345772 | 201326592 | 1.0790924646844042 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 317 | phase1-200m-downstream / 4 | motpe_quality / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 26345772 | 201326592 | 1.0767176130116278 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 318 | phase1-200m-downstream / 5 | motpe_quality / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 26345772 | 201326592 | 1.0819452756880665 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 319 | phase1-200m-downstream / 3 | tpe_quality / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 39846284 | 201326592 | 1.0537301077206473 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 320 | phase1-200m-downstream / 4 | tpe_quality / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 39846284 | 201326592 | 1.0587457893545689 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |
| 321 | phase1-200m-downstream / 5 | tpe_quality / phase1_downstream | completed / valid=1, excluded=1 | UNRESOLVED | 39846284 | 201326592 | 1.0590995232844744 | results/snapshots/runpod-<pod-id>-shutdown-20260901/extracted/results/phase1/downstream_200m.db |

## Detailed field coverage

For each experiment, generated/historical_experiments.json retains timestamp/order, condition, commit, parent, seed, agent/reasoning, exact campaign program, hypothesis, code_diff, intended change, recorded category, parameters, training tokens/time/speed, validation, deployment memory/throughput/KV, failure and decision, and source constants. Optimizer/MLP/head configuration is preserved in source rather than guessed when it cannot be statically recovered. generated/db_training_runs.json includes confirmation/ablation repetitions. generated/db_agent_actions.json contains available transcript/tool events; transcript coverage is incomplete and not inferred from absent rows. The five predefined HPO algorithms each have one 25-attempt campaign; no Centaur or quantization result exists in the recovered registry.

## Current budget attempts

| Trial | Parent | Edits (provisional) | Depth / width / heads | Tokens / updates / seconds | BPB | Quality / Pareto eligibility | Source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | baseline-calibration-repaired-20260903-run1 | batching | 8 / 512 / 4 | 87818240 / 335 / 300.462172414 | 1.1748071659697361 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/history.jsonl:1 |
| T2 | pair-a-fixed-time-seed3-attempt-01 | batching | 8 / 512 / 4 | 86638592 / 661 / 300.094932144 | 1.1034022374372043 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/history.jsonl:2 |
| T3 | pair-a-fixed-time-seed3-attempt-02 | batching | 8 / 512 / 4 | 82837504 / 1264 / 300.622502846 | 3.215914511088702 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/history.jsonl:3 |
| T4 | pair-a-fixed-time-seed3-attempt-02 | batching, learning-rate/schedule | 8 / 512 / 4 | 82837504 / 1264 / 300.528091315 | 3.2159010895186197 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/history.jsonl:4 |
| N1 | baseline-calibration-repaired-20260903-run1 | capacity/depth/width, batching | 10 / 640 / 5 | 88604672 / 169 / 526.599159589 | 1.163068402045889 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/history.jsonl:1 |
| N2 | pair-a-fixed-tokens-seed3-attempt-01 | capacity/depth/width, batching | 12 / 768 / 6 | 88604672 / 169 / 810.958101714 | 1.1642579472667223 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/history.jsonl:2 |
| N3 | pair-a-fixed-tokens-seed3-attempt-01 | batching | 10 / 640 / 5 | 88604672 / 338 / 529.794828783 | 1.1243123343687433 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/history.jsonl:3 |
| N4 | pair-a-fixed-tokens-seed3-attempt-03 | batching | 10 / 640 / 5 | 88604672 / 676 / 536.275079707 | 1.140167140433094 | True / False | runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/history.jsonl:4 |

### T1

Agent-stated hypothesis: Halving the baseline's effective batch should improve validation BPB by roughly doubling optimizer-update frequency within the fixed 300-second budget, while preserving most training throughput because the physical microbatch and model are unchanged.

Intended edit: Set TOTAL_BATCH_SIZE from 524288 to 262144, reducing accumulation from four to two microbatches per update. Retain the architecture and learning rates, enforce seed 3, and align progress, timing, and counters with Contract T.

Actual constant changes: `{"TOTAL_BATCH_SIZE": {"before": 524288, "after": 262144}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/sources/pair-a-fixed-time-seed3-attempt-01.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### T2

Agent-stated hypothesis: Because halving the effective batch from 524288 to 262144 improved validation BPB by 0.0189, halving it again to 131072 should further improve BPB by roughly doubling optimizer-update frequency within the fixed 300-second budget while preserving most token throughput.

Intended edit: Set TOTAL_BATCH_SIZE from 262144 to 131072, reducing gradient accumulation from two microbatches to one per optimizer update while retaining the architecture, optimizer hyperparameters, and time-normalized schedule.

Actual constant changes: `{"TOTAL_BATCH_SIZE": {"before": 262144, "after": 131072}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/sources/pair-a-fixed-time-seed3-attempt-02.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### T3

Agent-stated hypothesis: Because reducing the effective batch from 262144 to 131072 improved validation BPB from 1.1748 to 1.1034, halving it again should improve sample efficiency through twice as many optimizer updates while retaining most token throughput.

Intended edit: Set TOTAL_BATCH_SIZE to 65536 and DEVICE_BATCH_SIZE to 32, preserving one microbatch per optimizer update and leaving the architecture, optimizer hyperparameters, and time-normalized schedule unchanged.

Actual constant changes: `{"TOTAL_BATCH_SIZE": {"before": 131072, "after": 65536}, "DEVICE_BATCH_SIZE": {"before": 64, "after": 32}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/sources/pair-a-fixed-time-seed3-attempt-03.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### T4

Agent-stated hypothesis: The prior 65,536-token batch failed because retaining the parent learning rates doubled update intensity. Retrying it with square-root batch-size learning-rate scaling should improve validation BPB through more frequent, lower-variance optimizer updates without destabilizing training.

Intended edit: Reduce TOTAL_BATCH_SIZE from 131072 to 65536 and DEVICE_BATCH_SIZE from 64 to 32, retaining one microbatch per update. Scale all parameter-family learning rates by 1/sqrt(2): EMBEDDING_LR to 0.424264, UNEMBEDDING_LR to 0.002828, MATRIX_LR to 0.028284, and SCALAR_LR to 0.353553.

Actual constant changes: `{"TOTAL_BATCH_SIZE": {"before": 131072, "after": 65536}, "EMBEDDING_LR": {"before": 0.6, "after": 0.424264}, "UNEMBEDDING_LR": {"before": 0.004, "after": 0.002828}, "MATRIX_LR": {"before": 0.04, "after": 0.028284}, "SCALAR_LR": {"before": 0.5, "after": 0.353553}, "DEVICE_BATCH_SIZE": {"before": 64, "after": 32}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-time-seed3/sources/pair-a-fixed-time-seed3-attempt-04.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### N1

Agent-stated hypothesis: Scaling the transformer from 8×512 to 10×640 at the same fixed token budget will materially improve validation BPB by increasing model capacity, while preserving the global batch and optimization regime.

Intended edit: Increase DEPTH from 8 to 10, yielding width 640 and 5 attention heads; reduce DEVICE_BATCH_SIZE from 64 to 32 and increase accumulation from 4 to 8 to preserve TOTAL_BATCH_SIZE=524288 and memory safety. Enforce the 88604672-token Contract N schedule for exactly 169 updates using seed 3 and protected contract_progress boundaries.

Actual constant changes: `{"DEPTH": {"before": 8, "after": 10}, "DEVICE_BATCH_SIZE": {"before": 64, "after": 32}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/sources/pair-a-fixed-tokens-seed3-attempt-01.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### N2

Agent-stated hypothesis: Extending the quality-positive scaling trend from 10×640 to 12×768 will further improve validation BPB under the same fixed-token budget.

Intended edit: Increase DEPTH from 10 to 12, producing a 768-wide, 6-head transformer. Reduce DEVICE_BATCH_SIZE from 32 to 16 and increase derived gradient accumulation from 8 to 16, preserving TOTAL_BATCH_SIZE=524288 and the exact 88604672-token schedule.

Actual constant changes: `{"DEPTH": {"before": 10, "after": 12}, "DEVICE_BATCH_SIZE": {"before": 32, "after": 16}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/sources/pair-a-fixed-tokens-seed3-attempt-02.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### N3

Agent-stated hypothesis: Halving the global batch for the quality-leading 10×640 model will improve validation BPB by providing twice as many parameter updates over the same fixed token budget, while leaving inference throughput and memory unchanged.

Intended edit: Reduce TOTAL_BATCH_SIZE from 524288 to 262144 while retaining DEVICE_BATCH_SIZE=32, changing gradient accumulation from 8 to 4 and increasing the exact Contract-N schedule from 169 to 338 optimizer updates.

Actual constant changes: `{"TOTAL_BATCH_SIZE": {"before": 524288, "after": 262144}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/sources/pair-a-fixed-tokens-seed3-attempt-03.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

### N4

Agent-stated hypothesis: Halving the quality-leading 10×640 model’s global batch again will further improve validation BPB by doubling optimizer updates under the fixed-token budget, while preserving architecture, inference throughput, and inference memory.

Intended edit: Reduce TOTAL_BATCH_SIZE from 262144 to 131072 while retaining DEVICE_BATCH_SIZE=32, changing gradient accumulation from 4 to 2 and increasing the exact Contract-N schedule from 338 to 676 optimizer updates.

Actual constant changes: `{"TOTAL_BATCH_SIZE": {"before": 262144, "after": 131072}}`. Source: `runs/pair-a-official-20260906-fixed-warmup-v2/pair-a-fixed-tokens-seed3/sources/pair-a-fixed-tokens-seed3-attempt-04.train.py`. Human classification verification: pending. Decision: `{"decision": "quality_valid_speed_uncertain", "parent_eligible": true, "pareto_eligible": false, "quality_status": "quality_valid_speed_uncertain", "reason": "quality and cache correctness passed; protected throughput is uncertain"}`.

## Equal-token verification and complete downstream results

| Run | Params | Tokens | Steps | Heldout BPB | Decode tok/s | Memory MiB | Checks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| phase1-autopareto_292-seed3 | 29360392 | 201326592 | 1536 | 1.1396157998884338 | 382.02448853360204 | 162.8798828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_292-seed4 | 29360392 | 201326592 | 1536 | 1.148667552852964 | 378.69671479851536 | 162.8798828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_292-seed5 | 29360392 | 201326592 | 1536 | 1.1372280113565931 | 388.3442744088567 | 162.8798828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_294-seed3 | 17891526 | 201326592 | 1536 | 1.1909114546092532 | 500.05585206290755 | 106.9423828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_294-seed4 | 17891526 | 201326592 | 1536 | 1.1916783356808018 | 493.2229622939195 | 106.9423828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_294-seed5 | 17891526 | 201326592 | 1536 | 1.1875800088968536 | 493.19509868152807 | 106.9423828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_299-seed3 | 26214662 | 201326592 | 1536 | 1.1594194075079962 | 482.64356684048425 | 142.3798828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_299-seed4 | 26214662 | 201326592 | 1536 | 1.1588967873234026 | 484.34168672134695 | 142.3798828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autopareto_299-seed5 | 26214662 | 201326592 | 1536 | 1.1596106821487935 | 497.7890310157431 | 142.3798828125 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autoresearch_compatible-seed3 | 50332176 | 201326592 | 1536 | 1.100920623285839 | 193.98999738925096 | 250.8818359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autoresearch_compatible-seed4 | 50332176 | 201326592 | 1536 | 1.100928774575567 | 193.43706523094295 | 250.8818359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-autoresearch_compatible-seed5 | 50332176 | 201326592 | 1536 | 1.0975599489328192 | 191.37262101747785 | 250.8818359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-baseline-seed3 | 50332176 | 201326592 | 384 | 1.1431104635651022 | 199.60937030061908 | 250.8818359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-baseline-seed4 | 50332176 | 201326592 | 384 | 1.1519004871954783 | 195.03555491658713 | 250.8818359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-baseline-seed5 | 50332176 | 201326592 | 384 | 1.144757650043697 | 195.8172392695405 | 250.8818359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-motpe_quality-seed3 | 26345772 | 201326592 | 1536 | 1.1508093608896621 | 276.1409132836941 | 145.568359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-motpe_quality-seed4 | 26345772 | 201326592 | 1536 | 1.1491464505468767 | 265.9637657270107 | 145.568359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-motpe_quality-seed5 | 26345772 | 201326592 | 1536 | 1.1448024587454098 | 267.9145310328147 | 145.568359375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-tpe_quality-seed3 | 39846284 | 201326592 | 1536 | 1.1145083448168578 | 257.88189947705223 | 205.880859375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-tpe_quality-seed4 | 39846284 | 201326592 | 1536 | 1.122378990918381 | 256.0911466888535 | 205.880859375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |
| phase1-tpe_quality-seed5 | 39846284 | 201326592 | 1536 | 1.1213174081582233 | 262.70952891134027 | 205.880859375 | {'tokens': True, 'step_count': True, 'timing_rows': True, 'source_hash': True, 'inference': True, 'checkpoint_pairing': True, 'complete_suite': True, 'cutoff': True} |

| Model | Task / metric | Mean | Sample SD across seeds | Examples / segments | Split |
| --- | --- | --- | --- | --- | --- |
| Baseline | heldout_climbmix_bpb/bits_per_byte | 1.1465895336014258 | 0.004672577113590637 | [2560] | heldout |
| Baseline | lambada_openai/perplexity | 14.34738141155896 | 0.5710084331785448 | [5153] | test |
| Baseline | lambada_openai/exact_match | 0.20473510576363282 | 0.006053356864149894 | [5153] | test |
| Baseline | blimp/macro_accuracy | 0.7137213930348258 | 0.00824040560575822 | [67] | train |
| Baseline | blimp/micro_accuracy | 0.7137213930348258 | 0.00824040560575822 | [67000] | train |
| Baseline | piqa/accuracy | 0.5736307580703663 | 0.004155400521360054 | [1838] | validation |
| Baseline | piqa/length_normalized_accuracy | 0.5768951759158506 | 0.0027384238067721723 | [1838] | validation |
| Baseline | arc_easy/accuracy | 0.3528338945005612 | 0.014774665650581748 | [2376] | test |
| Baseline | arc_easy/length_normalized_accuracy | 0.3333333333333333 | 0.002628366160942099 | [2376] | test |
| AutoResearch | heldout_climbmix_bpb/bits_per_byte | 1.099803115598075 | 0.0019426435923677486 | [2560] | heldout |
| AutoResearch | lambada_openai/perplexity | 14.151781632706747 | 0.9978439403079917 | [5153] | test |
| AutoResearch | lambada_openai/exact_match | 0.20441166957759233 | 0.009295400682118567 | [5153] | test |
| AutoResearch | blimp/macro_accuracy | 0.7455920398009951 | 0.005919680963308624 | [67] | train |
| AutoResearch | blimp/micro_accuracy | 0.7455920398009951 | 0.005919680963308624 | [67000] | train |
| AutoResearch | piqa/accuracy | 0.5866884294523033 | 0.013659651837251008 | [1838] | validation |
| AutoResearch | piqa/length_normalized_accuracy | 0.5725426187885383 | 0.014294991946999304 | [1838] | validation |
| AutoResearch | arc_easy/accuracy | 0.36503928170594835 | 0.013743585746942753 | [2376] | test |
| AutoResearch | arc_easy/length_normalized_accuracy | 0.32435465768799104 | 0.009281551575209376 | [2376] | test |
| TPE | heldout_climbmix_bpb/bits_per_byte | 1.1194015812978206 | 0.004270780014603397 | [2560] | heldout |
| TPE | lambada_openai/perplexity | 13.915338920398085 | 1.4487468727677448 | [5153] | test |
| TPE | lambada_openai/exact_match | 0.21424412963322337 | 0.012587112492281406 | [5153] | test |
| TPE | blimp/macro_accuracy | 0.7329701492537314 | 0.007819912778634367 | [67] | train |
| TPE | blimp/micro_accuracy | 0.7329701492537314 | 0.007819912778634367 | [67000] | train |
| TPE | piqa/accuracy | 0.5763511062749365 | 0.007100751523558901 | [1838] | validation |
| TPE | piqa/length_normalized_accuracy | 0.5607544432354008 | 0.008978522883196597 | [1838] | validation |
| TPE | arc_easy/accuracy | 0.35704264870931535 | 0.01011269432712689 | [2376] | test |
| TPE | arc_easy/length_normalized_accuracy | 0.32463524130190796 | 0.012171476558217716 | [2376] | test |
| MOTPE | heldout_climbmix_bpb/bits_per_byte | 1.1482527567273162 | 0.0031015697356332984 | [2560] | heldout |
| MOTPE | lambada_openai/perplexity | 15.061574683812673 | 1.7271308147770943 | [5153] | test |
| MOTPE | lambada_openai/exact_match | 0.21527912542855293 | 0.022651527183797572 | [5153] | test |
| MOTPE | blimp/macro_accuracy | 0.724726368159204 | 0.005487918510057325 | [67] | train |
| MOTPE | blimp/micro_accuracy | 0.724726368159204 | 0.005487918510057372 | [67000] | train |
| MOTPE | piqa/accuracy | 0.5745375408052231 | 0.00445340194334738 | [1838] | validation |
| MOTPE | piqa/length_normalized_accuracy | 0.5656510700036271 | 0.0032794960681945943 | [1838] | validation |
| MOTPE | arc_easy/accuracy | 0.35409652076318743 | 0.018143262003896706 | [2376] | test |
| MOTPE | arc_easy/length_normalized_accuracy | 0.33964646464646464 | 0.015725203444671213 | [2376] | test |
| AP292 | heldout_climbmix_bpb/bits_per_byte | 1.141837121365997 | 0.006034606800928308 | [2560] | heldout |
| AP292 | lambada_openai/perplexity | 14.208496174615345 | 1.6320275548243781 | [5153] | test |
| AP292 | lambada_openai/exact_match | 0.21495568924251246 | 0.018502472264419343 | [5153] | test |
| AP292 | blimp/macro_accuracy | 0.7201741293532339 | 0.003572665879507454 | [67] | train |
| AP292 | blimp/micro_accuracy | 0.7201741293532339 | 0.003572665879507454 | [67000] | train |
| AP292 | piqa/accuracy | 0.582879941965905 | 0.0025711728108011853 | [1838] | validation |
| AP292 | piqa/length_normalized_accuracy | 0.5790714544795067 | 0.0041554005213599934 | [1838] | validation |
| AP292 | arc_easy/accuracy | 0.3691077441077441 | 0.0036448880630658266 | [2376] | test |
| AP292 | arc_easy/length_normalized_accuracy | 0.3416105499438833 | 0.012906083762185104 | [2376] | test |
| AP299 | heldout_climbmix_bpb/bits_per_byte | 1.1593089589933974 | 0.0003695410819012796 | [2560] | heldout |
| AP299 | lambada_openai/perplexity | 15.604130760739874 | 0.7258481267526238 | [5153] | test |
| AP299 | lambada_openai/exact_match | 0.20764603143799729 | 0.002099098355597129 | [5153] | test |
| AP299 | blimp/macro_accuracy | 0.7047363184079602 | 0.00709269896125488 | [67] | train |
| AP299 | blimp/micro_accuracy | 0.7047363184079602 | 0.00709269896125488 | [67000] | train |
| AP299 | piqa/accuracy | 0.5727239753355097 | 0.007386801543132498 | [1838] | validation |
| AP299 | piqa/length_normalized_accuracy | 0.5632934348929997 | 0.013616241816972062 | [1838] | validation |
| AP299 | arc_easy/accuracy | 0.35676206509539843 | 0.015551522402414156 | [2376] | test |
| AP299 | arc_easy/length_normalized_accuracy | 0.3256172839506173 | 0.01938465258071366 | [2376] | test |
| AP294 | heldout_climbmix_bpb/bits_per_byte | 1.1900565997289696 | 0.0021787962862015245 | [2560] | heldout |
| AP294 | lambada_openai/perplexity | 17.70683616514941 | 0.47692886020276726 | [5153] | test |
| AP294 | lambada_openai/exact_match | 0.19917200336373633 | 0.00608850901845663 | [5153] | test |
| AP294 | blimp/macro_accuracy | 0.6933432835820896 | 0.010543136102144612 | [67] | train |
| AP294 | blimp/micro_accuracy | 0.6933432835820896 | 0.010543136102144612 | [67000] | train |
| AP294 | piqa/accuracy | 0.5690968443960827 | 0.0023715445829927584 | [1838] | validation |
| AP294 | piqa/length_normalized_accuracy | 0.5627493652520856 | 0.0029965019299707176 | [1838] | validation |
| AP294 | arc_easy/accuracy | 0.34834455667789 | 0.008831700749459664 | [2376] | test |
| AP294 | arc_easy/length_normalized_accuracy | 0.33052749719416386 | 0.009049637162104463 | [2376] | test |
