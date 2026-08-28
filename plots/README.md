# Figure inventory

The four current publication figures are generated from the public aggregate Phase 1
summaries and describe the equal-token verification campaign at exactly 201,326,592
training tokens on an NVIDIA A40. They use means ± sample standard deviation across
seeds 3–5 unless a caption says otherwise:

- `main_pareto_frontier`: held-out BPB versus cached decode throughput.
- `ap292_vs_autoresearch`: separate-unit quality, efficiency, and downstream panels.
- `downstream_comparison`: downstream accuracy, LAMBADA perplexity, and held-out BPB.
- `discovery_to_verification`: historical five-minute selected representatives versus
  equal-token verification means; contextual, not a paired causal estimate.

The existing `discovery_frontier`, `ablation_validation_bpb`, and
`deployment_workload` figures are retained from the release branch as historical
supporting evidence. They are not embedded in the current README headline story and
should not be read as equal-token verification results.
