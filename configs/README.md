# Configurations

Each JSON file is an exact configuration snapshot extracted from the candidate source file used for the named experiment. It includes the source-file SHA-256 and the recorded discovery metrics. `manifest.json` indexes every exported configuration.

These are configuration exports, not a runnable code release. The full autonomous search and evaluation code is scheduled for release with the paper.

Discovery measurements in these files are bfloat16 cached decode on one NVIDIA A40 with 256 input tokens, 256 generated tokens, batch size 1, and 10 measured repetitions.

