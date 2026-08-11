# Configurations

Each JSON file is a parameter-assignment snapshot extracted from the candidate source file used for the named experiment. It includes the source-file SHA-256 and the recorded discovery metrics. `manifest.json` indexes every exported configuration.

These exports are not independently runnable candidate sources. Some candidates include source-level changes outside the exported top-level assignments. The full candidate sources, autonomous search, and evaluation code are scheduled for release with the paper.

Discovery measurements in these files are bfloat16 cached decode on one NVIDIA A40 with 256 input tokens, 256 generated tokens, batch size 1, and 10 measured repetitions.
