# thesis_remote_4090

This repository is the code-only sync boundary for the 4090 workspace.

What belongs here:

- experiment source code
- configs
- launch scripts
- lightweight analysis and plotting code
- task inventory JSON and markdown notes

What does not belong here:

- datasets
- model caches such as `hf_cache/`
- run outputs
- checkpoints
- generated figures and archives

Practical rule:

- sync `remote_4090/` with the remote machine using git
- keep model/data/output artifacts local to each machine and ignored

Cluster layout for the A800 environment:

- put the git checkout under the user workspace, for example `/data/chenwei/thesis_remote_4090`
- put datasets, caches, checkpoints, and outputs under the shared workspace, for example `/nfsdata/ganbei/chenwei_thesis`
- use the A800-specific launch scripts under `experiments/` when submitting long jobs on the cluster
