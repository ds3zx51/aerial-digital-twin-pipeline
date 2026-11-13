# AdvTabDDPM reference example version specification

## Identity and provenance

| Field | Value |
|---|---|
| Integration profile | `aodt-advtabddpm-reference-2026.08` |
| Snapshot date | 2026-08-13 |
| Role | User-replaceable AI plug-in reference example |
| Original project | <https://github.com/0425Ning/AdvTabDDPM> |
| Original author credit | GitHub `@0425Ning`; replace with the author's preferred academic name for publication |
| Relationship | Locally modified integration derived from an earlier AdvTabDDPM implementation |
| Upstream base commit | `UNRESOLVED` — must be identified before a reproducible public release |
| Local semantic version | Not claimed; this is a dated integration snapshot, not an upstream AdvTabDDPM release |
| License status | Excluded from the root Apache-2.0 grant pending an explicit upstream license or written authorization |

Do not call this snapshot an official new AdvTabDDPM release. It is the version
profile used by this repository's reference integration.

## AI Server runtime profile used by this project

| Component | Recorded specification | Evidence status |
|---|---|---|
| Deployment host | Developer-managed Ubuntu GPU workstation | Required by the checked-in Bash runner |
| Operating system | Ubuntu 24.04.4 LTS x86-64 | Recorded in the project paper, Table IV |
| CPU / RAM | Intel Core i7-14700, 20 cores / 28 threads; 32 GB RAM | Recorded in the project paper, Table IV |
| GPU | NVIDIA GeForce RTX 3080, 12 GB VRAM | Recorded in the project paper, Table IV and archived run logs |
| NVIDIA driver | 580.159.03 | Recorded in the project paper, Table IV |
| Python | 3.10.20 | Archived AI evaluation environment capture |
| PyTorch | 2.11.0+cu128 | Archived AI evaluation environment capture |
| CUDA runtime used by PyTorch | 12.8 | Archived AI evaluation environment capture and AODT evaluation log |
| System CUDA Toolkit | Not independently captured | Do not infer it from `torch.version.cuda` or the driver |
| Remote orchestration | Bash, `flock`, `tar`, `sha256sum`, SSH/SCP | Required by stages 11-13 |
| CPU/RAM minimum | Not characterized | The recorded workstation is not a universal minimum |

The archived platform string is
`Linux-6.17.0-35-generic-x86_64-with-glibc2.39`. The CUDA value above is the
runtime against which the installed PyTorch wheel was built; it is not a claim
that the host's separately installed CUDA Toolkit has the same version.

The CUDA-enabled PyTorch wheel must match the AI Server driver and GPU. The
repository does not install an NVIDIA driver. Record the actual host OS, GPU,
driver, PyTorch, CUDA runtime, and Python versions in every new result
manifest.

## Python dependencies

The primary pinned dependencies are defined in
`AdvTabDDPM/requirements.txt`. Compatibility packages `libzero==0.0.8` and
`rtdl==0.0.9` are listed in `AdvTabDDPM/requirements-legacy.txt` and are
installed with `--no-deps` so they do not replace the selected PyTorch build.
The optional attack dependency is listed in
`AdvTabDDPM/requirements-adversarial.txt`.

Key pinned versions include:

| Package | Version |
|---|---|
| NumPy | 1.26.4 |
| pandas | 2.2.1 |
| scikit-learn | 1.4.1.post1 |
| category-encoders | 2.3.0 |
| matplotlib | 3.9.0 |
| seaborn | 0.13.2 |
| vit-pytorch | 1.10.1 |
| catboost | 1.2.7 |
| skorch | 1.1.0 |
| nvidia-ml-py | 13.610.43 |

## Frozen reference configuration

These values document the checked-in example; they are not defaults imposed on
replacement AI plug-ins.

| Scope | Setting | Value |
|---|---|---:|
| AdvTabDDPM | random seed | 1 |
| AdvTabDDPM | diffusion timesteps | 1000 |
| AdvTabDDPM | scheduler | cosine |
| AdvTabDDPM | training steps | 3000 |
| AdvTabDDPM | learning rate | 0.0006309317827837794 |
| AdvTabDDPM | training batch size | 512 |
| AdvTabDDPM | generated samples | 6500 |
| AdvTabDDPM | sampling batch size | 1024 |
| AdvTabDDPM | sampling seed | 4 |
| Classifier | model | TransformerNN / `trans_v1` |
| Classifier | classes | 3 |
| Classifier | epochs | 350 |
| Classifier | batch size | 512 |
| Classifier | learning rate | 0.001 |
| Classifier | dropout | 0.25 |

No parameter in this table was changed while writing this version
specification.

## Integrity anchors

Machine-readable values are stored in `version_manifest.json`. Fixed
classifier and diffusion checkpoint hashes are stored in
`model_manifest.csv`. A publication run should copy both files into its config
snapshot and record the repository Git commit.

Environment evidence used for this profile is retained in the project's local
archive as `colosseum_advtabddpm_eval_20260712T102059Z_6e2c765b9f75` and
`aodt_advtabddpm_eval_20260717T171657Z_8b6ce428cfc2`. Those result directories
are evidence references, not files distributed by this source repository.
