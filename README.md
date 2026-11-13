# An Automated Aerial Digital Twin Pipeline for Reproducible and Scalable 6G Channel Datasets and Network-AI Validation

[繁體中文](README.zh-TW.md)

An AODT-only research workflow built around the official
[NVIDIA Aerial Omniverse Digital Twin](https://github.com/NVIDIA/aerial-omniverse-digital-twin).
It captures backend-native AODT channel artifacts, validates static and UAV
runs, supports pluggable dataset adapters, and can execute the bundled
AdvTabDDPM integration as a reference example on a self-hosted AI Server. The
AI application is user-replaceable: developers may supply another classifier,
generative model, or Network-AI plug-in together with its matching adapter.

Version 0.2.0 adds the fail-closed AERPAW Lake Wheeler validation track from
`BU00` through `BU15`, together with a pinned OpenStreetMap smoke-environment
snapshot and its provenance manifest. The OSM profile is for reproducible
plumbing tests and is not represented as the paper's formal foliage scene.

Project-authored source code and documentation are licensed under
[Apache-2.0](LICENSE). Third-party components are not relicensed; review
[LICENSE_SCOPE.md](LICENSE_SCOPE.md) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) before redistribution. The
bundled AdvTabDDPM snapshot remains a separate public-release blocker until its
copyright holders confirm an applicable license.

## Set up in this order

Do not start stage 00B/00BU until the AODT Worker and Client are working.

1. Start the official AODT Worker on the AODT Server.
2. Build the official AODT Client on the Client machine and verify gRPC access.
3. Install this repository's local orchestration environment.
4. Build the self-hosted AI Server and install the checked-in AdvTabDDPM test
   project.
5. Configure the two endpoints, then run the numbered stages.

## The two server addresses

There are two independent connections. No stage script contains a deployment
IP.

| Connection | Setting to edit | Typical port | Used by |
|---|---|---:|---|
| Client -> AODT Worker | `DT_SERVER_ADDRESS=<AODT_WORKER_IP>:50051` | 50051 | 06B/06BU AODT gRPC execution |
| Client -> AI Server | `REMOTE_HOST=<AI_SERVER_IP>` and `REMOTE_PORT=22` | 22 | 11B-13B and 11BU-13BU SSH/SCP |

The `CHANGE_ME_*` values in example files must be replaced with reachable IP
addresses or DNS names.

### Where to change the AODT Worker IP

Set the environment variable on the Client machine:

```powershell
$env:DT_SERVER_ADDRESS = "<AODT_WORKER_IP>:50051"
```

The same setting is documented in [.env.example](.env.example). Do not put the
AI Server IP here.

### Where to change the AI Server IP

Create the ignored local settings file once:

```bat
copy scripts\remote\remote_settings.example.bat scripts\remote\remote_settings.bat
```

Edit only `scripts\remote\remote_settings.bat`:

```bat
set "REMOTE_USER=ubuntu"
set "REMOTE_HOST=<AI_SERVER_IP_OR_DNS>"
set "REMOTE_PORT=22"
set "REMOTE_ROOT=/opt/aerial-digital-twin-pipeline"
set "REMOTE_PYTHON=/opt/aerial-digital-twin-pipeline/.venv-advtabddpm/bin/python"
set "REMOTE_ADVTABDDPM_RUNS_ROOT=/opt/aerial-digital-twin-pipeline/advtabddpm_runs"
```

`remote_settings.bat` is ignored by Git. Never copy these values into the 28
stage launchers or commit credentials.

## 1. Build the AODT Worker first

On a Linux NVIDIA GPU host, clone the official AODT repository and follow its
[Worker prerequisites and setup](https://github.com/NVIDIA/aerial-omniverse-digital-twin/blob/main/worker/README.md).
The official worker uses Docker Compose and is started from the upstream
checkout with:

```bash
git clone https://github.com/NVIDIA/aerial-omniverse-digital-twin.git
cd aerial-omniverse-digital-twin
./worker/up.sh
```

The upstream setup requires Docker, the NVIDIA Container Toolkit, Docker
Compose v2, and `yq` v4. Confirm the worker host is reachable on TCP 50051
before continuing.

## 2. Build the AODT Client

On the Client machine, use the `client/` directory from the same official
repository. Follow the platform-specific
[AODT Client build instructions](https://github.com/NVIDIA/aerial-omniverse-digital-twin/blob/main/client/README.md).
The upstream client supports remote gRPC operation on Windows, Linux/WSL2, and
macOS.

After the client build succeeds, set:

```powershell
$env:AODT_CLIENT_ROOT = "C:\path\to\aerial-omniverse-digital-twin"
$env:AODT_SCENE_ARCHIVE = "C:\path\to\authorized-scene.zip"
$env:DT_SERVER_ADDRESS = "<AODT_WORKER_IP>:50051"
Test-NetConnection <AODT_WORKER_IP> -Port 50051
```

Run an upstream client example before running this repository. A successful
TCP check alone proves reachability, not a valid AODT job.

## 3. Install the local orchestration project

Place the authorized Client-side scene archive under
`assets/scenes/<scene_id>/<scene_id>.zip`. Keep experiment definitions under
`configs/`; the AODT Worker must separately contain the expanded scene and the
Worker-side asset JSON files referenced by the YAML config. Exact folders,
formats, required ZIP entries, GPX rules, and Worker paths are documented in
[Configuration and scene placement](docs/CONFIG_AND_SCENES.md).
The same guide is directly reachable from the directory entry pages:
[configs/README.md](configs/README.md) and
[assets/README.md](assets/README.md).

Requirements: Python 3.11 or 3.12, NumPy, PyYAML, and an authorized scene.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[aerpaw,dev,plots]"
python -m unittest discover -s tests -v
```

The offline tests do not contact the worker or launch a GPU job.

## 4. Build the self-hosted AI Server

The repository includes the AdvTabDDPM test source, fixed classifier and
diffusion checkpoints, requirements, environment definition, and deployment
runner under [ai_server](ai_server/README.md). Generated datasets/results are
not checked in.

On the developer-managed GPU server, clone this repository at `REMOTE_ROOT`,
then create the environment:

```bash
git clone <THIS_REPOSITORY_URL> /opt/aerial-digital-twin-pipeline
cd /opt/aerial-digital-twin-pipeline
python3.10 -m venv .venv-advtabddpm
source .venv-advtabddpm/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install the CUDA-enabled PyTorch wheel compatible with this host first.
python -m pip install -r ai_server/AdvTabDDPM/requirements.txt
python -m pip install --no-deps -r ai_server/AdvTabDDPM/requirements-legacy.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The reference Conda definition is [ai_server/environment.yml](ai_server/environment.yml),
and fixed checkpoint hashes are in
[ai_server/model_manifest.csv](ai_server/model_manifest.csv). Set
`REMOTE_PYTHON` to this environment's Python executable.
The exact reference-example version profile, dependencies, frozen
configuration, provenance status, and file hashes are recorded in
[ai_server/VERSION.md](ai_server/VERSION.md) and
[ai_server/version_manifest.json](ai_server/version_manifest.json).
The recorded integration host is Ubuntu 24.04.4 LTS with an RTX 3080;
Python 3.10.20, PyTorch 2.11.0+cu128, and the PyTorch CUDA 12.8 runtime were
captured in the archived evaluation environment.

From the Windows Client, verify SSH before stage 11:

```bat
"C:\Program Files\PuTTY\plink.exe" -ssh -P 22 -l <SSH_USER> <AI_SERVER_IP> "test -f /opt/aerial-digital-twin-pipeline/ai_server/AdvTabDDPM/AdvTabDDPM.py"
```

## Stage layout

Each launcher is explicit; no launcher silently advances to the next numbered
stage. The original static and general UAV tracks remain unchanged, while the
AERPAW Lake Wheeler validation track is separate.

| Track | Launchers | Purpose |
|---|---|---|
| Static | `scripts/remote/static/00B...13B` | fixed base-station / ground-UE workflow |
| UAV | `scripts/remote/uav/00BU...13BU` | profile-defined UAV and ground-UE workflow |
| AERPAW Lake Wheeler | `scripts/remote/uav_aodt/00BU...15BU` | measured-evidence, geospatial, AODT EM, physical validation, adapter, and frozen-consumer workflow |

- 00-05: inspect inputs and preflight locally.
- 06-07: execute AODT through the official Client and collect native output.
- 08: validate native artifacts.
- 09-10: derive a trace and, when needed, build the optional legacy
  AdvTabDDPM example dataset.
- 11: upload only a passing 10B/10BU dataset and prepare an isolated copy of
  the checked-in AdvTabDDPM test project on the AI Server.
- 12: run AdvTabDDPM train/eval on the AI Server.
- 13: download the immutable result archive and SHA-256 file.

See [scripts/remote/README.md](scripts/remote/README.md) and
[docs/STAGES.md](docs/STAGES.md) for filenames and arguments.
The AERPAW-specific execution contract and limitations are documented in
[docs/BU_AERPAW_RUNBOOK.md](docs/BU_AERPAW_RUNBOOK.md).

## User-replaceable AI plug-in

AdvTabDDPM is the included reference example, not a required AI backend. A
developer may replace it with another AI project and implement the corresponding
dataset adapter. The example adapter's output is interface-compatible input for
that plug-in, not proof that every same-named field is physically equivalent to a Colosseum field.
Historical
`ul_sinr` is explicitly recorded as `aodt_em_proxy_ul_sinr_db`: an EM-derived
full-reuse proxy without actual PRB scheduling or UE activity masks. It is not
native AODT RAN telemetry and must not be used for a cross-backend physical
distribution claim.

This release does not change propagation parameters, AdvTabDDPM
hyperparameters, normalization, classifier thresholds, or AODT-only
offset/rescaling.

## Paper

The associated paper is currently under preparation. The preprint, DOI, and
complete citation will be added here after they become available.

- Preprint: forthcoming
- DOI: forthcoming
- Citation metadata: [CITATION.cff](CITATION.cff)

## Acknowledgements

We acknowledge the
[NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program) for
providing learning resources, technical support, and access pathways for the
NVIDIA AI Aerial ecosystem. We also acknowledge NVIDIA for making
[NVIDIA Aerial Omniverse Digital Twin (AODT)](https://github.com/NVIDIA/aerial-omniverse-digital-twin)
available. AODT is used as the external RF-simulation backend and provides the
upstream Client/Worker interfaces. This repository's adapters, orchestration,
and evaluation workflow are independent work; NVIDIA has not endorsed this
project.

We also acknowledge the original author of
[AdvTabDDPM](https://github.com/0425Ning/AdvTabDDPM) (`@0425Ning`) for the
earlier research implementation on which the locally modified AdvTabDDPM
example is based. This acknowledgement does not replace the copyright holder's
permission or an explicit upstream license.

Paper-ready wording and attribution notes are provided in
[docs/ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md).

## Repository layout

```text
ai_server/       bundled AdvTabDDPM test project, fixed models, environment, runner
configs/         versioned static, UAV, geospatial, and planning contracts
scripts/aodt/    AODT execution, collection, validation, and post-processing
scripts/adapter/ frozen legacy AI-contract adapter
scripts/remote/  static, general UAV, and AERPAW BU00-BU15 launchers
scenes/          public scene-source contracts and the OSM smoke snapshot
artifacts/       small deterministic trajectory artifacts
tests/           offline unit and contract tests
results/         local generated outputs; ignored by Git
old/             not part of this repository; historical files remain outside it
```

AODT runtime files, authorized scene geometry, credentials, raw CIR tensors,
Colosseum data, and generated experiment results are not included. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/DATA_AND_ARTIFACTS.md](docs/DATA_AND_ARTIFACTS.md).
