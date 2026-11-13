# An Automated Aerial Digital Twin Pipeline for Reproducible and Scalable 6G Channel Datasets and Network-AI Validation

[English](README.md) | 繁體中文

這是一套以官方 [NVIDIA Aerial Omniverse Digital Twin
(AODT)](https://github.com/NVIDIA/aerial-omniverse-digital-twin) 為 backend
的研究流程。專案會收集 AODT 原生通道產物、驗證固定基站與 UAV 實驗、支援可替換
的 Dataset Adapter，並可在開發者自行架設的 AI Server 上執行隨專案提供的
AdvTabDDPM 參考範例。AI 應用可由使用者自行替換：可以換成其他 classifier、
生成模型或 Network-AI plug-in，並提供與該 AI 對應的 dataset adapter。

v0.2.0 新增 AERPAW Lake Wheeler 的 fail-closed 驗證流程 `BU00–BU15`，並附上
固定版本的 OpenStreetMap smoke 環境與來源 manifest。這份 OSM profile 只供可重現
的流程測試，不會被宣稱為論文中的正式 foliage 場景。

本專案自行撰寫的程式與文件採 [Apache-2.0](LICENSE)。第三方元件不會因此被重新
授權；公開或轉散布前必須閱讀 [LICENSE_SCOPE.md](LICENSE_SCOPE.md) 與
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。目前 bundled AdvTabDDPM snapshot
仍須由其著作權人確認授權，不能只靠根目錄 Apache-2.0 就直接公開。

## 一定要照這個順序建置

00B/00BU 開始前，AODT Worker 與 Client 必須先建好並成功連線。

1. 在 AODT Server 啟動官方 AODT Worker。
2. 在 Client 建置官方 AODT Client，確認 gRPC 可連線。
3. 安裝本 repository 的本地控制流程。
4. 架設自己的 AI Server，安裝 repository 內的 AdvTabDDPM 測試專案。
5. 設定兩組 server 位址後，才執行 00B–13B 或 00BU–13BU。

## IP 到底改哪裡

流程有兩台不同 server；28 支 stage script 都沒有寫死部署 IP。

| 連線 | 唯一設定位置 | 預設服務 port | 使用階段 |
|---|---|---:|---|
| Client -> AODT Worker | `DT_SERVER_ADDRESS=<AODT_WORKER_IP>:50051` | 50051 | 06B/06BU 的 AODT gRPC |
| Client -> AI Server | `REMOTE_HOST=<AI_SERVER_IP>`、`REMOTE_PORT=22` | 22 | 11B–13B、11BU–13BU 的 SSH/SCP |

範例檔中的 `CHANGE_ME_*` 一定要換成實際可連線的 IP 或 DNS 名稱。

### AODT Worker IP

在 Client 的 PowerShell 設定：

```powershell
$env:DT_SERVER_ADDRESS = "<AODT_WORKER_IP>:50051"
```

範例也在 [.env.example](.env.example)。這裡不能填 AI Server IP。

### AI Server IP

先建立不會被 Git 追蹤的本地設定：

```bat
copy scripts\remote\remote_settings.example.bat scripts\remote\remote_settings.bat
```

只改 `scripts\remote\remote_settings.bat`：

```bat
set "REMOTE_USER=ubuntu"
set "REMOTE_HOST=<AI_SERVER_IP_OR_DNS>"
set "REMOTE_PORT=22"
set "REMOTE_ROOT=/opt/aerial-digital-twin-pipeline"
set "REMOTE_PYTHON=/opt/aerial-digital-twin-pipeline/.venv-advtabddpm/bin/python"
set "REMOTE_ADVTABDDPM_RUNS_ROOT=/opt/aerial-digital-twin-pipeline/advtabddpm_runs"
```

不要把 IP 複製進 11B/12B/13B 或 UAV scripts，也不要 commit
`remote_settings.bat` 或密碼。

## 1. 先建立 AODT Worker

AODT Worker 要放在具有 NVIDIA GPU 的 Linux 主機。Clone 官方 repository，依
[官方 Worker README](https://github.com/NVIDIA/aerial-omniverse-digital-twin/blob/main/worker/README.md)
安裝 Docker、NVIDIA Container Toolkit、Docker Compose v2 與 `yq` v4，再啟動：

```bash
git clone https://github.com/NVIDIA/aerial-omniverse-digital-twin.git
cd aerial-omniverse-digital-twin
./worker/up.sh
```

確認 AODT Worker 的 TCP 50051 可由 Client 連到，再往下做。

## 2. 再建立 AODT Client

在 Client 使用同一官方 repository 的 `client/`，依
[官方 Client README](https://github.com/NVIDIA/aerial-omniverse-digital-twin/blob/main/client/README.md)
選擇 Windows、Linux/WSL2 或 macOS 的建置方式。官方 Client 會用 gRPC 對遠端
Worker 執行 simulation。

建置成功後設定：

```powershell
$env:AODT_CLIENT_ROOT = "C:\path\to\aerial-omniverse-digital-twin"
$env:AODT_SCENE_ARCHIVE = "C:\path\to\authorized-scene.zip"
$env:DT_SERVER_ADDRESS = "<AODT_WORKER_IP>:50051"
Test-NetConnection <AODT_WORKER_IP> -Port 50051
```

正式跑本專案前，先成功執行官方 Client example。TCP 測試成功只代表網路可達，
不代表 AODT job 已驗證成功。

## 3. 安裝本地控制流程

Client 端合法取得的場景 ZIP 統一放在
`assets/scenes/<scene_id>/<scene_id>.zip`；實驗定義放在 `configs/`。
AODT Worker 必須另外具有展開後的相同場景，以及 YAML 所引用的 Worker-side
asset JSON。完整資料夾、格式、ZIP 必要內容、GPX 規則與 Worker paths 見
[Config 與場景放置規範](docs/CONFIG_AND_SCENES.zh-TW.md)。
從對應目錄首頁也能直接開啟：
[configs/README.md](configs/README.md) 與 [assets/README.md](assets/README.md)。

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[aerpaw,dev,plots]"
python -m unittest discover -s tests -v
```

本地 unit tests 不會連 AODT Worker，也不會啟動 GPU AdvTabDDPM job。

## 4. 架設自己的 AI Server

[ai_server](ai_server/README.md) 已包含 AdvTabDDPM 測試原始碼、固定 classifier、
diffusion checkpoints、requirements、環境定義與 remote runner。GitHub 不包含
每次實驗產生的 dataset 與 results。

在自己的 GPU Server，把這個 repository clone 到 `REMOTE_ROOT`：

```bash
git clone <THIS_REPOSITORY_URL> /opt/aerial-digital-twin-pipeline
cd /opt/aerial-digital-twin-pipeline
python3.10 -m venv .venv-advtabddpm
source .venv-advtabddpm/bin/activate
python -m pip install --upgrade pip setuptools wheel
# 先依 GPU/CUDA 安裝相容的 CUDA-enabled PyTorch。
python -m pip install -r ai_server/AdvTabDDPM/requirements.txt
python -m pip install --no-deps -r ai_server/AdvTabDDPM/requirements-legacy.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Conda 參考環境在 [ai_server/environment.yml](ai_server/environment.yml)，模型雜湊在
[ai_server/model_manifest.csv](ai_server/model_manifest.csv)。`REMOTE_PYTHON` 必須指向
這個 AI Server 環境中的 Python。
此參考範例的版本名稱、環境、dependency、固定設定、provenance 狀態與檔案雜湊
完整記錄在 [ai_server/VERSION.md](ai_server/VERSION.md) 與
[ai_server/version_manifest.json](ai_server/version_manifest.json)。該版本名稱是本專案
的 integration profile，不代表學姊 upstream AdvTabDDPM 的正式 release number。
本專案實際記錄的 AI Server 是 Ubuntu 24.04.4 LTS、RTX 3080、Python 3.10.20、
PyTorch 2.11.0+cu128 與 PyTorch CUDA 12.8 runtime。

## Stage 規劃

每一支 launcher 都是獨立入口，不會偷偷跨 stage。原本 static 與一般 UAV track
維持不變，AERPAW Lake Wheeler 另以獨立目錄提供：

| Track | 路徑 | 用途 |
|---|---|---|
| Static | `scripts/remote/static/00B...13B` | 固定基站與地面 UE |
| UAV | `scripts/remote/uav/00BU...13BU` | UAV 數量由 profile 定義，並包含地面 UE |
| AERPAW Lake Wheeler | `scripts/remote/uav_aodt/00BU...15BU` | measured evidence、地理座標、AODT EM、物理驗證、adapter 與 frozen consumer |

- 00–05：本地 input/preflight。
- 06–07：透過官方 Client 執行 AODT 並收回原生產物。
- 08：驗證 AODT raw artifacts。
- 09–10：建立 trace；需要跑 AdvTabDDPM 範例時，才建立 legacy example dataset。
- 11：只上傳通過 10B/10BU admission 的 dataset，並在 AI Server 建立隔離的
  AdvTabDDPM project copy。
- 12：在 AI Server 執行 train/eval。
- 13：下載 immutable result archive 與 SHA-256。

完整名稱與參數見 [scripts/remote/README.md](scripts/remote/README.md) 與
[docs/STAGES.md](docs/STAGES.md)。
AERPAW 專用執行契約與限制另見
[docs/BU_AERPAW_RUNBOOK.md](docs/BU_AERPAW_RUNBOOK.md)。

## 可自行替換的 AI plug-in

AdvTabDDPM 是隨附的參考範例，不是強制使用的 AI backend。使用者可以替換成其他
AI 專案並實作對應的 dataset adapter。範例 adapter 的輸出只代表該 plug-in 所需的
interface compatibility，
不等於同名欄位具有相同物理語意。歷史 `ul_sinr` 是
`aodt_em_proxy_ul_sinr_db`：沒有實際 PRB 排程與 UE
activity mask 的 EM full-reuse proxy；它不是 AODT RAN telemetry，不得拿來宣稱
跨 backend 的物理 distribution equivalence。

本次整理沒有修改 propagation parameters、AdvTabDDPM hyperparameters、
normalization、classifier threshold，也沒有加入 AODT 專屬 offset/rescaling。

## 論文

本 repository 的相關論文目前準備中；preprint、DOI 與完整引用資訊確認後，會更新在此處。

- Preprint：待發布
- DOI：待發布
- 引用資訊：[CITATION.cff](CITATION.cff)

## 致謝

本專案感謝 [NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program)
提供 NVIDIA AI Aerial 生態的學習資源、技術支援與軟體取得管道，並感謝 NVIDIA 公開提供
[NVIDIA Aerial Omniverse Digital Twin (AODT)](https://github.com/NVIDIA/aerial-omniverse-digital-twin)。
AODT 在本研究中作為外部 RF simulation backend，並提供官方 Client/Worker
interfaces；本 repository 的 adapters、流程控制與 evaluation 為獨立工作，並不代表
NVIDIA 對本專案背書。

本專案亦感謝 [AdvTabDDPM](https://github.com/0425Ning/AdvTabDDPM) 原始作者
`@0425Ning` 提供早期研究實作；本 repository 使用的 AdvTabDDPM 範例是基於該
舊版修改。這段致謝不取代著作權人的正式許可或 upstream license。

論文可使用的致謝文字見
[docs/ACKNOWLEDGEMENTS.md](docs/ACKNOWLEDGEMENTS.md)。

## Repository 目錄

```text
ai_server/       AdvTabDDPM 測試專案、固定模型、環境與 remote runner
configs/         static、UAV、geospatial 與 planning contracts
scripts/aodt/    AODT 執行、收集、驗證與 post-processing
scripts/adapter/ frozen legacy AI-contract adapter
scripts/remote/  static、一般 UAV 與 AERPAW BU00–BU15 launchers
scenes/          可公開的場景來源契約與 OSM smoke snapshot
artifacts/       小型 deterministic trajectory artifacts
tests/           offline unit/contract tests
results/         本地產物，Git ignore
```

AODT runtime、授權場景幾何、credentials、raw CIR、Colosseum data 與新產生的
實驗 results 均不放入 Git。詳見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
與 [docs/DATA_AND_ARTIFACTS.md](docs/DATA_AND_ARTIFACTS.md)。
