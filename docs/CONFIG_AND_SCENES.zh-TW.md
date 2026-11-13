# Config 與場景放置規範

本文件說明每類實驗設定與場景輸入應放在哪裡、使用什麼格式，以及是否能放入 Git。
以下 `configs/` 路徑皆以 repository 根目錄為基準。

## 快速對照表

| 輸入 | 放置位置 | 格式 | 放入 Git？ |
|---|---|---|---|
| Static 實驗 scenario | `configs/scenario_layout_tokyo_docomo.json` | UTF-8 JSON | 是 |
| UAV 實驗 scenario | `configs/scenarios/bu/<scenario_id>.json` | UTF-8 JSON | 是 |
| Worker asset references | `configs/aodt/<profile>_assets.yml` | UTF-8 YAML mapping | 是 |
| 場景地理座標 contract | `configs/geospatial/<scene_id>_aodt_scene_contract.json` | UTF-8 JSON | 是 |
| UAV design 與 canonical ENU trace | `configs/trajectories/shared/<trajectory_id>_design.json`、`<trajectory_id>_enu.json` | UTF-8 JSON | 是 |
| UAV GPX 與 trajectory manifest | `configs/gpx/shared/<trajectory_id>/` | GPX 1.1 XML 與 UTF-8 JSON | 是 |
| 合法取得的 AODT 場景壓縮檔 | `assets/scenes/<scene_id>/<scene_id>.zip` | 包含 OpenUSD 與 metadata 的 ZIP | 否 |
| Worker 解壓後場景 | AODT Worker object-store prefix，例如 `test_data/maps/<scene_id>/` | 展開後的 OpenUSD tree | 否 |
| Worker AODT asset definitions | Asset YAML 指定的 `test_data/assets/*.json` | AODT Worker JSON assets | 否 |
| 新產生的 results | `results/aodt_static/` 或 `results/aodt_uav/` | Generated artifacts | 否 |

Credentials、vendor scene ZIP、raw CIR 與 results 不可放進 `configs/`。

## 1. Static scenario config

目前 static reference 固定放在：

```text
configs/scenario_layout_tokyo_docomo.json
```

它必須是 UTF-8 JSON，top-level keys 正好為：

```json
{
  "meta": {},
  "bs": [],
  "ue": [],
  "experiment_definition": {}
}
```

目前 publication profile 有自己的 locked node counts、排序後 identity、radio
definition 與 topology validator。Adapter 使用
`AODT node ID = scenario node ID + 1`。JSON 中的座標是 OpenUSD scene-stage
units，必須配合該場景記錄的 unit conversion，不能直接假設原始值就是公尺。

若要檢查其他 JSON，不要覆蓋 reference。將新檔案放在：

```text
configs/scenarios/static/<scenario_id>.json
```

並明確傳入：

```bat
scripts\remote\static\00B_inspect_inputs.bat ^
  --scenario configs\scenarios\static\<scenario_id>.json ^
  --scene-archive assets\scenes\<scene_id>\<scene_id>.zip
```

換 topology 或場景等於建立新 profile，必須連同 validators、geospatial contract、
trajectory bindings 與 provenance 一起更新。只傳入 `--scenario` 不代表新 profile
已能正確執行到 stage 13。

## 2. UAV scenario 與 trajectory config

每個 UAV scenario 放在：

```text
configs/scenarios/bu/<scenario_id>.json
```

Repository 內的 example 是 `configs/scenarios/bu/tokyo_BU_multi_uav.json`，並以
repository-relative path 連結：

```text
configs/trajectories/shared/<trajectory_id>_design.json
configs/trajectories/shared/<trajectory_id>_enu.json
configs/gpx/shared/<trajectory_id>/trajectory_manifest.json
configs/gpx/shared/<trajectory_id>/uav_<id>_<trajectory_name>.gpx
```

UAV 數量由各 profile 自己的 scenario 與 trajectory manifest 決定，不是整條
pipeline 的固定常數。`aerial_ues` 數量、GPX records、canonical ENU routes、
manifest records、Worker GPX objects 與 runtime node mappings 必須一致。
Profile-specific validator 可能會鎖定該 profile 登記的數量；更換數量應建立並驗證
新 profile，不能只在原 profile 增減一支 GPX。

每支 GPX 必須是 GPX 1.1 XML。每個 `trkpt` 都要有 latitude、longitude、
`<ele>` 與 ISO-8601 UTC `<time>`。Sample count 與 interval 由 profile manifest
宣告，所有 routes 必須一致。Manifest 會以 SHA-256 綁定 design JSON、canonical
ENU JSON、GPX、source static scenario、geodetic contract 與衍生 artifacts。

不可只改一支 GPX 或手動改單一 hash。應先在 staging 重新產生整組 payload，
驗證、檢查 diff 後再整組發布：

```powershell
python tools\generate_shared_multi_uav_gpx.py --project-root . --staging-root "C:\path\to\empty-staging-dir"
python tools\validate_shared_multi_uav_gpx.py --artifact-root "C:\path\to\empty-staging-dir" --reference-project-root .
python tools\validate_bu_scenario.py configs\scenarios\bu\<scenario_id>.json
```

Bundled generator 是 example-profile generator。若更換 UAV 數量或 trajectory
design，必須把 generator 與 validator 一起納入新 profile。Generator 不會啟動
AODT 或 AI job。

## 3. Client 本機場景 ZIP

將合法取得的場景壓縮檔放在 Git ignore 的本機資料夾：

```text
assets/
  scenes/
    <scene_id>/
      <scene_id>.zip
```

在 Client PowerShell 設定絕對路徑：

```powershell
$env:AODT_SCENE_ARCHIVE = "$PWD\assets\scenes\<scene_id>\<scene_id>.zip"
```

目前 Tokyo profile 的 ZIP 至少必須具有以下精確路徑，外層不可再多包一層：

```text
tokyo/sim/master.usd
tokyo/sim/master_tiles/tile_0_0.usd
tokyo/sim/master_metadata.json
tokyo/sim/master_input_manifest.json
tokyo/sim/vegetation.geojson
```

ZIP 也必須包含 OpenUSD composition 所需的所有 layers、textures 與 referenced
dependencies。JSON 與 GeoJSON 必須是有效 UTF-8。Pipeline 會記錄 ZIP hash 與
inventory，但場景再散布權和 source-code license 不同，因此場景 ZIP 不加入 Git。

## 4. AODT Worker 上的場景與 assets

本機 ZIP 只是 Client 端驗證來源，本 repository 不會自動上傳它。Online run 前
必須使用獲授權的 upstream AODT asset workflow，將相同場景展開並放到 Worker。
目前 reference Worker prefix 是 `test_data/maps/tokyo`，因此 Worker 必須能解析：

```text
test_data/maps/tokyo/sim/master.usd
test_data/maps/tokyo/sim/vegetation.geojson
```

`configs/aodt/aodt_static_assets.yml` 是以下 Worker-side categories 的 YAML mapping：

```yaml
du: test_data/assets/du.json
ru: test_data/assets/gnb.json
ue: test_data/assets/ue.json
materials: test_data/assets/materials.json
scatterers: test_data/assets/car_small.json
vegetation_materials: test_data/assets/vegetation_materials.json
vegetation_assets: test_data/assets/street_tree.json
scenario: test_data/assets/scenario.json
panels: test_data/assets/panel.json
```

這些都是 Worker paths，不是 Windows Client 或 repository-relative paths。只有在
Worker 上逐一確認後，才能使用 `--ack-worker-assets-resolved`。

UAV run 必須把 selected manifest 列出的每一支 GPX 都放到
`--worker-gpx-prefix` 指定的 Worker prefix。只有在 Worker copies 與 committed
manifest hashes 相符後，才能使用 `--ack-worker-gpx-resolved`。

## 5. Client 環境變數與執行方式

Runner 會直接讀取 Client process 的環境變數：

```powershell
$env:AODT_CLIENT_ROOT = "C:\path\to\aerial-omniverse-digital-twin"
$env:AODT_SCENE_ARCHIVE = "$PWD\assets\scenes\<scene_id>\<scene_id>.zip"
$env:DT_SERVER_ADDRESS = "<AODT_WORKER_IP>:50051"
```

`.env.example` 只是範本；目前 batch scripts 不會自動讀取 `.env`。請在 shell、
作業系統環境變數或不加入 Git 的本機 wrapper 中設定。

Static 範例：

```bat
scripts\remote\static\05B_run_preflight.bat --ack-worker-assets-resolved
scripts\remote\static\06B_run_aodt.bat --ack-worker-assets-resolved
```

UAV 範例：

```bat
scripts\remote\uav\05BU_run_preflight.bat --ack-worker-assets-resolved --ack-worker-gpx-resolved
scripts\remote\uav\06BU_run_aodt.bat --ack-worker-assets-resolved --ack-worker-gpx-resolved
```

Acknowledgement flags 只是操作者的確認聲明，不是 upload commands，也不能取代
實際檢查 Worker 檔案。
