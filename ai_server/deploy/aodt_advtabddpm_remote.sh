#!/usr/bin/env bash
# Formal isolated AdvTabDDPM execution for an already validated AODT 10B CSV archive.
# Usage: aodt_advtabddpm_remote.sh {prepare|train|eval} <aodt_dataset_name> [metrics_zip]
set -euo pipefail

ROOT="${AODT_REMOTE_ROOT:?set AODT_REMOTE_ROOT}"
PYTHON="${ADVTABDDPM_PYTHON:?set ADVTABDDPM_PYTHON}"
BASELINE="${ROOT}/ai_server/AdvTabDDPM"
RUNS_ROOT="${AODT_ADVTABDDPM_RUNS_ROOT:-${ROOT}/advtabddpm_runs}"
ACTION="${1:?action required}"
SOURCE_NAME="${2:?AODT dataset name required}"
ARCHIVE="${3:-}"
EXPECTED_UE_COUNT="${AODT_EXPECTED_UE_COUNT:-50}"

[[ "${SOURCE_NAME}" =~ ^aodt_dataset_[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid AODT dataset name" >&2; exit 2; }
[[ "${EXPECTED_UE_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: invalid expected UE count" >&2; exit 3; }
[[ -x "${PYTHON}" && -f "${BASELINE}/AdvTabDDPM.py" ]] || { echo "ERROR: bundled AdvTabDDPM project or configured Python is unavailable" >&2; exit 20; }
SOURCE="${ROOT}/results/${SOURCE_NAME}"
PREPARED="${ROOT}/advtabddpm/aodt/${SOURCE_NAME}/AdvTabDDPM_V2_aodt"
LOCK_FILE="${RUNS_ROOT}/.gpu.lock"

prepare() {
  [[ -n "${ARCHIVE}" && -f "${ARCHIVE}" ]] || { echo "ERROR: metrics zip required for prepare" >&2; exit 21; }
  [[ ! -e "${SOURCE}" && ! -e "${PREPARED}" ]] || { echo "ERROR: AODT source/project already exists; refusing overwrite" >&2; exit 22; }
  local token temp parent project_sha classifier_sha
  token="$(${PYTHON} -c 'import uuid; print(uuid.uuid4().hex)')"
  parent="${ROOT}/advtabddpm/aodt/${SOURCE_NAME}"
  temp="${parent}/.AdvTabDDPM_V2_aodt.${token}.tmp"
  mkdir -p "${SOURCE}" "${parent}" "${RUNS_ROOT}/packages"
  "${PYTHON}" - "${ARCHIVE}" "${SOURCE}/metrics" "${EXPECTED_UE_COUNT}" <<'PY'
import re, sys, zipfile
from pathlib import Path
archive, target = map(Path, sys.argv[1:3])
expected = int(sys.argv[3])
target.mkdir(parents=True, exist_ok=False)
rule = re.compile(r"[^/\\]+_metrics\.csv$")
with zipfile.ZipFile(archive) as zf:
    names = zf.namelist()
    if len(names) != expected or any(not rule.fullmatch(Path(name).name) for name in names):
        raise SystemExit(f"metrics archive must contain exactly {expected} top-level *_metrics.csv files")
    for name in names:
        with zf.open(name) as src, (target / Path(name).name).open("wb") as dst:
            dst.write(src.read())
PY
  [[ "$(find "${SOURCE}/metrics" -maxdepth 1 -type f -name '*_metrics.csv' | wc -l)" -eq "${EXPECTED_UE_COUNT}" ]] || { echo "ERROR: invalid uploaded metric count" >&2; exit 23; }
  cp -a "${BASELINE}" "${temp}"
  classifier_sha="$(sha256sum "${temp}/classifier_train_result/slice_model_train_result/model.1.trans_v1.pt" | awk '{print $1}')"
  find "${temp}/data/colosseum" -maxdepth 1 -type f -name '*_metrics.csv' -delete
  find "${temp}/data/colosseum" -maxdepth 1 -type f -name '*.npy' -delete
  rm -rf "${temp}/data/colosseum/pickle_to_csv" "${temp}/data/colosseum_22" "${temp}/data/pickle"
  cp "${SOURCE}"/metrics/*_metrics.csv "${temp}/data/colosseum/"
  ( cd "${temp}"; "${PYTHON}" colosseum_dataset_csv_to_npy.py; "${PYTHON}" make_colosseum_22.py --src data/colosseum --dst data/colosseum_22 )
  "${PYTHON}" "${ROOT}/ai_server/deploy/materialize_aodt_dataset.py" --project "${temp}" --source-result-name "${SOURCE_NAME}" --expected-ue-count "${EXPECTED_UE_COUNT}" --minimum-formal-serving-coverage 1.0 --output "${temp}/aodt_dataset_replacement_manifest.json"
  project_sha="$(sha256sum "${temp}/classifier_train_result/slice_model_train_result/model.1.trans_v1.pt" | awk '{print $1}')"
  [[ "${classifier_sha}" == "${project_sha}" ]] || { echo "ERROR: bundled classifier changed during preparation" >&2; exit 24; }
  "${PYTHON}" - "${temp}/aodt_dataset_replacement_manifest.json" "${SOURCE_NAME}" "${classifier_sha}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1]); source = sys.argv[2]; classifier = sys.argv[3]
m = json.loads(path.read_text(encoding="utf-8"))
m.update({"source": "aodt_static_cir", "source_09b_run_id": source, "source_09a_result_name": None, "base_application": "AdvTabDDPM_V2_aodt", "replacement_scope": "dataset_only", "application_code_modified": False, "config_modified": False, "classifier_retrained": False, "bundled_classifier_preserved": True, "bundled_classifier_sha256": classifier, "claim_boundary": "AODT-derived channel CSVs plus unchanged deterministic adapter and AdvTabDDPM application."})
path.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
PY
  mv "${temp}" "${PREPARED}"
  "${PYTHON}" - "${SOURCE}" "${SOURCE_NAME}" "${EXPECTED_UE_COUNT}" <<'PY'
import json, sys
from pathlib import Path
root, name, expected = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
manifest = {"status":"aodt_dataset_succeeded", "backend":"aodt", "source_run_id":name, "expected_ue_count":expected, "metrics_checks":{"metrics_csv_count":expected, "metrics_rows":expected*256, "columns":32}}
(root / "run_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
(root / "adapter_report.json").write_text(json.dumps({"status":"metrics_generation_succeeded", "channel_source":"aodt", "metrics_csv_count":expected, "metrics_rows":expected*256}, indent=2)+"\n", encoding="utf-8")
(root / "dataset_provenance.json").write_text(json.dumps({"channel_source":"aodt", "source_run_id":name, "contract":"optional_advtabddpm_example_contract"}, indent=2)+"\n", encoding="utf-8")
PY
  echo "STATUS=formal_aodt_project_ready"
  echo "AODT_PROJECT=${PREPARED}"
}

train() {
  [[ -f "${PREPARED}/aodt_dataset_replacement_manifest.json" ]] || { echo "ERROR: run prepare first" >&2; exit 30; }
  mkdir -p "${RUNS_ROOT}"; exec 9>"${LOCK_FILE}"; flock -n 9 || { echo "ERROR: another GPU job is active" >&2; exit 34; }
  local stamp result run project before after rc
  stamp="$(date -u +%Y%m%dT%H%M%SZ)_$(${PYTHON} -c 'import uuid;print(uuid.uuid4().hex[:12])')"
  result="aodt_advtabddpm_train_${stamp}"; run="${RUNS_ROOT}/${result}"; project="${run}/project"
  mkdir "${run}"; cp -a "${PREPARED}" "${project}"; mkdir "${run}/artifacts"; : >"${run}/stdout.log"; : >"${run}/stderr.log"
  before="$(sha256sum "${project}/classifier_train_result/slice_model_train_result/model.1.trans_v1.pt" | awk '{print $1}')"
  find "${project}/exp/colosseum/ddpm_tune_best" -mindepth 1 -maxdepth 1 ! -name config.toml -exec rm -rf {} +
  set +e
  (cd "${project}" && "${PYTHON}" AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result --exp_name slice_model_train_result --transformer v1 --slicelen 1 --mode test --train) >>"${run}/stdout.log" 2>>"${run}/stderr.log"
  rc=$?; set -e
  after="$(sha256sum "${project}/classifier_train_result/slice_model_train_result/model.1.trans_v1.pt" | awk '{print $1}')"
  [[ "${before}" == "${after}" ]] || rc=40
  "${PYTHON}" - "${run}/summary.json" "${rc}" "${SOURCE_NAME}" "${result}" "${project}" "${before}" <<'PY'
import json, sys
from pathlib import Path
out, code, source, result, project, classifier = sys.argv[1:]
m={"status":"formal_training_succeeded" if int(code)==0 else "formal_training_failed", "exit_code":int(code), "channel_source":"aodt", "source_09b_run_id":source, "result_name":result, "training_project_path":project, "replacement_scope":"dataset_only", "application_code_modified":False, "classifier_retrained":False, "bundled_classifier_preserved":True, "bundled_classifier_sha256":classifier}
Path(out).write_text(json.dumps(m,indent=2)+"\n",encoding="utf-8")
PY
  cp "${run}/summary.json" "${SOURCE}/aodt_training_manifest.json"; cp "${project}/aodt_dataset_replacement_manifest.json" "${run}/artifacts/"
  echo "RESULT_NAME=${result}"; echo "STATUS=$([[ ${rc} -eq 0 ]] && echo formal_training_succeeded || echo formal_training_failed)"; return "${rc}"
}

eval_run() {
  [[ -f "${SOURCE}/aodt_training_manifest.json" ]] || { echo "ERROR: successful training manifest missing" >&2; exit 41; }
  local training_project
  training_project="$(${PYTHON} - "${SOURCE}/aodt_training_manifest.json" <<'PY'
import json, sys
m=json.load(open(sys.argv[1],encoding="utf-8")); assert m["status"]=="formal_training_succeeded"; print(m["training_project_path"])
PY
)"
  mkdir -p "${RUNS_ROOT}"; exec 9>"${LOCK_FILE}"; flock -n 9 || { echo "ERROR: another GPU job is active" >&2; exit 42; }
  local stamp result run project rc clean adv
  stamp="$(date -u +%Y%m%dT%H%M%SZ)_$(${PYTHON} -c 'import uuid;print(uuid.uuid4().hex[:12])')"; result="aodt_advtabddpm_eval_${stamp}"; run="${RUNS_ROOT}/${result}"; project="${run}/project"
  mkdir "${run}"; cp -a "${training_project}" "${project}"; mkdir "${run}/artifacts"; : >"${run}/stdout.log"; : >"${run}/stderr.log"
  set +e
  (cd "${project}" && "${PYTHON}" AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result --exp_name slice_model_train_result --transformer v1 --slicelen 1 --mode test) >>"${run}/stdout.log" 2>>"${run}/stderr.log"; clean=$?
  adv=-1
  if [[ ${clean} -eq 0 ]]; then (cd "${project}" && "${PYTHON}" AdvTabDDPM.py --config exp/colosseum/ddpm_tune_best/config.toml --ds_path data/colosseum/pickle_to_csv --ds_file colosseum_dataset_globalnorm.pkl --norm_param_file colosseum_global_cols_maxmin.pkl --isNorm --log_base_dir ./classifier_train_result --exp_name slice_model_train_result --transformer v1 --slicelen 1 --mode test --eval) >>"${run}/stdout.log" 2>>"${run}/stderr.log"; adv=$?; fi
  rc=${clean}; [[ ${rc} -eq 0 ]] && rc=${adv}; set -e
  if [[ ${rc} -eq 0 ]]; then (cd "${project}" && "${PYTHON}" compare_attack_results.py --clean classifier_train_result/slice_model_train_result/eval_confidence_details.csv --advtabddpm exp/colosseum/ddpm_tune_best/eval_confidence_details.csv --output-dir "${run}/artifacts/attack_comparison") >>"${run}/stdout.log" 2>>"${run}/stderr.log" || rc=$?; fi
  cp "${project}/classifier_train_result/slice_model_train_result/eval_confidence_details.csv" "${run}/artifacts/clean_eval_confidence_details.csv" 2>/dev/null || true
  cp "${project}/exp/colosseum/ddpm_tune_best/eval_confidence_details.csv" "${run}/artifacts/advtabddpm_eval_confidence_details.csv" 2>/dev/null || true
  "${PYTHON}" - "${run}/summary.json" "${rc}" "${SOURCE_NAME}" "${result}" "${clean}" "${adv}" <<'PY'
import json,sys
from pathlib import Path
out,code,source,result,clean,adv=sys.argv[1:]
Path(out).write_text(json.dumps({"status":"formal_evaluation_succeeded" if int(code)==0 else "formal_evaluation_failed","exit_code":int(code),"channel_source":"aodt","source_09b_run_id":source,"result_name":result,"clean_evaluation_exit_code":int(clean),"advtabddpm_evaluation_exit_code":int(adv),"replacement_scope":"dataset_only","application_code_modified":False,"classifier_retrained":False,"bundled_classifier_preserved":True},indent=2)+"\n",encoding="utf-8")
PY
  tar -czf "${RUNS_ROOT}/packages/${result}.tar.gz" -C "${RUNS_ROOT}" "${result}"; sha256sum "${RUNS_ROOT}/packages/${result}.tar.gz" >"${RUNS_ROOT}/packages/${result}.tar.gz.sha256"
  echo "RESULT_NAME=${result}"; echo "RESULT_ARCHIVE=${RUNS_ROOT}/packages/${result}.tar.gz"; echo "STATUS=$([[ ${rc} -eq 0 ]] && echo formal_evaluation_succeeded || echo formal_evaluation_failed)"; return "${rc}"
}

case "${ACTION}" in prepare) prepare ;; train) train ;; eval) eval_run ;; *) echo "ERROR: action must be prepare, train, or eval" >&2; exit 64 ;; esac
