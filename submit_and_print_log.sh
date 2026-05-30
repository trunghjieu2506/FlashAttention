#!/bin/bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
submit_script="${script_dir}/submit_job.sh"

resolve_slurm_path() {
  local template="$1"
  local job_id="$2"
  echo "${template//%j/${job_id}}"
}

extract_directive_value() {
  local directive="$1"
  local file="$2"
  sed -n "s/^#SBATCH[[:space:]]*--${directive}=//p" "$file" | tail -n 1
}

wait_for_job_exit() {
  local job_id="$1"
  while squeue -h -j "${job_id}" >/dev/null 2>&1; do
    if [[ -z "$(squeue -h -j "${job_id}")" ]]; then
      break
    fi
    sleep 1
  done
}

wait_for_file() {
  local path="$1"
  local attempts=0
  while [[ ! -f "${path}" ]]; do
    attempts=$((attempts + 1))
    if (( attempts > 60 )); then
      echo "Timed out waiting for ${path}" >&2
      return 1
    fi
    sleep 1
  done
}

submit_output="$("${submit_script}" "$@")"
printf '%s\n' "${submit_output}"

job_id="$(printf '%s\n' "${submit_output}" | sed -n 's/.*Submitted batch job \([0-9][0-9]*\).*/\1/p' | tail -n 1)"
if [[ -z "${job_id}" ]]; then
  echo "Failed to determine job ID from sbatch output." >&2
  exit 1
fi

output_template="$(extract_directive_value output "${submit_script}")"
error_template="$(extract_directive_value error "${submit_script}")"

stdout_path="${script_dir}/$(resolve_slurm_path "${output_template}" "${job_id}")"
stderr_path="${script_dir}/$(resolve_slurm_path "${error_template}" "${job_id}")"
combined_path="${script_dir}/slurm-${job_id}.log"

echo "Waiting for job ${job_id} to finish..."
wait_for_job_exit "${job_id}"

wait_for_file "${stdout_path}"

{
  echo "===== STDOUT (${stdout_path##*/}) ====="
  cat "${stdout_path}"
  if [[ -f "${stderr_path}" ]]; then
    echo
    echo "===== STDERR (${stderr_path##*/}) ====="
    cat "${stderr_path}"
  fi
} > "${combined_path}"

echo "Combined log written to ${combined_path}"
cat "${combined_path}"
