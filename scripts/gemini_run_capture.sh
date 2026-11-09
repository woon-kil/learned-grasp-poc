#!/usr/bin/env bash
# learned-grasp-camera 컨테이너 안에서 저장한다. 로봇 명령은 없다.
# 사용: bash scripts/gemini_run_capture.sh <출력디렉터리>
set -euo pipefail

OUT="${1:-/tmp/lg-capture}"
mkdir -p "${OUT}"
set +u
source /opt/ros/humble/setup.bash
if [ -f /opt/cad-less-install/setup.bash ]; then
  source /opt/cad-less-install/setup.bash
fi
set -u

LOG="${OUT}/driver.log"
: > "${LOG}"
stdbuf -oL -eL ros2 launch orbbec_camera gemini_330_series.launch.py \
  depth_registration:=true \
  align_mode:=SW \
  align_target_stream:=COLOR \
  enable_colored_point_cloud:=true \
  enable_publish_extrinsic:=true \
  enable_point_cloud:=true \
  enable_frame_sync:=true \
  > "${LOG}" 2>&1 &
echo $! > /tmp/lg_gemini.pid

cleanup() {
  if [ -f /tmp/lg_gemini.pid ]; then
    kill "$(cat /tmp/lg_gemini.pid)" >/dev/null 2>&1 || true
    wait "$(cat /tmp/lg_gemini.pid)" >/dev/null 2>&1 || true
  fi
  pkill -f "gemini_330_series.launch.py" >/dev/null 2>&1 || true
  pkill -f orbbec_camera_node >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 90); do
  if grep -q "Serial number:" "${LOG}"; then
    ready=1
    break
  fi
  sleep 1
done
if [ "${ready}" -ne 1 ]; then
  echo "드라이버가 시리얼을 보고하지 않았다." >&2
  exit 4
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${LG_PROBE_SEC:-}" ]; then
  PYTHONPATH="${HERE}:${PYTHONPATH:-}" python3 "${HERE}/gemini_probe.py" "${LOG}" "${OUT}" "${LG_PROBE_SEC}"
else
  python3 "${HERE}/gemini_capture.py" "${LOG}" "${OUT}" "${LG_REPEAT:-1}" "${LG_INTERVAL_SEC:-0}"
fi
