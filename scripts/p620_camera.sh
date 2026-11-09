#!/usr/bin/env bash
# p620 공유 서버에서 learned-grasp-poc만의 카메라 컨테이너를 다룬다.
# 다른 프로젝트 컨테이너는 멈추거나 지우지 않는다. USB는 촬영 중에만 붙인다.
# 사용: bash scripts/p620_camera.sh status|idle|acquire|release|capture <출력디렉터리>
# 원본: simple-vision-tech-poc/scripts/p620_camera.sh.
set -euo pipefail

IMAGE="${LG_CAMERA_IMAGE:-cad-less-robotics:humble-p003}"
NAME="learned-grasp-camera"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="/ws/learned-grasp-poc"

if [ "${NAME}" != "learned-grasp-camera" ]; then
  echo "컨테이너 이름이 이 프로젝트용이 아니다." >&2
  exit 2
fi

usb_line() {
  lsusb -d 2bc5:0804 | head -n 1 || true
}

usb_node_from_line() {
  local line="$1"
  local bus dev
  bus="$(printf '%s\n' "${line}" | awk '{print $2}')"
  dev="$(printf '%s\n' "${line}" | awk '{print $4}' | tr -d ':')"
  printf '/dev/bus/usb/%s/%s' "${bus}" "${dev}"
}

other_holder() {
  local node="$1" name devices
  while read -r name; do
    [ -z "${name}" ] && continue
    [ "${name}" = "${NAME}" ] && continue
    devices="$(docker inspect --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "${name}" 2>/dev/null || true)"
    if [[ "${devices}" == *"${node}"* ]]; then
      printf '%s' "${name}"
      return 0
    fi
  done < <(docker ps --format '{{.Names}}')
  return 1
}

report_status() {
  local line node holder
  echo "host $(hostname)"
  echo "gpu $(nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || echo 확인불가)"
  echo "containers"
  docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}'
  line="$(usb_line)"
  if [ -z "${line}" ]; then
    echo "camera ABSENT 2bc5:0804"
    return 0
  fi
  node="$(usb_node_from_line "${line}")"
  echo "camera node ${node}"
  if fuser "${node}" >/dev/null 2>&1; then
    echo "camera BUSY fuser ${node}"
    fuser -v "${node}" || true
  else
    echo "camera fuser clear"
  fi
  if holder="$(other_holder "${node}")"; then
    echo "camera HELD_BY ${holder}"
  else
    echo "camera live node is not attached to another running container"
  fi
  if docker container inspect "${NAME}" >/dev/null 2>&1; then
    echo "ours $(docker inspect --format 'running={{.State.Running}} image={{.Config.Image}} devices={{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' "${NAME}")"
  else
    echo "ours ABSENT"
  fi
}

assert_free() {
  local line node holder
  line="$(usb_line)"
  if [ -z "${line}" ]; then
    echo "호스트에 Gemini 335L(2bc5:0804)이 없다." >&2
    exit 4
  fi
  node="$(usb_node_from_line "${line}")"
  if fuser "${node}" >/dev/null 2>&1; then
    echo "카메라가 사용 중이다. 장치 ${node}" >&2
    fuser -v "${node}" >&2 || true
    exit 3
  fi
  if holder="$(other_holder "${node}")"; then
    echo "카메라 USB가 다른 프로젝트 컨테이너에 붙어 있다: ${holder} ${node}" >&2
    exit 3
  fi
  printf '%s' "${node}"
}

recreate() {
  local node="${1:-}"
  local -a args
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  args=(run -d --name "${NAME}" -v "${ROOT}:${WS}" -v bytesize-robotics-camera-install:/opt/cad-less-install:ro -w "${WS}")
  if [ -n "${node}" ]; then
    args+=(--device "${node}")
  fi
  docker "${args[@]}" "${IMAGE}" sleep infinity >/dev/null
}

stop_our_camera() {
  if ! docker container inspect "${NAME}" >/dev/null 2>&1; then
    return 0
  fi
  docker exec "${NAME}" bash -lc '
    if [ -f /tmp/lg_gemini.pid ]; then
      kill "$(cat /tmp/lg_gemini.pid)" >/dev/null 2>&1 || true
    fi
    pkill -f gemini_330_series.launch.py >/dev/null 2>&1 || true
    pkill -f orbbec_camera_node >/dev/null 2>&1 || true
  ' || true
}

cmd="${1:-status}"
case "${cmd}" in
  status)
    report_status
    ;;
  idle)
    recreate
    echo "idle ${NAME} image ${IMAGE} root ${ROOT}"
    ;;
  acquire)
    node="$(assert_free)"
    echo "acquire ${node}"
    recreate "${node}"
    ;;
  release)
    stop_our_camera
    if docker container inspect "${NAME}" >/dev/null 2>&1; then
      recreate
    fi
    echo "released ${NAME}"
    ;;
  capture)
    out="${2:-}"
    if [ -z "${out}" ]; then
      echo "사용: bash scripts/p620_camera.sh capture <출력디렉터리>" >&2
      exit 2
    fi
    case "${out}" in
      "${ROOT}"/*) ;;
      *)
        echo "촬영 결과는 이 체크아웃 안에 둔다: ${ROOT}" >&2
        exit 2
        ;;
    esac
    node="$(assert_free)"
    trap 'stop_our_camera; recreate; echo "released ${NAME}"' EXIT
    recreate "${node}"
    mkdir -p "${out}"
    docker exec \
      -e LG_HOST="$(hostname)" \
      -e LG_CAMERA_IMAGE="${IMAGE}" \
      -e LG_CONTAINER="${NAME}" \
      -e LG_USB_NODE="${node}" \
      -e LG_PROJECT_ROOT="${ROOT}" \
      -e LG_PROBE_SEC="${LG_PROBE_SEC:-}" \
      -e HOME=/tmp \
      -w "${WS}" \
      "${NAME}" \
      bash scripts/gemini_run_capture.sh "${WS}${out#"${ROOT}"}"
    docker exec -u 0 "${NAME}" chown -R "$(id -u):$(id -g)" "${WS}${out#"${ROOT}"}"
    ;;
  serve)
    out="${2:-}"
    case "${out}" in
      "${ROOT}"/*) ;;
      *)
        echo "사용: bash scripts/p620_camera.sh serve <체크아웃 안의 출력디렉터리>" >&2
        exit 2
        ;;
    esac
    node="$(assert_free)"
    trap 'docker exec -u 0 "${NAME}" chown -R "$(id -u):$(id -g)" "${WS}${out#"${ROOT}"}" >/dev/null 2>&1 || true; stop_our_camera; recreate >/dev/null; echo "released ${NAME}" >&2' EXIT
    recreate "${node}"
    mkdir -p "${out}"
    docker exec -i \
      -e LG_SERVE=1 \
      -e HOME=/tmp \
      -w "${WS}" \
      "${NAME}" \
      bash scripts/gemini_run_capture.sh "${WS}${out#"${ROOT}"}"
    ;;
  *)
    echo "사용: $0 status|idle|acquire|release|capture <출력디렉터리>" >&2
    exit 2
    ;;
esac
