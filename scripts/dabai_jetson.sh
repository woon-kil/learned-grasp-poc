#!/usr/bin/env bash
# Jetson에서 DaBai 드라이버를 띄우고, 정지 구간마다 프로브를 돌려 결과를 가져온다. 로봇 명령 없음.
# 사용: bash scripts/dabai_jetson.sh start [런치 인자...] | probe <이름> | stop
# 기본 인자(enable_ir:=false 등) 뒤에 덧붙인 인자가 뒤따라 이긴다(ROS 런치는 마지막 값을 쓴다).
# 드라이버는 한 번만 띄워 두고(start) 세션 내내 유지한다. 프로브는 12초 촬영한다.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a; . "${ROOT}/.env"; set +a
H="${LG_JETSON_HOST:?LG_JETSON_HOST 없음}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=8 "${H}")
case "${1:-}" in
  start)
    scp -q -o BatchMode=yes "${ROOT}/scripts/dabai_probe.py" "${H}:/tmp/dabai_probe.py"
    EXTRA="$(printf '%q ' "${@:2}")"   # 기본 인자 뒤에 덧붙이는 런치 인자. 예: enable_ir:=true depth_registration:=true
    "${SSH[@]}" 'pgrep -f "[_]_ns:=/dabai" >/dev/null && { echo "이미 드라이버가 떠 있다"; exit 3; }
      source /opt/ros/humble/setup.bash; source ~/dofbot_pro_ws/install/setup.bash
      nohup ros2 launch orbbec_camera dabai_dcw2.launch.py camera_name:=dabai enable_point_cloud:=false enable_colored_point_cloud:=false enable_ir:=false '"${EXTRA}"' > /tmp/dabai_drv.log 2>&1 &
      echo $! > /tmp/dabai_drv.pid; sleep 8; pgrep -f "[_]_ns:=/dabai" >/dev/null && echo "드라이버 시작"'
    ;;
  probe)
    name="${2:?이름 필요}"
    "${SSH[@]}" "source /opt/ros/humble/setup.bash; source ~/dofbot_pro_ws/install/setup.bash; python3 /tmp/dabai_probe.py /tmp/dabai_view/${name} 12 dabai >/dev/null; echo probe_exit=\$?"
    mkdir -p "${ROOT}/data/dabai/view/${name}"
    for f in color.png depth.png probe.json; do scp -q -o BatchMode=yes "${H}:/tmp/dabai_view/${name}/${f}" "${ROOT}/data/dabai/view/${name}/"; done
    echo "저장: data/dabai/view/${name}"
    ;;
  stop)
    "${SSH[@]}" 'kill -INT $(cat /tmp/dabai_drv.pid 2>/dev/null) 2>/dev/null; sleep 2; pkill -INT -f "[_]_ns:=/dabai" 2>/dev/null; sleep 2; pgrep -fa "[_]_ns:=/dabai" || echo "드라이버 없음"'
    ;;
  *) echo "사용: $0 start|probe <이름>|stop" >&2; exit 2 ;;
esac
