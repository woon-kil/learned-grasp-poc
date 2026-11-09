#!/usr/bin/env bash
# Jetson 서버 파일을 복사하고 문법만 확인한다. 실행하지 않고 장치도 열지 않는다.
# 사용: bash scripts/deploy_jetson.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ ! -f .env ]; then
  echo ".env가 없다. .env.example을 참고한다." >&2
  exit 2
fi
host="$(grep -E '^LG_JETSON_HOST=' .env | cut -d= -f2-)"
if [ -z "${host}" ]; then
  echo "LG_JETSON_HOST가 비어 있다." >&2
  exit 2
fi
ssh -o BatchMode=yes "${host}" 'mkdir -p learned_grasp'
scp -q src/learned_grasp/jetson_arm_server.py "${host}:learned_grasp/jetson_arm_server.py"
ssh -o BatchMode=yes "${host}" 'python3 -m py_compile learned_grasp/jetson_arm_server.py && sha256sum learned_grasp/jetson_arm_server.py'
shasum -a 256 src/learned_grasp/jetson_arm_server.py
