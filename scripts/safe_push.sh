#!/usr/bin/env bash
# 공개 저장소 푸시의 표준 경로. 게이트를 모두 통과해야 origin/main으로 푸시한다.
# 사용: bash scripts/safe_push.sh [--check]
#   --check 는 게이트만 수행하고 푸시하지 않는다.
set -euo pipefail

expected_repo="woon-kil/learned-grasp-poc"
cd "$(git rev-parse --show-toplevel)"

branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "${branch}" != "main" ]; then
  echo "거절: 푸시 대상은 main뿐이다 (현재 ${branch})." >&2
  exit 1
fi

url="$(git remote get-url origin)"
case "${url}" in
  "https://github.com/${expected_repo}"|"https://github.com/${expected_repo}.git"|"git@github.com:${expected_repo}.git") ;;
  *)
    echo "거절: origin이 지정 저장소가 아니다 (${url})." >&2
    exit 1
    ;;
esac

if [ -n "$(git status --porcelain)" ]; then
  echo "거절: 커밋하지 않은 변경이 있다." >&2
  exit 1
fi

if git ls-remote --exit-code origin refs/heads/main >/dev/null 2>&1; then
  git fetch -q origin main
  if ! git merge-base --is-ancestor FETCH_HEAD HEAD; then
    echo "거절: 원격 main이 로컬의 조상이 아니다. 이력 재작성 금지." >&2
    exit 1
  fi
fi

bash scripts/public_check.sh
uv run pytest -q

if [ "${1:-}" = "--check" ]; then
  echo "게이트 통과. 푸시하지 않았다."
  exit 0
fi

git push origin main
