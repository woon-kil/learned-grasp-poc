#!/usr/bin/env bash
# 공개 저장소에 올리면 안 되는 내용을 검사한다. 추적 파일과 HEAD에서 닿는 이력 전체를 본다.
# 삭제 커밋은 이력의 유출을 지우지 않으므로 이력까지 검사한다.
# 사용: bash scripts/public_check.sh
set -euo pipefail

fail=0
# \b는 macOS regcomp에서 동작하지 않아 경계를 문자 집합으로 쓴다.
secrets='192\.168\.[0-9]+\.[0-9]+|(^|[^0-9.])10\.[0-9]+\.[0-9]+\.[0-9]+|(^|[^0-9.])172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+|/Users/[a-zA-Z]|/home/[a-zA-Z]|BEGIN [A-Z ]*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}'
email='[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}'
allowed_email='noreply|example\.(com|org)|git@github\.com'
excludes=(-- . ':!uv.lock' ':!scripts/public_check.sh' ':!tests/test_public_check.py')

scan_tree() {
  # 인자가 없으면 작업 트리의 추적 파일을 본다. 빈 배열은 macOS bash 3.2의 set -u에 걸려 쓰지 않는다.
  if git grep -nIE "${secrets}" "$@" "${excludes[@]}"; then
    return 1
  fi
  if git grep -nIE "${email}" "$@" "${excludes[@]}" | grep -vE "${allowed_email}"; then
    return 1
  fi
  return 0
}

blocked_paths() {
  # 표준입력의 경로 중 공개 금지 파일을 출력한다. data/, runs/는 .gitkeep만 허용한다.
  grep -iE '\.(urdf|srdf|stl|dae|env|pem|key|pt|pth|onnx|ckpt|safetensors)$|(^|/)\.env(\.|$)|^data/|^runs/|^assets/urdf/' \
    | grep -vE '(^|/)\.env\.example$|^(data|runs)/\.gitkeep$' || true
}

if ! scan_tree; then
  echo "거절: 추적 파일에 내부 주소, 개인 정보 또는 자격증명 형태가 있다." >&2
  fail=1
fi

blocked="$(git ls-files | blocked_paths)"
if [ -n "${blocked}" ]; then
  echo "거절: 공개 금지 파일이 추적 중이다." >&2
  printf '%s\n' "${blocked}" >&2
  fail=1
fi

while IFS= read -r -d '' file; do
  if [ -f "${file}" ] && [ "$(wc -c <"${file}")" -gt 5242880 ]; then
    echo "거절: 5 MB를 넘는 파일 ${file}" >&2
    fail=1
  fi
done < <(git ls-files -z)

if git rev-parse --verify -q HEAD >/dev/null; then
  while read -r commit; do
    if ! scan_tree "${commit}"; then
      echo "거절: 이력 ${commit}에 공개 금지 내용이 있다." >&2
      fail=1
    fi
  done < <(git rev-list HEAD)

  blocked="$(git log HEAD --name-only --format= | sort -u | blocked_paths)"
  if [ -n "${blocked}" ]; then
    echo "거절: 이력에 공개 금지 파일이 있다." >&2
    printf '%s\n' "${blocked}" >&2
    fail=1
  fi

  metadata="$(git log HEAD --format='%an%n%ae%n%cn%n%ce%n%B')"
  if printf '%s\n' "${metadata}" | grep -E "${secrets}"; then
    echo "거절: 커밋 메타데이터에 공개 금지 내용이 있다." >&2
    fail=1
  fi
  if printf '%s\n' "${metadata}" | grep -E "${email}" | grep -vE "${allowed_email}"; then
    echo "거절: 커밋 메타데이터에 개인 메일이 있다." >&2
    fail=1
  fi
fi

exit "${fail}"
