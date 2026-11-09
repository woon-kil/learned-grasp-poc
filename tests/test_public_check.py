from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from learned_grasp.config import ROOT

SCRIPT = Path(ROOT) / "scripts" / "public_check.sh"
HOOK = Path(ROOT) / ".githooks" / "pre-push"


def repo(tmp_path: Path, name: str, text: str) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "scripts").mkdir()
    shutil.copy(SCRIPT, tmp_path / "scripts" / "public_check.sh")
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def check(path: Path) -> int:
    return subprocess.run(["bash", "scripts/public_check.sh"], cwd=path, capture_output=True).returncode


def test_change_fraction_detects_removed_object():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gemini_serve_core", Path(ROOT) / "scripts" / "gemini_serve_core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    before = np.full((100, 100, 3), 200, np.uint8)
    after = before.copy()
    before[40:60, 40:60] = (230, 230, 40)
    roi = (0, 100, 0, 100)
    assert core.change_fraction(before, after, roi) == pytest.approx(0.04)
    assert core.change_fraction(before, before, roi) == 0.0
    assert core.change_fraction(before, after, (0, 10, 0, 10)) == 0.0


def test_clean_repo_passes(tmp_path):
    assert check(repo(tmp_path, "note.md", "버전 10.3.3, 포트 없음\n")) == 0


def _commit(path: Path, message: str = "m") -> None:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", message], cwd=path, check=True)


def test_deleted_file_in_history_fails(tmp_path):
    """삭제 커밋은 이력의 유출을 지우지 않는다."""
    path = repo(tmp_path, "assets/urdf/robot.urdf", "<robot/>\n")
    _commit(path)
    (path / "assets" / "urdf" / "robot.urdf").unlink()
    (path / "note.md").write_text("ok\n")
    _commit(path)
    assert check(path) == 1


def test_secret_removed_from_tree_but_in_history_fails(tmp_path):
    path = repo(tmp_path, "note.md", "접속 192.168.0.94\n")
    _commit(path)
    (path / "note.md").write_text("ok\n")
    _commit(path)
    assert check(path) == 1


def test_private_mail_in_commit_message_fails(tmp_path):
    path = repo(tmp_path, "note.md", "ok\n")
    _commit(path, "문의 someone@corp-mail.com")
    assert check(path) == 1


def test_clean_history_passes(tmp_path):
    path = repo(tmp_path, "note.md", "ok\n")
    _commit(path)
    assert check(path) == 0


def test_env_example_is_allowed_but_env_variants_are_not(tmp_path):
    assert check(repo(tmp_path, ".env.example", "X=<value>\n")) == 0
    (tmp_path / ".env.local").write_text("X=1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    assert check(tmp_path) == 1


@pytest.mark.parametrize(
    "name, text",
    [
        ("a.md", "접속 jetson@192.168.0.94\n"),
        ("a.md", "경로 /Users/someone/x\n"),
        ("a.md", "메일 someone@gmail.com\n"),
        ("a.md", "ghp_" + "a" * 30 + "\n"),
        ("assets/urdf/robot.urdf", "<robot/>\n"),
        (".env", "X=1\n"),
        ("data/sample.txt", "x\n"),
        ("data/.private.csv", "x\n"),
        ("runs/.private/log.json", "x\n"),
        ("results/model.pt", "x\n"),
        ("model.URDF", "x\n"),
        ("a.md", "내부 172.16.1.2\n"),
        ("a.md", "내부 10.1.2.3\n"),
        ("a.md", "github_pat_" + "a" * 30 + "\n"),
        ("a.md", "메일 someone@corp-mail.com\n"),
    ],
)
def test_blocked_content_fails(tmp_path, name, text):
    assert check(repo(tmp_path, name, text)) == 1


def test_project_is_clean():
    assert subprocess.run(["bash", str(SCRIPT)], cwd=ROOT, capture_output=True).returncode == 0


def run_hook(tmp_path: Path, stdin: str) -> subprocess.CompletedProcess:
    """훅 앞부분의 대상 검사만 본다. 통과하면 이후 단계에서 멈추도록 스크립트와 테스트가 없는 저장소를 쓴다."""
    return subprocess.run(["bash", str(HOOK)], cwd=tmp_path, input=stdin, capture_output=True, text=True)


def _commits(tmp_path: Path) -> tuple[str, str]:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    env = ["-c", "user.name=t", "-c", "user.email=t@example.com"]
    shas = []
    for text in ("a", "b"):
        (tmp_path / "f").write_text(text)
        subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
        subprocess.run(["git", *env, "commit", "-qm", text], cwd=tmp_path, check=True)
        shas.append(subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True).stdout.strip())
    return shas[0], shas[1]


ZERO = "0" * 40


def test_hook_rejects_other_branch(tmp_path):
    old, new = _commits(tmp_path)
    result = run_hook(tmp_path, f"refs/heads/x {new} refs/heads/feature {old}\n")
    assert result.returncode == 1 and "main뿐" in result.stderr


def test_hook_rejects_delete(tmp_path):
    old, _ = _commits(tmp_path)
    result = run_hook(tmp_path, f"(delete) {ZERO} refs/heads/main {old}\n")
    assert result.returncode == 1 and "삭제" in result.stderr


def test_hook_rejects_history_rewrite(tmp_path):
    old, new = _commits(tmp_path)
    subprocess.run(["git", "reset", "-q", "--hard", old], cwd=tmp_path, check=True)
    result = run_hook(tmp_path, f"refs/heads/main {old} refs/heads/main {new}\n")
    assert result.returncode == 1 and "force" in result.stderr


def test_hook_rejects_ref_other_than_head(tmp_path):
    old, new = _commits(tmp_path)
    subprocess.run(["git", "reset", "-q", "--hard", old], cwd=tmp_path, check=True)
    result = run_hook(tmp_path, f"refs/heads/x {new} refs/heads/main {old}\n")
    assert result.returncode == 1 and "HEAD" in result.stderr


def _safe_push(tmp_path: Path) -> subprocess.CompletedProcess:
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(Path(ROOT) / "scripts" / "safe_push.sh", tmp_path / "scripts" / "safe_push.sh")
    return subprocess.run(["bash", "scripts/safe_push.sh", "--check"], cwd=tmp_path, capture_output=True, text=True)


def test_safe_push_rejects_wrong_origin(tmp_path):
    _commits(tmp_path)
    subprocess.run(["git", "branch", "-M", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "remote", "add", "origin", "https://github.com/someone/other.git"], cwd=tmp_path, check=True)
    result = _safe_push(tmp_path)
    assert result.returncode == 1 and "origin" in result.stderr


def test_safe_push_rejects_other_branch(tmp_path):
    _commits(tmp_path)
    subprocess.run(["git", "checkout", "-qb", "feature"], cwd=tmp_path, check=True)
    result = _safe_push(tmp_path)
    assert result.returncode == 1 and "main뿐" in result.stderr


def test_hook_rejects_dirty_tree(tmp_path):
    old, new = _commits(tmp_path)
    (tmp_path / "f").write_text("dirty")
    result = run_hook(tmp_path, f"refs/heads/main {new} refs/heads/main {old}\n")
    assert result.returncode == 1 and "커밋하지 않은" in result.stderr
