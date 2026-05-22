#!/usr/bin/env python3
"""Sync files changed by one git commit to a remote checkout.

This is useful when the remote machine cannot pull from GitHub but can be
reached through SSH from the local workstation.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommitChanges:
    uploads: tuple[str, ...]
    deletes: tuple[str, ...]


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[bytes] | None:
    printable = shlex.join(cmd)
    if dry_run:
        print(f"+ {printable}")
        return None
    print(f"+ {printable}")
    return subprocess.run(cmd, cwd=cwd, input=input_bytes, check=True)


def capture(cmd: list[str], *, cwd: Path) -> bytes:
    return subprocess.check_output(cmd, cwd=cwd)


def git_root(start: Path) -> Path:
    root = capture(["git", "rev-parse", "--show-toplevel"], cwd=start)
    return Path(root.decode("utf-8").strip())


def parse_name_status(raw: bytes) -> CommitChanges:
    parts = raw.split(b"\0")
    if parts and parts[-1] == b"":
        parts.pop()

    uploads: list[str] = []
    deletes: list[str] = []
    i = 0
    while i < len(parts):
        status = parts[i].decode("utf-8")
        i += 1
        code = status[0]

        if code in {"R", "C"}:
            if i + 1 >= len(parts):
                raise ValueError(f"malformed git diff-tree output near {status!r}")
            old_path = parts[i].decode("utf-8")
            new_path = parts[i + 1].decode("utf-8")
            i += 2
            if code == "R":
                deletes.append(old_path)
            uploads.append(new_path)
            continue

        if i >= len(parts):
            raise ValueError(f"malformed git diff-tree output near {status!r}")
        path = parts[i].decode("utf-8")
        i += 1

        if code == "D":
            deletes.append(path)
        elif code in {"A", "M", "T"}:
            uploads.append(path)
        else:
            raise ValueError(f"unsupported git status {status!r} for {path!r}")

    return CommitChanges(tuple(sorted(set(uploads))), tuple(sorted(set(deletes))))


def changes_for_commit(repo: Path, commit: str) -> CommitChanges:
    raw = capture(
        [
            "git",
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-M",
            "-z",
            commit,
        ],
        cwd=repo,
    )
    return parse_name_status(raw)


def verify_commit(repo: Path, commit: str) -> str:
    return (
        capture(["git", "rev-parse", "--verify", f"{commit}^{{commit}}"], cwd=repo)
        .decode("utf-8")
        .strip()
    )


def quote_remote(path: str) -> str:
    return shlex.quote(path)


def remote_rsync_target(remote: str, remote_dir: str) -> str:
    clean_dir = remote_dir.rstrip("/") + "/"
    return f"{remote}:{quote_remote(clean_dir)}"


def chunked(items: tuple[str, ...], size: int) -> list[tuple[str, ...]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def extract_commit_paths(repo: Path, commit: str, paths: tuple[str, ...], output_dir: Path) -> None:
    archive_cmd = ["git", "archive", "--format=tar", commit, "--", *paths]
    extract_cmd = ["tar", "-xf", "-", "-C", os.fspath(output_dir)]
    print(f"+ {shlex.join(archive_cmd)} | {shlex.join(extract_cmd)}")
    with subprocess.Popen(archive_cmd, cwd=repo, stdout=subprocess.PIPE) as archive_proc:
        try:
            subprocess.run(extract_cmd, stdin=archive_proc.stdout, check=True)
        finally:
            if archive_proc.stdout is not None:
                archive_proc.stdout.close()
        archive_return_code = archive_proc.wait()
    if archive_return_code != 0:
        raise subprocess.CalledProcessError(archive_return_code, archive_cmd)


def sync_changes(
    *,
    repo: Path,
    commit: str,
    changes: CommitChanges,
    remote: str,
    remote_dir: str,
    apply: bool,
) -> None:
    dry_run = not apply

    if apply:
        run(["ssh", remote, "mkdir", "-p", "--", quote_remote(remote_dir)])

    if changes.uploads:
        file_list = b"".join(path.encode("utf-8") + b"\0" for path in changes.uploads)
        rsync_cmd = [
            "rsync",
            "-a",
            "--from0",
            "--files-from=-",
            "./",
            remote_rsync_target(remote, remote_dir),
        ]
        if dry_run:
            run(rsync_cmd, dry_run=True)
        else:
            with tempfile.TemporaryDirectory(prefix="sync-commit-") as temp_dir:
                archive_dir = Path(temp_dir)
                extract_commit_paths(repo, commit, changes.uploads, archive_dir)
                run(rsync_cmd, cwd=archive_dir, input_bytes=file_list)

    if changes.deletes:
        for batch in chunked(changes.deletes, 100):
            quoted_paths = " ".join(quote_remote(path) for path in batch)
            remote_cmd = f"cd {quote_remote(remote_dir)} && rm -f -- {quoted_paths}"
            run(["ssh", remote, remote_cmd], dry_run=dry_run)


def print_plan(commit_sha: str, changes: CommitChanges, remote: str, remote_dir: str) -> None:
    print(f"Commit: {commit_sha}")
    print(f"Remote: {remote}:{remote_dir}")
    print(f"Upload files: {len(changes.uploads)}")
    for path in changes.uploads:
        print(f"  U {path}")
    print(f"Delete files: {len(changes.deletes)}")
    for path in changes.deletes:
        print(f"  D {path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync files changed by one local git commit to a remote checkout."
    )
    parser.add_argument(
        "commit",
        nargs="?",
        default="HEAD",
        help="Commit to sync. Default: HEAD.",
    )
    parser.add_argument(
        "--remote",
        default="tj_server",
        help="SSH host alias. Default: tj_server.",
    )
    parser.add_argument(
        "--remote-dir",
        default=None,
        help="Remote repository directory. Default: same absolute path as local repo.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write to the remote. Without this flag the script is a dry-run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    repo = git_root(Path.cwd())
    commit_sha = verify_commit(repo, args.commit)
    remote_dir = args.remote_dir or os.fspath(repo)

    changes = changes_for_commit(repo, commit_sha)
    print_plan(commit_sha, changes, args.remote, remote_dir)
    if not changes.uploads and not changes.deletes:
        print("Nothing to sync.")
        return 0

    if not args.apply:
        print("Dry-run only. Re-run with --apply to write to the remote.")

    sync_changes(
        repo=repo,
        commit=commit_sha,
        changes=changes,
        remote=args.remote,
        remote_dir=remote_dir,
        apply=args.apply,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
