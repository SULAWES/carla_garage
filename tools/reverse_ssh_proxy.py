#!/usr/bin/env python3
"""Expose a local proxy to a remote host through an SSH reverse tunnel.

Example:
    python tools/reverse_ssh_proxy.py

Then, on the remote host, run the printed export commands before using git or
curl. The tunnel stays alive until this script is stopped with Ctrl-C.
"""

from __future__ import annotations

import argparse
import secrets
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass


DEFAULT_REMOTE_PORT_MIN = 20000
DEFAULT_REMOTE_PORT_MAX = 60999
DEFAULT_REMOTE = "tj_server"
DEFAULT_LOGIN_NODE = "logini01"
DEFAULT_LOGIN_NODE_SSH_PORT = 10022


@dataclass(frozen=True)
class ProxyEnv:
    http_proxy: str
    all_proxy: str


def parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid port: {value!r}") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port out of range: {port}")
    return port


def parse_remote_port(value: str) -> int | None:
    if value.lower() in {"auto", "random", "rand"}:
        return None
    return parse_port(value)


def check_local_proxy(host: str, port: int, timeout: float) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(
            f"cannot connect to local proxy {host}:{port}; "
            "start the local proxy first or pass --skip-local-check"
        ) from exc


def remote_proxy_env(remote_host: str, remote_port: int, *, all_proxy_scheme: str) -> ProxyEnv:
    if remote_host in {"", "*", "0.0.0.0", "::"}:
        remote_host = "127.0.0.1"
    http_proxy = f"http://{remote_host}:{remote_port}"
    all_proxy = f"{all_proxy_scheme}://{remote_host}:{remote_port}"
    return ProxyEnv(http_proxy=http_proxy, all_proxy=all_proxy)


def print_remote_usage(env: ProxyEnv) -> None:
    print("\nRemote shell temporary proxy:")
    print(f"  export http_proxy={shlex.quote(env.http_proxy)}")
    print(f"  export https_proxy={shlex.quote(env.http_proxy)}")
    print(f"  export HTTP_PROXY={shlex.quote(env.http_proxy)}")
    print(f"  export HTTPS_PROXY={shlex.quote(env.http_proxy)}")
    print(f"  export all_proxy={shlex.quote(env.all_proxy)}")
    print(f"  export ALL_PROXY={shlex.quote(env.all_proxy)}")
    print("\nRemote Git-only temporary proxy:")
    print(f"  git -c http.proxy={shlex.quote(env.http_proxy)} ls-remote https://github.com/git/git.git")
    print("\nRemote quick check:")
    print("  curl -I https://github.com")
    print()


def build_inner_ssh_command(args: argparse.Namespace, login_node_port: int, gateway_port: int) -> list[str]:
    forward_spec = (
        f"{args.login_node_bind}:{login_node_port}:"
        f"127.0.0.1:{gateway_port}"
    )
    cmd = [
        "ssh",
        "-N",
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        forward_spec,
        "-p",
        str(args.login_node_ssh_port),
    ]
    for option in args.login_node_ssh_option:
        cmd.extend(["-o", option])
    cmd.append(args.login_node)
    return cmd


def build_direct_ssh_command(args: argparse.Namespace, remote_port: int) -> list[str]:
    forward_spec = (
        f"{args.remote_bind}:{remote_port}:"
        f"{args.local_host}:{args.local_port}"
    )
    cmd = [
        "ssh",
        "-N",
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        forward_spec,
    ]
    if args.ssh_port is not None:
        cmd.extend(["-p", str(args.ssh_port)])
    if args.identity_file:
        cmd.extend(["-i", args.identity_file])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    cmd.append(args.remote)
    return cmd


def build_nested_ssh_command(args: argparse.Namespace, gateway_port: int, login_node_port: int) -> list[str]:
    gateway_forward_spec = (
        f"{args.remote_bind}:{gateway_port}:"
        f"{args.local_host}:{args.local_port}"
    )
    inner_cmd = build_inner_ssh_command(args, login_node_port, gateway_port)
    cmd = [
        "ssh",
        "-T",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
        "-R",
        gateway_forward_spec,
    ]
    if args.ssh_port is not None:
        cmd.extend(["-p", str(args.ssh_port)])
    if args.identity_file:
        cmd.extend(["-i", args.identity_file])
    for option in args.ssh_option:
        cmd.extend(["-o", option])
    cmd.extend([args.remote, shlex.join(inner_cmd)])
    return cmd


def random_remote_port(args: argparse.Namespace) -> int:
    span = args.remote_port_max - args.remote_port_min + 1
    return args.remote_port_min + secrets.randbelow(span)


def wait_for_early_exit(proc: subprocess.Popen[bytes], timeout: float) -> int | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = proc.poll()
        if return_code is not None:
            return return_code
        time.sleep(0.1)
    return proc.poll()


def wait_for_tunnel(proc: subprocess.Popen[bytes]) -> int:
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\nStopping SSH reverse tunnel...")
        proc.terminate()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return 130


def run_ssh_tunnel(cmd: list[str], env: ProxyEnv) -> int:
    print("Starting SSH reverse tunnel:")
    print(f"  {shlex.join(cmd)}")
    print("\nAfter SSH authentication succeeds, use these commands on the remote host.")
    print_remote_usage(env)
    print("Stop the tunnel with Ctrl-C.\n")

    proc = subprocess.Popen(cmd)
    return wait_for_tunnel(proc)


def run_direct_ssh_tunnel_with_random_port(args: argparse.Namespace) -> int:
    for attempt in range(1, args.retry_count + 1):
        remote_port = random_remote_port(args)
        env = remote_proxy_env(
            args.remote_bind,
            remote_port,
            all_proxy_scheme=args.all_proxy_scheme,
        )
        cmd = build_direct_ssh_command(args, remote_port)
        print("Starting SSH reverse tunnel:")
        print(f"  {shlex.join(cmd)}")
        print(f"\nRandom remote port attempt {attempt}/{args.retry_count}: {remote_port}")

        proc = subprocess.Popen(cmd)
        return_code = wait_for_early_exit(proc, args.startup_wait)
        if return_code is None:
            print("\nAfter SSH authentication succeeds, use these commands on the remote host.")
            print_remote_usage(env)
            print("Stop the tunnel with Ctrl-C.\n")
            return wait_for_tunnel(proc)

        if attempt == args.retry_count:
            return return_code

        print(f"SSH exited early with status {return_code}; retrying with another random port.\n")

    return 1


def run_ssh_tunnel_with_random_port(args: argparse.Namespace) -> int:
    for attempt in range(1, args.retry_count + 1):
        gateway_port = random_remote_port(args)
        login_node_port = random_remote_port(args)
        env = remote_proxy_env(
            args.login_node_bind,
            login_node_port,
            all_proxy_scheme=args.all_proxy_scheme,
        )
        cmd = build_nested_ssh_command(args, gateway_port, login_node_port)
        print("Starting SSH reverse tunnel:")
        print(f"  {shlex.join(cmd)}")
        print(
            f"\nRandom port attempt {attempt}/{args.retry_count}: "
            f"gateway={gateway_port}, login_node={login_node_port}"
        )

        proc = subprocess.Popen(cmd)
        return_code = wait_for_early_exit(proc, args.startup_wait)
        if return_code is None:
            print("\nAfter SSH authentication succeeds, use these commands on the remote host.")
            print_remote_usage(env)
            print("Stop the tunnel with Ctrl-C.\n")
            return wait_for_tunnel(proc)

        if attempt == args.retry_count:
            return return_code

        print(f"SSH exited early with status {return_code}; retrying with another random port.\n")

    return 1


def validate_args(args: argparse.Namespace) -> None:
    if args.remote_port_min > args.remote_port_max:
        raise ValueError(
            f"--remote-port-min ({args.remote_port_min}) cannot be greater than "
            f"--remote-port-max ({args.remote_port_max})"
        )
    if args.retry_count < 1:
        raise ValueError("--retry-count must be at least 1")
    if args.startup_wait <= 0:
        raise ValueError("--startup-wait must be positive")


def print_dry_run(args: argparse.Namespace) -> None:
    login_node_port = args.remote_port if args.remote_port is not None else random_remote_port(args)
    env_host = args.remote_bind if args.direct else args.login_node_bind
    env = remote_proxy_env(env_host, login_node_port, all_proxy_scheme=args.all_proxy_scheme)
    print_remote_usage(env)
    if args.direct:
        cmd = build_direct_ssh_command(args, login_node_port)
    else:
        gateway_port = random_remote_port(args)
        cmd = build_nested_ssh_command(args, gateway_port, login_node_port)
    print("SSH command:")
    print(f"  {shlex.join(cmd)}")


def run(args: argparse.Namespace) -> int:
    if args.direct:
        if args.remote_port is None:
            return run_direct_ssh_tunnel_with_random_port(args)

        env = remote_proxy_env(
            args.remote_bind,
            args.remote_port,
            all_proxy_scheme=args.all_proxy_scheme,
        )
        cmd = build_direct_ssh_command(args, args.remote_port)
        return run_ssh_tunnel(cmd, env)

    if args.remote_port is None:
        return run_ssh_tunnel_with_random_port(args)

    env = remote_proxy_env(
        args.login_node_bind,
        args.remote_port,
        all_proxy_scheme=args.all_proxy_scheme,
    )
    gateway_port = random_remote_port(args)
    cmd = build_nested_ssh_command(args, gateway_port, args.remote_port)
    return run_ssh_tunnel(cmd, env)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open an ssh -R tunnel so a remote machine can use this "
            "workstation's local proxy, useful for temporary GitHub access."
        )
    )
    parser.add_argument(
        "remote",
        nargs="?",
        default=DEFAULT_REMOTE,
        help=(
            "Gateway SSH target, for example user@server or a Host alias in ~/.ssh/config. "
            f"Default: {DEFAULT_REMOTE}"
        ),
    )
    parser.add_argument(
        "--local-host",
        default="127.0.0.1",
        help="Local proxy host to forward to. Default: %(default)s",
    )
    parser.add_argument(
        "--local-port",
        type=parse_port,
        default=10808,
        help="Local proxy port to forward to. Default: %(default)s",
    )
    parser.add_argument(
        "--remote-bind",
        default="127.0.0.1",
        help="Gateway bind address. Keep loopback on shared login nodes. Default: %(default)s",
    )
    parser.add_argument(
        "--remote-port",
        type=parse_remote_port,
        default=None,
        help=(
            "Final login-node port exposed on the server. Use 'random' for a random high "
            "port. Default: random"
        ),
    )
    parser.add_argument(
        "--remote-port-min",
        type=parse_port,
        default=DEFAULT_REMOTE_PORT_MIN,
        help="Lowest remote port used by random mode. Default: %(default)s",
    )
    parser.add_argument(
        "--remote-port-max",
        type=parse_port,
        default=DEFAULT_REMOTE_PORT_MAX,
        help="Highest remote port used by random mode. Default: %(default)s",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=8,
        help="Random-port retry count if ssh exits early. Default: %(default)s",
    )
    parser.add_argument(
        "--startup-wait",
        type=float,
        default=1.5,
        help="Seconds to wait for immediate ssh failure before printing remote commands. Default: %(default)s",
    )
    parser.add_argument(
        "--login-node",
        default=DEFAULT_LOGIN_NODE,
        help=(
            "Fixed login node reached from the gateway after SSH login. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--login-node-ssh-port",
        type=parse_port,
        default=DEFAULT_LOGIN_NODE_SSH_PORT,
        help="SSH port used from the gateway to the fixed login node. Default: %(default)s",
    )
    parser.add_argument(
        "--login-node-bind",
        default="127.0.0.1",
        help="Final login-node bind address. Keep loopback on shared nodes. Default: %(default)s",
    )
    parser.add_argument(
        "--login-node-ssh-option",
        action="append",
        default=[],
        help="Additional ssh -o option for the gateway -> login-node hop. Can be specified multiple times.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Do a single-hop reverse tunnel directly to the gateway target.",
    )
    parser.add_argument(
        "-p",
        "--ssh-port",
        type=parse_port,
        help="Gateway SSH server port. Leave unset to use ~/.ssh/config.",
    )
    parser.add_argument(
        "-i",
        "--identity-file",
        help="SSH identity file.",
    )
    parser.add_argument(
        "-o",
        "--ssh-option",
        action="append",
        default=[],
        help="Additional ssh -o option. Can be specified multiple times.",
    )
    parser.add_argument(
        "--all-proxy-scheme",
        default="socks5",
        choices=("socks5", "socks", "http"),
        help="Scheme printed for all_proxy on the remote host. Default: %(default)s",
    )
    parser.add_argument(
        "--skip-local-check",
        action="store_true",
        help="Do not test whether the local proxy port is reachable before starting ssh.",
    )
    parser.add_argument(
        "--check-timeout",
        type=float,
        default=2.0,
        help="Timeout in seconds for the local proxy connectivity check. Default: %(default)s",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ssh command and remote proxy commands without starting the tunnel.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if shutil.which("ssh") is None:
        print("error: ssh executable not found in PATH", file=sys.stderr)
        return 127

    if not args.skip_local_check:
        try:
            check_local_proxy(args.local_host, args.local_port, args.check_timeout)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.dry_run:
        print_dry_run(args)
        return 0

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
