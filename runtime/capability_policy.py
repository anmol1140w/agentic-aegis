"""Capability classification for sandbox command approval."""

from __future__ import annotations

import shlex
from enum import StrEnum
from pathlib import Path


class CommandCapability(StrEnum):
    READ_ONLY = "read_only"
    DEVELOPMENT = "development"
    WORKSPACE_MUTATION = "workspace_mutation"
    NETWORK = "network"
    SYSTEM_MUTATION = "system_mutation"
    PRIVILEGED = "privileged"
    BLOCKED = "blocked"


_READ_ONLY = {"pwd", "ls", "find", "locate", "which", "whereis", "file", "stat", "readlink",
              "realpath", "basename", "dirname", "du", "df", "cat", "tac", "head", "tail", "less",
              "more", "nl", "wc", "strings", "od", "hexdump", "xxd", "cut", "paste", "column",
              "fold", "fmt", "sed", "awk", "grep", "egrep", "fgrep", "sort", "uniq", "tr", "diff",
              "cmp", "comm", "join", "rg", "fd", "bat", "type", "command", "apropos", "echo", "printf",
              "env", "printenv", "hostname", "uname", "arch", "id", "whoami", "groups", "tty", "uptime",
              "date", "locale", "ps", "pgrep", "pidof", "pstree", "git"}
_DEVELOPMENT = {"python", "python3", "pytest", "pip", "pip3", "gcc", "g++", "clang", "clang++", "make",
                "cmake", "ninja", "node", "npm", "yarn", "pnpm", "cargo", "rustc", "go", "java", "javac",
                "ruby", "perl", "php", "dotnet", "mvn", "gradle", "tar", "gzip", "gunzip", "zip", "unzip",
                "xz", "unxz", "bzip2", "bunzip2", "7z"}
_WORKSPACE_MUTATION = {"touch", "mkdir", "cp", "mv", "rm", "rmdir", "ln", "install", "truncate", "tee", "chmod", "umask",
                       "split", "csplit", "expand", "unexpand", "rev"}
_NETWORK = {"curl", "wget", "nc", "netcat", "ssh", "scp", "ftp", "telnet", "ping", "nslookup", "dig"}
_SYSTEM_MUTATION = {"apt", "apt-get", "dpkg", "rpm", "dnf", "yum", "chown", "chgrp", "setfacl", "mount",
                    "umount", "systemctl", "service", "shred"}
_PRIVILEGED = {"sudo", "su", "doas", "pkexec", "setuid"}
_BLOCKED = {"mkfs", "fdisk", "parted", "dd", "shutdown", "reboot", "poweroff", "halt", "insmod", "rmmod",
            "modprobe", "iptables", "nft"}


def classify_command(command: str) -> CommandCapability:
    """Classify the first executable; shell policy handles operators separately."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return CommandCapability.BLOCKED
    if not tokens:
        return CommandCapability.BLOCKED
    executable = Path(tokens[0]).name.lower()
    if executable in _PRIVILEGED:
        return CommandCapability.PRIVILEGED
    if executable in _BLOCKED:
        return CommandCapability.BLOCKED
    if executable in _NETWORK:
        return CommandCapability.NETWORK
    if executable in _SYSTEM_MUTATION:
        return CommandCapability.SYSTEM_MUTATION
    if executable in _WORKSPACE_MUTATION:
        return CommandCapability.WORKSPACE_MUTATION
    if executable in _DEVELOPMENT:
        return CommandCapability.DEVELOPMENT
    if executable in _READ_ONLY:
        return CommandCapability.READ_ONLY
    # Unknown commands are allowed only inside the isolated backend, where
    # the approval gate and namespace boundary still apply.
    return CommandCapability.DEVELOPMENT
