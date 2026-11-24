"""Linux candidate-process boundary used by the trusted remote launcher.

The current A40 deployment uses a hardened root-filesystem Unix-permission
boundary, seccomp, ``setpriv``, and a sanitized environment.  The candidate
sees only explicitly approved read-only inputs plus its private output area.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import os
import pathlib
import platform
import pwd
import shutil
import sys


# The process-runtime closure is immutable. Bump this generation whenever a
# file copied by _prepare_process_runtime changes, so an existing read-only
# runtime is never mixed with a newer trusted implementation.
SANDBOX_POLICY_VERSION = "autopareto_linux_rootfs_permissions_seccomp_setpriv_v13"
PROCESS_SANDBOX_POLICY_VERSION = SANDBOX_POLICY_VERSION
DEFAULT_REGISTERED_CACHE = "/var/cache/autopareto/pair_a_registered_cache"
CANDIDATE_USER = "autopareto-candidate"


class ProcessSandboxUnavailableError(RuntimeError):
    """The A40 host cannot enforce the candidate process boundary."""


# Linux x86_64 syscall and Landlock constants.  The process boundary fails
# closed on another architecture rather than relying on an unregistered map.
_LANDLOCK_CREATE_RULESET = 444
_LANDLOCK_ADD_RULE = 445
_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
_LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
_LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
_LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
_LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
_LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
_LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
_LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
_LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
_LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
_LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
_LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
_LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
_LANDLOCK_ACCESS_FS_REFER = 1 << 13
_LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14
_LANDLOCK_ACCESS_NET_CONNECT_TCP = 1 << 0
_LANDLOCK_ACCESS_NET_BIND_TCP = 1 << 1

_READ_ACCESS = (
    _LANDLOCK_ACCESS_FS_EXECUTE
    | _LANDLOCK_ACCESS_FS_READ_FILE
    | _LANDLOCK_ACCESS_FS_READ_DIR
)
_WRITE_ACCESS = (
    _LANDLOCK_ACCESS_FS_WRITE_FILE
    | _LANDLOCK_ACCESS_FS_REMOVE_DIR
    | _LANDLOCK_ACCESS_FS_REMOVE_FILE
    | _LANDLOCK_ACCESS_FS_MAKE_CHAR
    | _LANDLOCK_ACCESS_FS_MAKE_DIR
    | _LANDLOCK_ACCESS_FS_MAKE_REG
    | _LANDLOCK_ACCESS_FS_MAKE_SOCK
    | _LANDLOCK_ACCESS_FS_MAKE_FIFO
    | _LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | _LANDLOCK_ACCESS_FS_MAKE_SYM
    | _LANDLOCK_ACCESS_FS_REFER
    | _LANDLOCK_ACCESS_FS_TRUNCATE
)


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
try:
    _LIBC.capset.restype = ctypes.c_int
except AttributeError:
    # macOS has no Linux capset symbol; importing this module for static
    # validation remains safe, while an actual non-Linux launch fails closed.
    pass


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int32)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _require_bwrap() -> None:
    if os.name != "posix" or shutil.which("bwrap") is None:
        raise RuntimeError(
            "Linux scientific candidate execution requires bubblewrap (or a dedicated "
            "container boundary); no unsandboxed fallback exists"
        )


def _runtime_paths(repository: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    worker_dir = repository / "study/contracts_2026"
    venv = pathlib.Path(
        os.environ.get("AUTOPARETO_STUDY_VENV", "/usr/local")
    ).resolve(strict=True)
    if not venv.is_dir():
        raise FileNotFoundError(f"protected study virtual environment is missing: {venv}")
    return worker_dir, venv


def _translated_command(
    command: list[str], *, context: pathlib.Path, run_directory: pathlib.Path, venv: pathlib.Path
) -> list[str]:
    translated = [_replace_path(value, venv, "/opt/autopareto-venv") for value in command]
    return [
        _replace_path(_replace_path(value, context, "/candidate"), run_directory, "/run")
        for value in translated
    ]


def _base_bwrap(
    command: list[str],
    *,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
    study_files: tuple[str, ...],
    protected_files: tuple[str, ...] = (),
) -> list[str]:
    """Build one minimal read-only candidate/inference root.

    Both training and fresh inference use this function.  Keeping inference in the
    same boundary is essential: loading candidate definitions in the trusted parent
    would otherwise let a candidate read the repository after training completed.
    """
    _require_bwrap()
    context = context.resolve()
    run_directory = run_directory.resolve()
    repository = repository.resolve()
    if not context.is_dir() or not run_directory.is_dir():
        raise ValueError("candidate context and run directory must already exist")
    worker_dir, venv = _runtime_paths(repository)
    for relative in study_files:
        path = worker_dir / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"protected study module is incomplete: {path}")
    for relative in protected_files:
        path = repository / "protected" / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"protected inference module is incomplete: {path}")

    args = [
        "bwrap", "--die-with-parent", "--unshare-all", "--new-session", "--clearenv",
        "--tmpfs", "/", "--dir", "/usr", "--ro-bind", "/usr", "/usr",
        "--dir", "/bin", "--ro-bind", "/bin", "/bin",
        "--dir", "/lib", "--ro-bind", "/lib", "/lib",
        "--dir", "/lib64", "--ro-bind", "/lib64", "/lib64",
        "--dir", "/etc", "--ro-bind", "/etc", "/etc",
        "--dir", "/sys", "--ro-bind", "/sys", "/sys",
        "--dir", "/opt", "--dir", "/opt/autopareto-venv",
        "--ro-bind", str(venv), "/opt/autopareto-venv",
        "--dir", "/opt/study", "--dir", "/opt/study/contracts_2026",
    ]
    for relative in study_files:
        args.extend(("--ro-bind", str(worker_dir / relative), f"/opt/study/contracts_2026/{relative}"))
    if protected_files:
        args.extend(("--dir", "/opt/protected"))
        for relative in protected_files:
            args.extend(("--ro-bind", str(repository / "protected" / relative), f"/opt/protected/{relative}"))
    args.extend([
        "--ro-bind", str(context), "/candidate",
        "--dir", "/run", "--bind", str(run_directory), "/run",
        "--dir", "/candidate-home", "--tmpfs", "/tmp",
        "--proc", "/proc", "--dev", "/dev", "--dir", "/dev/shm", "--tmpfs", "/dev/shm",
        "--setenv", "HOME", "/candidate-home", "--setenv", "PYTHONNOUSERSITE", "1",
        "--setenv", "PYTHONPATH", "/opt",
    ])
    # Expose only NVIDIA character devices needed by CUDA, never the host's full
    # /dev tree.  A plain ``--dev /dev`` is intentionally insufficient for GPU use.
    for device in sorted(pathlib.Path("/dev").glob("nvidia*")):
        if device.is_char_device():
            args.extend(("--dev-bind", str(device), f"/dev/{device.name}"))
    for name in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME"):
        value = os.environ.get(name)
        if value:
            args.extend(("--setenv", name, value))
    args.extend(("--setenv", "AUTOPARETO_CANDIDATE_SANDBOX", SANDBOX_POLICY_VERSION))
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", DEFAULT_REGISTERED_CACHE)
    ).resolve()
    if registered_cache.is_dir():
        args.extend([
            "--dir", "/candidate-home/.cache",
            "--ro-bind", str(registered_cache), "/candidate-home/.cache/autoresearch",
        ])
    args.extend(("--chdir", "/candidate", "--", *_translated_command(
        command, context=context, run_directory=run_directory, venv=venv
    )))
    return args


def landlock_abi() -> int:
    """Return the host Landlock ABI, or fail closed when it is unavailable."""
    if platform.machine() != "x86_64":
        raise ProcessSandboxUnavailableError(
            f"process sandbox has no registered syscall map for {platform.machine()}"
        )
    ctypes.set_errno(0)
    result = _LIBC.syscall(
        _LANDLOCK_CREATE_RULESET,
        None,
        0,
        _LANDLOCK_CREATE_RULESET_VERSION,
    )
    if result < 0:
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"Landlock ABI query failed: {errno.errorcode.get(error, error)}"
        )
    return int(result)


def _landlock_add_path(ruleset_fd: int, path: pathlib.Path, access: int) -> None:
    path = path.resolve(strict=True)
    try:
        path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError as exc:
        raise ProcessSandboxUnavailableError(f"cannot open allowlisted path {path}: {exc}") from exc
    try:
        rule = _LandlockPathBeneathAttr(access, path_fd)
        ctypes.set_errno(0)
        result = _LIBC.syscall(
            _LANDLOCK_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(rule),
            0,
        )
        if result < 0:
            error = ctypes.get_errno()
            raise ProcessSandboxUnavailableError(
                f"Landlock rule failed for {path}: {errno.errorcode.get(error, error)}"
            )
    finally:
        os.close(path_fd)


def _landlock_restrict(
    *,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
    process_runtime: pathlib.Path,
    candidate_home: pathlib.Path,
    study_files: tuple[str, ...],
    protected_files: tuple[str, ...],
) -> None:
    """Install a deny-by-default filesystem and TCP policy in this process."""
    if landlock_abi() < 4:
        raise ProcessSandboxUnavailableError("Landlock ABI 4 is required for the registered policy")
    context = context.resolve(strict=True)
    run_directory = run_directory.resolve(strict=True)
    repository = repository.resolve(strict=True)
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", DEFAULT_REGISTERED_CACHE)
    ).resolve()
    venv = pathlib.Path(
        os.environ.get("AUTOPARETO_STUDY_VENV", "/usr/local")
    ).resolve(strict=True)

    ruleset_attr = _LandlockRulesetAttr(
        (1 << 15) - 1,
        _LANDLOCK_ACCESS_NET_CONNECT_TCP | _LANDLOCK_ACCESS_NET_BIND_TCP,
    )
    ctypes.set_errno(0)
    ruleset_fd = _LIBC.syscall(
        _LANDLOCK_CREATE_RULESET,
        ctypes.byref(ruleset_attr),
        ctypes.sizeof(ruleset_attr),
        0,
    )
    if ruleset_fd < 0:
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"Landlock ruleset creation failed: {errno.errorcode.get(error, error)}"
        )

    try:
        # System libraries, Python dependencies, CUDA support files, and the
        # registered read-only data cache are approved inputs.  The repository
        # root, results tree, and credentials are intentionally absent.
        for path in (
            pathlib.Path("/usr"),
            pathlib.Path("/bin"),
            pathlib.Path("/lib"),
            pathlib.Path("/lib64"),
            pathlib.Path("/etc"),
            pathlib.Path("/proc"),
            pathlib.Path("/sys"),
            pathlib.Path("/dev"),
            pathlib.Path("/dev/shm"),
            pathlib.Path("/run"),
            venv,
            registered_cache,
            context,
            candidate_home,
        ):
            if not path.exists():
                continue
            access = _READ_ACCESS
            if path in (pathlib.Path("/etc"), pathlib.Path("/proc"), pathlib.Path("/sys")):
                access = _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_READ_DIR
            elif path in (pathlib.Path("/dev"), pathlib.Path("/dev/shm")):
                access = (
                    _LANDLOCK_ACCESS_FS_READ_FILE
                    | _LANDLOCK_ACCESS_FS_READ_DIR
                    | _LANDLOCK_ACCESS_FS_WRITE_FILE
                )
            elif path == pathlib.Path("/run"):
                access = _LANDLOCK_ACCESS_FS_READ_DIR
            elif path == registered_cache or path == context:
                access = _LANDLOCK_ACCESS_FS_READ_FILE | _LANDLOCK_ACCESS_FS_READ_DIR
            _landlock_add_path(ruleset_fd, path, access)
        _landlock_add_path(ruleset_fd, run_directory, _READ_ACCESS | _WRITE_ACCESS)
        cuda_runtime_cache = candidate_home / ".nv"
        if cuda_runtime_cache.exists():
            if cuda_runtime_cache.is_symlink() or not cuda_runtime_cache.is_dir():
                raise ProcessSandboxUnavailableError("candidate CUDA runtime cache is not a regular directory")
            # CUDA may create driver-runtime metadata below HOME during the
            # post-exec initialization.  This is a per-attempt cache, not the
            # registered data/tokenizer cache, and is the only writable input
            # subtree granted to the candidate.
            _landlock_add_path(ruleset_fd, cuda_runtime_cache, _READ_ACCESS | _WRITE_ACCESS)
        nvidia_persistenced = pathlib.Path("/run/nvidia-persistenced")
        if nvidia_persistenced.exists():
            _landlock_add_path(ruleset_fd, nvidia_persistenced, _READ_ACCESS)
        kernel_cache = registered_cache.parent / "huggingface/hub/models--kernels-community--flash-attn3"
        if kernel_cache.exists():
            _landlock_add_path(ruleset_fd, kernel_cache, _READ_ACCESS)

        # Only the named protected modules are readable.  Landlock path-beneath
        # rules are directory rules on this ABI/kernel, so the copied runtime
        # tree is verified to contain exactly these files before this directory
        # read rule is installed, and the directory is made immutable below.
        package_dir = process_runtime / "study"
        contracts_dir = package_dir / "contracts_2026"
        _landlock_add_path(ruleset_fd, process_runtime, _READ_ACCESS)
        _landlock_add_path(ruleset_fd, package_dir, _READ_ACCESS)
        _landlock_add_path(ruleset_fd, contracts_dir, _READ_ACCESS)
        if protected_files:
            protected_dir = process_runtime / "protected"
            _landlock_add_path(ruleset_fd, protected_dir, _READ_ACCESS)
    except BaseException:
        os.close(ruleset_fd)
        raise

    if _LIBC.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        error = ctypes.get_errno()
        os.close(ruleset_fd)
        raise ProcessSandboxUnavailableError(
            f"PR_SET_NO_NEW_PRIVS failed: {errno.errorcode.get(error, error)}"
        )
    ctypes.set_errno(0)
    result = _LIBC.syscall(_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    os.close(ruleset_fd)
    if result < 0:
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"Landlock restrict_self failed: {errno.errorcode.get(error, error)}"
        )


def _install_network_seccomp() -> None:
    """Deny non-Unix sockets while retaining CUDA and Unix-domain IPC."""
    if platform.machine() != "x86_64":
        raise ProcessSandboxUnavailableError("network seccomp has no registered syscall map")
    bpf_ld_abs_word = 0x20
    bpf_jmp_jeq = 0x05 | 0x10
    bpf_ret = 0x06
    seccomp_ret_errno = 0x00050000 | errno.EPERM
    seccomp_ret_allow = 0x7FFF0000
    audit_arch_x86_64 = 0xC000003E
    instructions = [
        _SockFilter(0x00 | bpf_ld_abs_word, 0, 0, 4),
        _SockFilter(bpf_jmp_jeq, 1, 0, audit_arch_x86_64),
        _SockFilter(bpf_ret, 0, 0, seccomp_ret_errno),
        _SockFilter(0x00 | bpf_ld_abs_word, 0, 0, 0),
        # socket(2) is the only creation path that needs argument inspection.
        # AF_UNIX is retained for local runtime IPC; every other family is
        # denied, which covers IPv4, IPv6, raw, packet, netlink, and vsock.
        _SockFilter(bpf_jmp_jeq, 0, 3, 41),
        _SockFilter(0x00 | bpf_ld_abs_word, 0, 0, 16),
        _SockFilter(bpf_jmp_jeq, 1, 0, 1),  # AF_UNIX -> common allow
        _SockFilter(bpf_ret, 0, 0, seccomp_ret_errno),
    ]
    instructions.append(_SockFilter(bpf_ret, 0, 0, seccomp_ret_allow))
    array_type = _SockFilter * len(instructions)
    array = array_type(*instructions)
    program = _SockFprog(len(instructions), ctypes.cast(array, ctypes.POINTER(_SockFilter)))
    if _LIBC.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"PR_SET_NO_NEW_PRIVS failed before seccomp: {errno.errorcode.get(error, error)}"
        )
    ctypes.set_errno(0)
    result = _LIBC.prctl(22, 2, ctypes.byref(program), 0, 0)  # PR_SET_SECCOMP/FILTER
    if result != 0:
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"seccomp filter installation failed: {errno.errorcode.get(error, error)}"
        )


def _preload_cuda_driver() -> None:
    """Initialize CUDA before dropping identity and installing seccomp."""
    try:
        import torch

        if not torch.cuda.is_available():
            raise ProcessSandboxUnavailableError("CUDA is unavailable in the candidate launcher")
        torch.cuda.init()
    except ProcessSandboxUnavailableError:
        raise
    except BaseException as exc:
        raise ProcessSandboxUnavailableError(f"CUDA pre-initialization failed: {exc}") from exc


def _drop_candidate_privileges(uid: int, gid: int) -> None:
    """Drop supplementary groups, UID/GID, and every Linux capability in place."""
    if os.geteuid() != 0:
        raise ProcessSandboxUnavailableError("candidate launcher must start as root to drop identity")
    if _LIBC.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"PR_SET_NO_NEW_PRIVS failed before identity drop: {errno.errorcode.get(error, error)}"
        )
    # PR_SET_KEEPCAPS=8, value 0: a UID transition must clear permitted caps.
    _LIBC.prctl(8, 0, 0, 0, 0)
    # PR_CAPBSET_DROP=24.  Only bits present in the current bounding set are
    # dropped; an EPERM for an already-absent bit is harmless.
    for capability in range(64):
        ctypes.set_errno(0)
        result = _LIBC.prctl(24, capability, 0, 0, 0)
        if result != 0 and ctypes.get_errno() not in (errno.EPERM, errno.EINVAL):
            error = ctypes.get_errno()
            raise ProcessSandboxUnavailableError(
                f"capability bounding-set drop failed: {errno.errorcode.get(error, error)}"
            )
    try:
        os.setgroups([])
        os.setresgid(gid, gid, gid)
        os.setresuid(uid, uid, uid)
    except OSError as exc:
        raise ProcessSandboxUnavailableError(f"candidate identity drop failed: {exc}") from exc
    header = _CapHeader(0x20080522, 0)
    data = (_CapData * 2)()
    capset = getattr(_LIBC, "capset", None)
    if capset is None:
        raise ProcessSandboxUnavailableError("Linux capset is unavailable")
    ctypes.set_errno(0)
    if capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise ProcessSandboxUnavailableError(
            f"effective capability drop failed: {errno.errorcode.get(error, error)}"
        )


def _exec_candidate_python(
    command: list[str], *, context: pathlib.Path, process_runtime: pathlib.Path
) -> None:
    """Replace the restricted launcher with a clean candidate Python process.

    The sandbox setup process must not remain on the candidate's Python stack.
    ``execve`` preserves only the already-installed OS restrictions and the
    explicitly inherited pipes; it does not preserve the trusted wrapper's
    Python frames or module globals.
    """
    if len(command) < 2 or not pathlib.Path(command[0]).name.startswith("python"):
        raise ProcessSandboxUnavailableError("candidate command must be a Python entry point")
    os.chdir(context)
    if command[1] == "-m":
        if len(command) < 3 or command[2] not in {
            "study.contracts_2026.candidate_process",
            "study.contracts_2026.inference_benchmark",
            "study.contracts_2026.inference_model_host",
        }:
            raise ProcessSandboxUnavailableError("candidate module is not on the protected allowlist")
    else:
        script = pathlib.Path(command[1]).resolve(strict=True)
        if not script.is_relative_to(context):
            raise ProcessSandboxUnavailableError(f"candidate script is outside its context: {script}")
    child_environment = dict(os.environ)
    child_environment["PYTHONPATH"] = os.pathsep.join((str(context), str(process_runtime)))
    os.execve(command[0], command, child_environment)
    raise AssertionError("execve returned unexpectedly")


def _candidate_identity() -> tuple[int, int]:
    try:
        entry = pwd.getpwnam(CANDIDATE_USER)
    except KeyError as exc:
        raise ProcessSandboxUnavailableError(
            f"dedicated candidate user {CANDIDATE_USER!r} is not provisioned"
        ) from exc
    if entry.pw_uid == 0 or entry.pw_gid == 0:
        raise ProcessSandboxUnavailableError("candidate identity must be non-root")
    return int(entry.pw_uid), int(entry.pw_gid)


def _prepare_process_runtime(
    *,
    repository: pathlib.Path,
    study_files: tuple[str, ...],
    protected_files: tuple[str, ...],
) -> pathlib.Path:
    """Create or verify a read-only package containing only required modules."""
    kind = "inference" if protected_files else "worker"
    # Include the policy revision so an immutable runtime generated by an older
    # deployment cannot block or silently mix with the current allowlist.
    runtime_root = pathlib.Path(
        os.environ.get(
            "AUTOPARETO_CANDIDATE_RUNTIME_ROOT",
            str(repository / f".candidate-process-runtime-{SANDBOX_POLICY_VERSION}"),
        )
    ).resolve()
    if runtime_root.exists() and (runtime_root.is_symlink() or not runtime_root.is_dir()):
        raise ProcessSandboxUnavailableError(f"candidate process runtime root is not a directory: {runtime_root}")
    runtime_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    runtime = runtime_root / f"{kind}-{SANDBOX_POLICY_VERSION}"
    if runtime.exists() and (runtime.is_symlink() or not runtime.is_dir()):
        raise ProcessSandboxUnavailableError(f"candidate process runtime is not a directory: {runtime}")
    runtime.mkdir(mode=0o755, exist_ok=True)
    study_root = runtime / "study" / "contracts_2026"
    study_root.mkdir(mode=0o755, parents=True, exist_ok=True)
    expected: dict[pathlib.Path, pathlib.Path] = {}
    for relative in study_files:
        expected[study_root / relative] = repository / "study/contracts_2026" / relative
    if protected_files:
        protected_root = runtime / "protected"
        protected_root.mkdir(mode=0o755, exist_ok=True)
        for relative in protected_files:
            expected[protected_root / relative] = repository / "protected" / relative
        # Phase 1 candidate sources import the protected telemetry/timing
        # recorder, which in turn imports the trusted append-only event
        # transport.  Both are copied into the isolated process runtime as
        # protected dependencies; neither is supplied by the candidate.
        if "phase1_runtime.py" in protected_files:
            expected[runtime / "event_log.py"] = repository / "event_log.py"
    for target, source in expected.items():
        if source.is_symlink() or not source.is_file():
            raise ProcessSandboxUnavailableError(f"required protected module is incomplete: {source}")
        if target.exists():
            if target.is_symlink() or not target.is_file() or _sha256_file(target) != _sha256_file(source):
                raise ProcessSandboxUnavailableError(f"candidate process runtime drifted: {target}")
        else:
            shutil.copy2(source, target)
        target.chmod(0o444)
    allowed_targets = set(expected)
    observed = {path for path in runtime.rglob("*") if path.is_file()}
    if observed != allowed_targets:
        raise ProcessSandboxUnavailableError(
            f"candidate process runtime contains unregistered files: {sorted(str(path) for path in observed - allowed_targets)}"
        )
    for directory in (runtime, runtime / "study", study_root, runtime / "protected"):
        if directory.exists():
            directory.chmod(0o555)
    return runtime


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepare_candidate_paths(context: pathlib.Path, run_directory: pathlib.Path, uid: int, gid: int) -> None:
    if not context.is_dir() or not run_directory.is_dir():
        raise ProcessSandboxUnavailableError("candidate context and output directory must exist")
    if os.geteuid() == 0:
        # Some provider workspace mounts reject chown even for container root.
        # The directory is a fresh, unique per-attempt path, so a root-owned
        # world-writable mode is the safe fallback; Landlock still confines all
        # writes to this directory and the child itself remains non-root.
        try:
            os.chown(run_directory, uid, gid)
            os.chmod(run_directory, 0o700)
        except PermissionError:
            os.chmod(run_directory, 0o777)
        os.chmod(context, 0o555)
        for path in context.iterdir():
            if path.is_symlink():
                raise ProcessSandboxUnavailableError(f"candidate context contains symlink {path}")
            if path.is_dir():
                os.chmod(path, 0o555)
            elif path.is_file():
                os.chmod(path, 0o444)
            else:
                raise ProcessSandboxUnavailableError(f"candidate context contains unsupported path {path}")
        for name in ("tmp", "torchinductor-cache", "triton-cache", "cuda-cache"):
            path = run_directory / name
            try:
                path.mkdir(mode=0o700, exist_ok=True)
                os.chown(path, uid, gid)
                os.chmod(path, 0o700)
            except PermissionError:
                # Provider workspace mounts may reject chown, but these are
                # fresh children of the unique Landlock-writeable output root.
                path.mkdir(mode=0o777, exist_ok=True)
                os.chmod(path, 0o777)


def _sanitized_candidate_environment(
    run_directory: pathlib.Path, process_runtime: pathlib.Path, candidate_home: pathlib.Path
) -> dict[str, str]:
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", DEFAULT_REGISTERED_CACHE)
    ).resolve()
    home = pathlib.Path(
        os.environ.get(
            "AUTOPARETO_CANDIDATE_HOME",
            str(registered_cache.parent.parent if registered_cache.parent.name == ".cache" else registered_cache.parent),
        )
    ).resolve()
    study_venv = pathlib.Path(
        os.environ.get("AUTOPARETO_STUDY_VENV", "/usr/local")
    ).resolve()
    environment = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(home),
        "TMPDIR": str(run_directory / "tmp"),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TORCHINDUCTOR_CACHE_DIR": str(run_directory / "torchinductor-cache"),
        "TRITON_CACHE_DIR": str(run_directory / "triton-cache"),
        "CUDA_CACHE_PATH": str(run_directory / "cuda-cache"),
        "PYTHONPATH": str(process_runtime),
        "AUTOPARETO_REGISTERED_CACHE": str(registered_cache),
        "AUTOPARETO_CANDIDATE_HOME": str(home),
        "AUTOPARETO_STUDY_VENV": str(study_venv),
        "AUTOPARETO_CANDIDATE_SANDBOX": PROCESS_SANDBOX_POLICY_VERSION,
        "HF_HOME": str(candidate_home / ".cache" / "huggingface"),
        "HF_HUB_CACHE": str(candidate_home / ".cache" / "huggingface" / "hub"),
        "HF_HUB_OFFLINE": "1",
    }
    for name in ("LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "CUDA_HOME"):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    n_star = os.environ.get("N_STAR")
    if n_star is not None:
        try:
            if int(n_star) <= 0:
                raise ValueError
        except ValueError as exc:
            raise ProcessSandboxUnavailableError("candidate N_STAR environment value is invalid") from exc
        environment["N_STAR"] = n_star
    return environment


def _copy_or_link_readonly_tree(
    source: pathlib.Path, target: pathlib.Path, *, approved_root: pathlib.Path | None = None
) -> None:
    """Create a private read-only view without exposing the source directory."""
    source = source.resolve(strict=True)
    if approved_root is None:
        approved_root = source
    else:
        approved_root = approved_root.resolve(strict=True)
    if not source.is_relative_to(approved_root):
        raise ProcessSandboxUnavailableError(f"candidate cache source escapes its approved tree: {source}")
    if source.is_symlink() or not source.is_dir():
        raise ProcessSandboxUnavailableError(f"candidate cache source is not a directory: {source}")
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise ProcessSandboxUnavailableError(f"candidate cache target is not a directory: {target}")
        return
    target.mkdir(mode=0o555, parents=True)
    for source_child in sorted(source.iterdir(), key=lambda path: path.name):
        target_name = source_child.name
        if source_child.is_symlink():
            resolved_child = source_child.resolve(strict=True)
            if not resolved_child.is_relative_to(approved_root):
                raise ProcessSandboxUnavailableError(
                    f"candidate cache symlink escapes its approved tree: {source_child}"
                )
            source_child = resolved_child
        target_child = target / target_name
        if source_child.is_dir():
            _copy_or_link_readonly_tree(source_child, target_child, approved_root=approved_root)
            continue
        if not source_child.is_file():
            raise ProcessSandboxUnavailableError(f"candidate cache contains unsupported path: {source_child}")
        source_mode = source_child.stat().st_mode & 0o777
        if source_mode & 0o222:
            # A controller-managed canonical cache may be writable.  Copy it
            # into the private candidate view rather than hard-linking it, then
            # make the candidate copy immutable.
            shutil.copy2(source_child, target_child)
            target_child.chmod(0o444)
            continue
        try:
            os.link(source_child, target_child)
        except OSError:
            shutil.copy2(source_child, target_child)
            target_child.chmod(0o444)


def _prepare_candidate_home(*, repository: pathlib.Path, run_directory: pathlib.Path) -> pathlib.Path:
    """Stage only approved data/tokenizer/kernel-cache inputs for this attempt."""
    input_root = pathlib.Path(
        os.environ.get("AUTOPARETO_CANDIDATE_INPUT_ROOT", "/opt/autopareto_candidate_inputs/per_attempts")
    ).resolve()
    input_root.mkdir(mode=0o711, parents=True, exist_ok=True)
    candidate_home = input_root / f"{run_directory.parent.name}-{run_directory.name}"
    if candidate_home.exists():
        if candidate_home.is_symlink() or not candidate_home.is_dir():
            raise ProcessSandboxUnavailableError(f"candidate home is not a directory: {candidate_home}")
    else:
        candidate_home.mkdir(mode=0o555)
    cache_root = candidate_home / ".cache"
    cache_root.mkdir(mode=0o555, exist_ok=True)
    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", DEFAULT_REGISTERED_CACHE)
    ).resolve(strict=True)
    if not registered_cache.is_dir() or registered_cache.is_symlink():
        raise ProcessSandboxUnavailableError("registered cache must be a regular directory")
    # Keep the canonical registered cache layout intact.  These two links are
    # the only compatibility view needed by the legacy prepare.py and the HF
    # resolver; Landlock below grants read access only to their canonical
    # targets, never to their parent filesystem.
    cache_views = {
        cache_root / "autoresearch": registered_cache,
        cache_root / "huggingface": registered_cache / "huggingface",
    }
    for view, target in cache_views.items():
        target = target.resolve(strict=True)
        if not target.is_dir():
            raise ProcessSandboxUnavailableError(f"registered cache view target is not a directory: {target}")
        if view.exists() or view.is_symlink():
            if not view.is_symlink() or view.resolve(strict=True) != target:
                raise ProcessSandboxUnavailableError(f"candidate cache view drifted: {view}")
        else:
            view.symlink_to(target, target_is_directory=True)

    # ``mkdir(..., parents=True)`` applies the host umask to intermediate
    # directories.  Normalize only real directories/files; never chmod a
    # compatibility link because that would follow it into the canonical cache.
    for path in candidate_home.rglob("*"):
        if path.is_symlink():
            expected = cache_views.get(path)
            if expected is None or path.resolve(strict=True) != expected.resolve(strict=True):
                raise ProcessSandboxUnavailableError(f"candidate input path has an unapproved symlink: {path}")
            continue
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file():
            path.chmod(0o444)
    return candidate_home


def _verify_permission_boundary(
    *,
    repository: pathlib.Path,
    process_runtime: pathlib.Path,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    candidate_home: pathlib.Path,
    uid: int,
    protected_state_root: pathlib.Path,
    protected_mirror_root: pathlib.Path | None = None,
) -> None:
    """Fail closed unless the live Unix permission layout matches the policy."""
    def stat_mode(path: pathlib.Path) -> tuple[os.stat_result, int]:
        if path.is_symlink():
            raise ProcessSandboxUnavailableError(f"permission boundary rejects symlink: {path}")
        try:
            result = path.stat()
        except OSError as exc:
            raise ProcessSandboxUnavailableError(f"permission boundary cannot stat {path}: {exc}") from exc
        return result, result.st_mode & 0o777

    def reject_unverified_acl(path: pathlib.Path, label: str) -> None:
        try:
            xattrs = os.listxattr(path, follow_symlinks=False)
        except OSError as exc:
            raise ProcessSandboxUnavailableError(
                f"permission boundary cannot inspect ACL metadata for {label}: {exc}"
            ) from exc
        if any(name in {"system.posix_acl_access", "system.posix_acl_default"} for name in xattrs):
            raise ProcessSandboxUnavailableError(f"permission boundary rejects unverified ACLs on {label}")

    def require_root_private(path: pathlib.Path, label: str) -> None:
        result, mode = stat_mode(path)
        reject_unverified_acl(path, label)
        if result.st_uid != 0 or mode & 0o077:
            raise ProcessSandboxUnavailableError(f"{label} must be root-owned and inaccessible to the candidate")

    def require_cache_readonly(path: pathlib.Path) -> None:
        result, mode = stat_mode(path)
        reject_unverified_acl(path, "registered cache")
        if result.st_uid != 0 or mode & 0o022 or mode & 0o005 != 0o005:
            raise ProcessSandboxUnavailableError("registered cache root must be root-owned, readable, and not writable")
        for child in path.rglob("*"):
            if child.is_symlink():
                target = child.resolve(strict=True)
                if not target.is_relative_to(path):
                    raise ProcessSandboxUnavailableError(
                        f"registered cache symlink escapes the approved cache: {child}"
                    )
                continue
            child_stat, child_mode = stat_mode(child)
            reject_unverified_acl(child, "registered cache entry")
            if child_stat.st_uid != 0 or child_mode & 0o022:
                raise ProcessSandboxUnavailableError(f"registered cache entry is writable: {child}")
            if child.is_dir() and child_mode & 0o005 != 0o005:
                raise ProcessSandboxUnavailableError(f"registered cache directory is not readable: {child}")
            if child.is_file() and not child_mode & 0o004:
                raise ProcessSandboxUnavailableError(f"registered cache file is not readable: {child}")

    context_stat, context_mode = stat_mode(context)
    if context_stat.st_uid != 0 or context_mode & 0o222:
        raise ProcessSandboxUnavailableError("candidate context must be root-owned and read-only")
    runtime_stat, runtime_mode = stat_mode(process_runtime)
    if runtime_stat.st_uid != 0 or runtime_mode & 0o222:
        raise ProcessSandboxUnavailableError("protected process runtime must be root-owned and read-only")
    home_stat, home_mode = stat_mode(candidate_home)
    if home_stat.st_uid != 0 or home_mode & 0o222:
        raise ProcessSandboxUnavailableError("candidate input home must be root-owned and read-only")
    run_stat, run_mode = stat_mode(run_directory)
    if run_stat.st_uid != uid or run_mode & 0o077:
        raise ProcessSandboxUnavailableError("candidate output directory must be private to the candidate UID")

    # The trusted repository is a root-only tree.  The candidate never needs to
    # traverse it: only the explicitly copied process runtime is readable by
    # the candidate.  This matters on the migrated FUSE /workspace mount,
    # where permissions on the old tree are not reliable.
    repository_stat, repository_mode = stat_mode(repository)
    if repository_stat.st_uid != 0 or repository_mode & 0o077:
        raise ProcessSandboxUnavailableError("trusted repository must be root-only")

    require_root_private(protected_state_root, "trusted state root")
    if protected_mirror_root is not None and protected_mirror_root.exists():
        require_root_private(protected_mirror_root, "trusted persistent mirror")

    registered_cache = pathlib.Path(
        os.environ.get("AUTOPARETO_REGISTERED_CACHE", DEFAULT_REGISTERED_CACHE)
    ).resolve(strict=True)
    require_cache_readonly(registered_cache)

    venv = pathlib.Path(
        os.environ.get("AUTOPARETO_STUDY_VENV", "/usr/local")
    ).resolve(strict=True)
    for approved, label in (
        (venv, "approved virtual environment"),
        (process_runtime, "candidate process runtime"),
    ):
        approved_stat, approved_mode = stat_mode(approved)
        if approved_stat.st_uid != 0 or approved_mode & 0o022:
            raise ProcessSandboxUnavailableError(f"{label} must be root-owned and read-only")

    # The staged cache is the only data view visible through HOME.  Its files
    # are checked recursively.  The two compatibility links are explicitly
    # allowlisted and resolved into the registered cache; every other symlink
    # is rejected before the candidate is execed.
    cuda_runtime_cache = candidate_home / ".nv"
    for path in [candidate_home, candidate_home / ".cache"]:
        _stat, mode = stat_mode(path)
        if mode & 0o222:
            raise ProcessSandboxUnavailableError(f"candidate input path is writable: {path}")
    for path in candidate_home.rglob("*"):
        if path == cuda_runtime_cache or path.is_relative_to(cuda_runtime_cache):
            if path.is_symlink():
                raise ProcessSandboxUnavailableError(f"candidate CUDA runtime cache contains a symlink: {path}")
            stat_result, mode = stat_mode(path)
            if stat_result.st_uid != uid or mode & 0o022:
                raise ProcessSandboxUnavailableError(f"candidate CUDA runtime cache is not private: {path}")
            continue
        if path.is_symlink():
            registered_cache = pathlib.Path(
                os.environ.get("AUTOPARETO_REGISTERED_CACHE", DEFAULT_REGISTERED_CACHE)
            ).resolve(strict=True)
            allowed = {
                candidate_home / ".cache" / "autoresearch": registered_cache,
                candidate_home / ".cache" / "huggingface": registered_cache / "huggingface",
            }
            target = path.resolve(strict=True)
            if path not in allowed or target != allowed[path].resolve(strict=True):
                raise ProcessSandboxUnavailableError(f"candidate input symlink escapes approved cache: {path}")
            continue
        stat_result, mode = stat_mode(path)
        if stat_result.st_uid != 0 or mode & 0o222:
            raise ProcessSandboxUnavailableError(f"candidate input path is not immutable: {path}")


def _process_launch(
    *,
    command: list[str],
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
    process_runtime: pathlib.Path,
    study_files: tuple[str, ...],
    protected_files: tuple[str, ...],
    uid: int,
    gid: int,
) -> int:
    if not command:
        raise ProcessSandboxUnavailableError("process sandbox received no child command")
    _prepare_candidate_paths(context, run_directory, uid, gid)
    candidate_home = _prepare_candidate_home(repository=repository, run_directory=run_directory)
    cuda_runtime_cache = candidate_home / ".nv"
    cuda_runtime_cache.mkdir(mode=0o700, exist_ok=True)
    os.chown(cuda_runtime_cache, uid, gid)
    cuda_runtime_cache.chmod(0o700)
    os.environ["AUTOPARETO_CANDIDATE_HOME"] = str(candidate_home)
    protected_state_root = pathlib.Path(
        os.environ.get("AUTOPARETO_TRUSTED_STATE_ROOT", "/var/lib/autopareto/pair_a_state")
    ).resolve()
    protected_mirror_root = pathlib.Path(
        os.environ.get("AUTOPARETO_TRUSTED_MIRROR_ROOT", "/var/lib/autopareto/pair_a_trusted_backup")
    ).resolve()
    process_environment = _sanitized_candidate_environment(run_directory, process_runtime, candidate_home)
    os.environ.clear()
    os.environ.update(process_environment)
    # CUDA initialization may create a driver cache under HOME.  Keep that
    # root-side setup artifact in the per-attempt writable area, then restore
    # the immutable approved input home before execing the candidate.
    candidate_home_environment = process_environment["HOME"]
    os.environ["HOME"] = str(run_directory)
    _preload_cuda_driver()
    os.environ["HOME"] = candidate_home_environment
    _verify_permission_boundary(
        repository=repository,
        process_runtime=process_runtime,
        context=context,
        run_directory=run_directory,
        candidate_home=candidate_home,
        uid=uid,
        protected_state_root=protected_state_root,
        protected_mirror_root=protected_mirror_root,
    )
    _install_network_seccomp()
    _drop_candidate_privileges(uid, gid)
    _exec_candidate_python(command, context=context, process_runtime=process_runtime)
    raise AssertionError("candidate exec returned unexpectedly")


def _process_wrapper_command(
    command: list[str],
    *,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
    study_files: tuple[str, ...],
    protected_files: tuple[str, ...] = (),
) -> list[str]:
    if os.name != "posix" or platform.system() != "Linux":
        raise ProcessSandboxUnavailableError("A40 candidate process boundary requires Linux")
    if shutil.which("setpriv") is None:
        raise ProcessSandboxUnavailableError("setpriv is unavailable")
    uid, gid = _candidate_identity()
    context = context.resolve(strict=True)
    run_directory = run_directory.resolve(strict=True)
    repository = repository.resolve(strict=True)
    process_runtime = _prepare_process_runtime(
        repository=repository,
        study_files=study_files,
        protected_files=protected_files,
    )
    wrapped = [
        sys.executable,
        "-m",
        "study.contracts_2026.candidate_sandbox",
        "--process-launch",
        "--context",
        str(context),
        "--run-directory",
        str(run_directory),
        "--repository",
        str(repository),
        "--process-runtime",
        str(process_runtime),
        "--uid",
        str(uid),
        "--gid",
        str(gid),
    ]
    for value in study_files:
        wrapped.extend(("--study-file", value))
    for value in protected_files:
        wrapped.extend(("--protected-file", value))
    wrapped.extend(("--", *command))
    return wrapped


def _replace_path(value: str, source: pathlib.Path, target: str) -> str:
    source_text = str(source)
    if value == source_text:
        return target
    prefix = source_text.rstrip("/") + "/"
    if value.startswith(prefix):
        return target.rstrip("/") + "/" + value[len(prefix):]
    return value


def wrap_worker_command(
    command: list[str],
    *,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
) -> list[str]:
    """Return a permission/seccomp/setpriv candidate command."""
    return _process_wrapper_command(
        command,
        context=context,
        run_directory=run_directory,
        repository=repository,
        study_files=("__init__.py", "candidate_process.py"),
    )


def wrap_inference_command(
    command: list[str],
    *,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
) -> list[str]:
    """Wrap fresh checkpoint inference in the same process boundary as training."""
    return _process_wrapper_command(
        command,
        context=context,
        run_directory=run_directory,
        repository=repository,
        study_files=("__init__.py", "benchmark_policy.py", "correctness_policy.py", "inference_benchmark.py"),
        protected_files=("inference_adapter.py", "phase1_runtime.py"),
    )


def wrap_model_host_command(
    command: list[str],
    *,
    context: pathlib.Path,
    run_directory: pathlib.Path,
    repository: pathlib.Path,
) -> list[str]:
    """Return a restricted command for the untrusted inference model host.

    The trusted evaluator does not enter this boundary.  Only the model host,
    candidate definitions, approved runtime, checkpoint, and anonymous Unix
    IPC descriptor are placed in the restricted process.
    """
    return _process_wrapper_command(
        command,
        context=context,
        run_directory=run_directory,
        repository=repository,
        study_files=("__init__.py", "inference_model_host.py", "inference_ipc.py"),
        protected_files=("inference_adapter.py",),
    )


def _process_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-launch", action="store_true")
    parser.add_argument("--context", type=pathlib.Path, required=True)
    parser.add_argument("--run-directory", type=pathlib.Path, required=True)
    parser.add_argument("--repository", type=pathlib.Path, required=True)
    parser.add_argument("--process-runtime", type=pathlib.Path, required=True)
    parser.add_argument("--uid", type=int, required=True)
    parser.add_argument("--gid", type=int, required=True)
    parser.add_argument("--study-file", action="append", default=[])
    parser.add_argument("--protected-file", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _process_parser().parse_args(argv)
    if not args.process_launch or not args.command or args.command[0] != "--":
        raise SystemExit("candidate_sandbox is only an internal process launcher")
    return _process_launch(
        command=args.command[1:],
        context=args.context,
        run_directory=args.run_directory,
        repository=args.repository,
        process_runtime=args.process_runtime,
        study_files=tuple(args.study_file),
        protected_files=tuple(args.protected_file),
        uid=args.uid,
        gid=args.gid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
