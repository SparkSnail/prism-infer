import json

import pytest

from prism_infer.server import process_identity


def _identity():
    return {
        "schema_version": 1,
        "component": "worker",
        "instance_id": "d0",
        "pod_uid": "pod-d0",
        "process_generation": "boot-d0",
        "instance_epoch": "pod-d0:boot-d0",
        "app_pid": 8,
        "process_start_ticks": 12345,
    }


def test_worker_process_identity_publish_and_pidfd_signal(tmp_path, monkeypatch):
    path = tmp_path / "identity.json"
    calls = []
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 8)
    monkeypatch.setattr(process_identity, "process_start_ticks", lambda pid: 12345)
    published = process_identity.publish_process_identity(
        path,
        component="worker",
        instance_id="d0",
        pod_uid="pod-d0",
        process_generation="boot-d0",
    )
    assert published == _identity()
    assert json.loads(path.read_text(encoding="utf-8")) == _identity()

    monkeypatch.setattr(
        process_identity.os, "pidfd_open",
        lambda pid, flags: calls.append(("open", pid)) or 17,
        raising=False,
    )
    monkeypatch.setattr(
        process_identity.signal, "pidfd_send_signal",
        lambda fd, sig: calls.append(("signal", fd, int(sig))),
        raising=False,
    )
    monkeypatch.setattr(process_identity.os, "close", lambda fd: None)
    process_identity.signal_exact_process(
        path,
        component="worker",
        instance_id="d0",
        pod_uid="pod-d0",
        process_generation="boot-d0",
    )
    assert calls == [("open", 8), ("signal", 17, 9)]


def test_worker_pidfd_syscall_fallback_when_python_omits_wrappers(monkeypatch):
    calls = []
    monkeypatch.delattr(process_identity.os, "pidfd_open", raising=False)
    monkeypatch.delattr(
        process_identity.signal, "pidfd_send_signal", raising=False
    )
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 21)
    monkeypatch.setattr(
        process_identity.os, "close", lambda fd: calls.append(("close", fd))
    )

    def syscall(number, *args):
        values = tuple(getattr(argument, "value", argument) for argument in args)
        calls.append(("syscall", number, values))
        return 17 if number == process_identity._PIDFD_OPEN_SYSCALL else 0

    monkeypatch.setattr(process_identity, "_linux_syscall", syscall)

    process_identity.assert_pidfd_support()

    assert calls == [
        ("syscall", process_identity._PIDFD_OPEN_SYSCALL, (21, 0)),
        (
            "syscall",
            process_identity._PIDFD_SEND_SIGNAL_SYSCALL,
            (17, 0, None, 0),
        ),
        ("close", 17),
    ]


def test_worker_process_identity_rejects_pid1_and_wrong_epoch(
    tmp_path, monkeypatch,
):
    path = tmp_path / "identity.json"
    monkeypatch.setattr(process_identity.os, "getpid", lambda: 1)
    with pytest.raises(RuntimeError, match="app_pid > 1"):
        process_identity.publish_process_identity(
            path,
            component="worker",
            instance_id="d0",
            pod_uid="pod-d0",
            process_generation="boot-d0",
        )

    path.write_text(json.dumps(_identity()), encoding="utf-8")
    signalled = []
    monkeypatch.setattr(
        process_identity.os, "pidfd_open", lambda pid, flags: 17,
        raising=False,
    )
    monkeypatch.setattr(process_identity.os, "close", lambda fd: None)
    monkeypatch.setattr(process_identity, "process_start_ticks", lambda pid: 12345)
    monkeypatch.setattr(
        process_identity.signal, "pidfd_send_signal",
        lambda fd, sig: signalled.append((fd, sig)),
        raising=False,
    )
    with pytest.raises(RuntimeError, match="process_generation"):
        process_identity.signal_exact_process(
            path,
            component="worker",
            instance_id="d0",
            pod_uid="pod-d0",
            process_generation="boot-other",
        )
    assert signalled == []


def test_worker_process_identity_rejects_pidfile_replacement_after_pidfd_open(
    tmp_path, monkeypatch,
):
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(_identity()), encoding="utf-8")
    signalled = []

    def open_and_replace(pid, flags):
        replacement = _identity()
        replacement["instance_epoch"] = "pod-d0:boot-replacement"
        path.write_text(json.dumps(replacement), encoding="utf-8")
        return 17

    monkeypatch.setattr(
        process_identity.os, "pidfd_open", open_and_replace, raising=False,
    )
    monkeypatch.setattr(process_identity.os, "close", lambda fd: None)
    monkeypatch.setattr(
        process_identity.signal, "pidfd_send_signal",
        lambda fd, sig: signalled.append((fd, sig)),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="changed after pidfd open"):
        process_identity.signal_exact_process(
            path,
            component="worker",
            instance_id="d0",
            pod_uid="pod-d0",
            process_generation="boot-d0",
        )

    assert signalled == []
