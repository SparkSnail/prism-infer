"""Worker startup permits and pod-incarnation fail-stop records."""

import hashlib
import json
import os
from pathlib import Path
import re
import time
from collections.abc import Callable, Mapping


STARTUP_PERMIT_SCHEMA = "prism.week12.worker-startup-permit/v1"
INCARNATION_RECORD_SCHEMA = "prism.week12.worker-incarnation/v1"
EXPECTED_MEMBERS = frozenset({"p0", "p1", "d0", "d1"})
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_PERMIT_FIELDS = frozenset({
    "schema_version",
    "issuance_mode",
    "permit_id",
    "topology_generation",
    "members",
    "canonical_digest",
})
_RECORD_FIELDS = frozenset({
    "schema_version",
    "state",
    "instance_id",
    "topology_generation",
    "pod_uid",
    "process_generation",
    "permit_id",
    "permit_digest",
    "watchdog",
    "canonical_digest",
})
_ENDPOINT_REF_FIELDS = frozenset({
    "topology_generation",
    "owner_generation",
    "operation_seq",
    "target_instance",
    "target_worker_epoch",
    "operation_id",
    "payload_digest",
})
_WATCHDOG_FIELDS = frozenset({
    "pair_id",
    "operation_id",
    "endpoint_ref",
    "reason",
    "watchdog_timeout_s",
})


class StartupPermitWaitCancelled(RuntimeError):
    """The caller cancelled bootstrap before the permit became ready."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_without(
    value: Mapping[str, object], field: str
) -> str:
    canonical = {
        key: item for key, item in value.items() if key != field
    }
    return "sha256:" + hashlib.sha256(_canonical_json(canonical)).hexdigest()


def validate_startup_permit(
    value: object,
    *,
    instance_id: str,
    pod_uid: str,
    topology_generation: str,
) -> dict[str, object]:
    """Validate one projected-file snapshot for all four worker pods."""

    if not isinstance(value, Mapping):
        raise ValueError("startup permit must be an object")
    permit = dict(value)
    if set(permit) != _PERMIT_FIELDS:
        raise ValueError("startup permit fields are not exact")
    if permit["schema_version"] != STARTUP_PERMIT_SCHEMA:
        raise ValueError("startup permit schema_version mismatch")
    if permit["issuance_mode"] not in {"INIT", "RESTART"}:
        raise ValueError("startup permit issuance_mode is invalid")
    for field in ("permit_id", "topology_generation"):
        if not isinstance(permit[field], str) or not permit[field]:
            raise ValueError(f"startup permit {field} must be a string")
    if permit["topology_generation"] != topology_generation:
        raise ValueError("startup permit topology_generation mismatch")

    members = permit["members"]
    if not isinstance(members, Mapping):
        raise ValueError("startup permit members must be an object")
    members = dict(members)
    if set(members) != EXPECTED_MEMBERS:
        raise ValueError("startup permit requires exact four members")
    if any(not isinstance(value, str) or not value for value in members.values()):
        raise ValueError("startup permit member Pod UIDs must be strings")
    if len(set(members.values())) != len(EXPECTED_MEMBERS):
        raise ValueError("startup permit member Pod UIDs must be unique")
    if instance_id not in EXPECTED_MEMBERS:
        raise ValueError("worker instance is not a fixed 2P2D member")
    if members[instance_id] != pod_uid:
        raise ValueError("startup permit does not authorize this Pod UID")

    digest = permit["canonical_digest"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("startup permit canonical_digest is invalid")
    if digest != _digest_without(permit, "canonical_digest"):
        raise ValueError("startup permit canonical_digest mismatch")
    permit["members"] = members
    return permit


def read_startup_permit(
    path: str | Path,
    *,
    instance_id: str,
    pod_uid: str,
    topology_generation: str,
) -> dict[str, object]:
    """Read one symlink target so fields cannot span ConfigMap revisions."""

    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("utf-8"))
    return validate_startup_permit(
        value,
        instance_id=instance_id,
        pod_uid=pod_uid,
        topology_generation=topology_generation,
    )


def wait_for_startup_permit(
    path: str | Path,
    *,
    instance_id: str,
    pod_uid: str,
    topology_generation: str,
    poll_interval_s: float = 0.2,
    cancelled: Callable[[], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object]:
    """Wait for an exact permit without initializing model or distributed state."""

    if poll_interval_s <= 0:
        raise ValueError("startup permit poll interval must be positive")
    while True:
        if cancelled is not None and cancelled():
            raise StartupPermitWaitCancelled(
                "startup permit wait was cancelled"
            )
        try:
            return read_startup_permit(
                path,
                instance_id=instance_id,
                pod_uid=pod_uid,
                topology_generation=topology_generation,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            sleep(poll_interval_s)


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("incarnation record write made no progress")
        offset += written


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class PodIncarnationLifecycle:
    """Persist create-once ACTIVE and idempotent FAIL_STOP state per pod."""

    def __init__(
        self,
        path: str | Path,
        *,
        instance_id: str,
        pod_uid: str,
        topology_generation: str,
        process_generation: str,
    ) -> None:
        self.path = Path(path)
        self.instance_id = instance_id
        self.pod_uid = pod_uid
        self.topology_generation = topology_generation
        self.process_generation = process_generation

    @staticmethod
    def _seal(record: dict[str, object]) -> dict[str, object]:
        sealed = {**record, "canonical_digest": ""}
        sealed["canonical_digest"] = _digest_without(
            sealed, "canonical_digest"
        )
        return sealed

    @staticmethod
    def _validate(record: object) -> dict[str, object]:
        if not isinstance(record, Mapping):
            raise ValueError("incarnation record must be an object")
        value = dict(record)
        if set(value) != _RECORD_FIELDS:
            raise ValueError("incarnation record fields are not exact")
        if value["schema_version"] != INCARNATION_RECORD_SCHEMA:
            raise ValueError("incarnation record schema mismatch")
        if value["state"] not in {"ACTIVE", "FAIL_STOP"}:
            raise ValueError("incarnation record state is invalid")
        for field in (
            "instance_id",
            "topology_generation",
            "pod_uid",
            "process_generation",
            "permit_id",
            "permit_digest",
        ):
            if not isinstance(value[field], str) or not value[field]:
                raise ValueError(f"incarnation record {field} is invalid")
        if not _SHA256.fullmatch(str(value["permit_digest"])):
            raise ValueError("incarnation record permit_digest is invalid")
        digest = value["canonical_digest"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise ValueError("incarnation record canonical_digest is invalid")
        if digest != _digest_without(value, "canonical_digest"):
            raise ValueError("incarnation record canonical_digest mismatch")
        if value["state"] == "ACTIVE":
            if value["watchdog"] is not None:
                raise ValueError("ACTIVE incarnation record cannot contain watchdog")
        else:
            PodIncarnationLifecycle._validate_watchdog(value["watchdog"])
        return value

    @staticmethod
    def _validate_watchdog(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise ValueError("FAIL_STOP watchdog must be an object")
        watchdog = dict(value)
        if set(watchdog) != _WATCHDOG_FIELDS:
            raise ValueError("FAIL_STOP watchdog fields are not exact")
        for field in ("pair_id", "operation_id", "reason"):
            if not isinstance(watchdog[field], str) or not watchdog[field]:
                raise ValueError(f"FAIL_STOP watchdog {field} is invalid")
        timeout = watchdog["watchdog_timeout_s"]
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError("FAIL_STOP watchdog timeout is invalid")
        ref = watchdog["endpoint_ref"]
        if not isinstance(ref, Mapping) or set(ref) != _ENDPOINT_REF_FIELDS:
            raise ValueError("FAIL_STOP endpoint_ref fields are not exact")
        ref = dict(ref)
        seq = ref["operation_seq"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq <= 0:
            raise ValueError("FAIL_STOP endpoint_ref operation_seq is invalid")
        for field in _ENDPOINT_REF_FIELDS - {"operation_seq"}:
            if not isinstance(ref[field], str) or not ref[field]:
                raise ValueError(
                    f"FAIL_STOP endpoint_ref {field} is invalid"
                )
        watchdog["endpoint_ref"] = ref
        return watchdog

    def read(self) -> dict[str, object]:
        value = json.loads(self.path.read_text(encoding="utf-8"))
        return self._validate(value)

    def _matches_identity(self, record: Mapping[str, object]) -> bool:
        return (
            record["instance_id"] == self.instance_id
            and record["pod_uid"] == self.pod_uid
            and record["topology_generation"] == self.topology_generation
            and record["process_generation"] == self.process_generation
        )

    def create_active(self, permit: Mapping[str, object]) -> bool:
        """Claim initialization with O_EXCL; existing or corrupt state holds."""

        permit = validate_startup_permit(
            permit,
            instance_id=self.instance_id,
            pod_uid=self.pod_uid,
            topology_generation=self.topology_generation,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = self._seal({
            "schema_version": INCARNATION_RECORD_SCHEMA,
            "state": "ACTIVE",
            "instance_id": self.instance_id,
            "topology_generation": self.topology_generation,
            "pod_uid": self.pod_uid,
            "process_generation": self.process_generation,
            "permit_id": permit["permit_id"],
            "permit_digest": permit["canonical_digest"],
            "watchdog": None,
        })
        data = _canonical_json(record)
        try:
            fd = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            return False
        try:
            _write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_directory(self.path.parent)
        return True

    def _atomic_replace(self, record: Mapping[str, object]) -> None:
        temporary = self.path.with_name(
            f".{self.path.name}.{os.getpid()}.tmp"
        )
        data = _canonical_json(record)
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            try:
                _write_all(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def latch_fail_stop(
        self, evidence: Mapping[str, object]
    ) -> dict[str, object]:
        """Promote ACTIVE atomically; replay identical evidence and reject conflicts."""

        watchdog = self._validate_watchdog({
            "pair_id": evidence.get("pair_id"),
            "operation_id": evidence.get("operation_id"),
            "endpoint_ref": evidence.get("endpoint_ref"),
            "reason": evidence.get("reason"),
            "watchdog_timeout_s": evidence.get("watchdog_timeout_s"),
        })
        ref = watchdog["endpoint_ref"]
        if (
            ref["target_instance"] != self.instance_id
            or ref["target_worker_epoch"]
            != f"{self.pod_uid}:{self.process_generation}"
            or ref["topology_generation"] != self.topology_generation
            or ref["operation_id"] != watchdog["operation_id"]
        ):
            raise ValueError("FAIL_STOP endpoint_ref identity mismatch")

        current = self.read()
        if not self._matches_identity(current):
            raise ValueError("incarnation record identity mismatch")
        if current["state"] == "FAIL_STOP":
            if current["watchdog"] != watchdog:
                raise ValueError("FAIL_STOP evidence conflicts with durable record")
            _fsync_directory(self.path.parent)
            return current

        latched = self._seal({
            **current,
            "state": "FAIL_STOP",
            "watchdog": watchdog,
        })
        self._atomic_replace(latched)
        return latched
