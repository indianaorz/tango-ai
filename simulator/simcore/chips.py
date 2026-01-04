# mmbn_sim/simcore/chips.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from .actions import EventSpec

ChipId = int
ChipKey = str


# -----------------------------------------------------------------------------
# Generic chip program
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Wait:
    cust: int


@dataclass(frozen=True, slots=True)
class Emit:
    event: EventSpec


ChipOp = Union[Wait, Emit]


@dataclass(frozen=True, slots=True)
class ChipSpec:
    chip_id: ChipId        # boundary identity (replays/protocol)
    chip_key: ChipKey      # internal stable identifier (authoring/reference)
    name: str
    lock_cust: int
    program: Tuple[ChipOp, ...]


class ChipSpecError(ValueError):
    pass


def ev(event_type: str, **payload: object) -> EventSpec:
    return EventSpec(str(event_type), dict(payload))


def _compile_program(program: Tuple[ChipOp, ...]) -> Tuple[Tuple[int, EventSpec], ...]:
    t = 0
    out: List[Tuple[int, EventSpec]] = []
    for op in program:
        if isinstance(op, Wait):
            dt = int(op.cust)
            if dt < 0:
                raise ChipSpecError(f"Wait.cust must be >= 0 (got {dt})")
            t += dt
        else:  # Emit
            out.append((t, op.event))
    return tuple(out)


def _validate_spec(spec: ChipSpec) -> None:
    if int(spec.chip_id) <= 0:
        raise ChipSpecError(f"chip_id must be > 0 (got {spec.chip_id})")
    if not spec.chip_key or not spec.chip_key.strip():
        raise ChipSpecError(f"chip_id {spec.chip_id}: chip_key must be non-empty")
    if not spec.name or not spec.name.strip():
        raise ChipSpecError(f"chip_id {spec.chip_id}: name must be non-empty")
    if int(spec.lock_cust) < 0:
        raise ChipSpecError(f"chip_id {spec.chip_id}: lock_cust must be >= 0")

    tl = _compile_program(spec.program)
    if tl:
        last_off = tl[-1][0]
        # If later you need post-lock lingering effects, relax/remove this invariant.
        if last_off > int(spec.lock_cust):
            raise ChipSpecError(
                f"chip_id {spec.chip_id}: last event offset ({last_off}) exceeds lock_cust ({spec.lock_cust})"
            )


class ChipRegistry:
    def __init__(self) -> None:
        self._by_id: Dict[ChipId, ChipSpec] = {}
        self._by_key: Dict[ChipKey, ChipSpec] = {}
        self._compiled: Dict[ChipId, Tuple[Tuple[int, EventSpec], ...]] = {}

    def register(self, spec: ChipSpec) -> None:
        _validate_spec(spec)
        cid = int(spec.chip_id)
        key = str(spec.chip_key)

        if cid in self._by_id:
            raise ChipSpecError(f"duplicate chip_id: {cid}")
        if key in self._by_key:
            raise ChipSpecError(f"duplicate chip_key: {key}")

        self._by_id[cid] = spec
        self._by_key[key] = spec

    def get_by_id(self, chip_id: int) -> ChipSpec:
        cid = int(chip_id)
        try:
            return self._by_id[cid]
        except KeyError as e:
            raise ChipSpecError(f"unknown chip id: {cid}") from e

    def get_by_key(self, chip_key: str) -> ChipSpec:
        key = str(chip_key)
        try:
            return self._by_key[key]
        except KeyError as e:
            raise ChipSpecError(f"unknown chip key: {key}") from e

    def compiled_timeline(self, chip_id: int) -> Tuple[Tuple[int, EventSpec], ...]:
        cid = int(chip_id)
        if cid in self._compiled:
            return self._compiled[cid]
        spec = self.get_by_id(cid)
        tl = _compile_program(spec.program)
        self._compiled[cid] = tl
        return tl

    def all_specs(self) -> Mapping[ChipId, ChipSpec]:
        return dict(self._by_id)


_REGISTRY = ChipRegistry()


# -----------------------------------------------------------------------------
# Chip definitions (balance is embedded, no free-floating constants)
# -----------------------------------------------------------------------------
_REGISTRY.register(
    ChipSpec(
        chip_id=1,
        chip_key="cannon",
        name="Cannon",
        lock_cust=4,
        program=(
            Wait(2),
            Emit(
                ev(
                    "CHIP_RAY",
                    chip_id=1,
                    dmg=40,
                    kind="cannon",
                    push_step=0,
                    hitstun_cust=0,
                )
            ),
        ),
    )
)

_REGISTRY.register(
    ChipSpec(
        chip_id=4,
        chip_key="airshot",
        name="AirShot",
        lock_cust=3,
        program=(
            Wait(1),
            Emit(
                ev(
                    "CHIP_RAY",
                    chip_id=4,
                    dmg=20,
                    kind="airshot",
                    push_step=1,
                    hitstun_cust=2,
                )
            ),
        ),
    )
)

# AreaGrab (BN6): treat as time-freeze, but in this sim:
# - no lock
# - effect happens "next cust" after use
_REGISTRY.register(
    ChipSpec(
        chip_id=163,
        chip_key="areagrab",
        name="AreaGrab",
        lock_cust=1,  # must be >= last event offset; keep it minimal but valid
        program=(
            Wait(1),
            Emit(
                ev(
                    "AREA_GRAB",
                    chip_id=163,
                )
            ),
        ),
    )
)

# Barrier family (BN6)
# IMPORTANT: Barrier should apply on the same cust it is used.
# We implement that by emitting at offset 0.
# Barrier absorbs hits; no overflow to HP (implemented in GameState damage logic).
_REGISTRY.register(
    ChipSpec(
        chip_id=178,
        chip_key="barrier",
        name="Barrier",
        lock_cust=1,
        program=(
            Emit(ev("BARRIER", chip_id=178, hp=10)),
        ),
    )
)

_REGISTRY.register(
    ChipSpec(
        chip_id=179,
        chip_key="barrier100",
        name="Barrier100",
        lock_cust=1,
        program=(
            Emit(ev("BARRIER", chip_id=179, hp=100)),
        ),
    )
)

_REGISTRY.register(
    ChipSpec(
        chip_id=180,
        chip_key="barrier200",
        name="Barrier200",
        lock_cust=1,
        program=(
            Emit(ev("BARRIER", chip_id=180, hp=200)),
        ),
    )
)

# Back-compat: old name
CHIPS: Dict[ChipId, ChipSpec] = dict(_REGISTRY.all_specs())


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def chip_spec(chip_id: int) -> ChipSpec:
    return _REGISTRY.get_by_id(chip_id)


def chip_spec_by_key(chip_key: str) -> ChipSpec:
    return _REGISTRY.get_by_key(chip_key)


def chip_timeline(chip_id: int) -> Tuple[Tuple[int, EventSpec], ...]:
    return _REGISTRY.compiled_timeline(chip_id)


def chip_name(chip_id: int) -> str:
    spec = CHIPS.get(int(chip_id))
    return spec.name if spec is not None else f"UnknownChip({int(chip_id)})"


def safe_chip_summary(hand: Sequence[int], max_items: int = 8) -> List[int]:
    n = max(0, int(max_items))
    return [int(x) for x in hand[:n]]
