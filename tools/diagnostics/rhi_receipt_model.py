"""CPU-only model of a proposed controlled-process RHI diagnostic contract.

This does not observe Vulkan or authenticate an RHI/SlangPy runtime. Objects in
this module are simulations and cannot provide allocation or device evidence.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    generation: int
    buffer: int
    memory: int
    offset: int
    size: int
    allocation_size: int


class ReceiptModel:
    def __init__(self, uuid: bytes, provider: int):
        if len(uuid) != 16 or not any(uuid) or provider <= 0:
            raise ValueError('unknown physical UUID or instance dispatch provider')
        self.uuid = bytes(uuid)
        self.provider = provider
        self._next = 1
        self._live: dict[int, Observation] = {}
        self._closed = False

    def register(self, *, buffer: int, memory: int, offset: int, size: int,
                 allocation_size: int, ordinary: bool, external: bool = False,
                 sparse: bool = False, rebound: bool = False) -> Observation:
        if self._closed:
            raise ValueError('owner closed')
        if not ordinary or external or sparse or rebound:
            raise ValueError('unsupported buffer path')
        if min(buffer, memory, size, allocation_size) <= 0 or offset != 0 or size > allocation_size:
            raise ValueError('invalid allocation observation')
        if buffer in self._live or any(x.memory == memory for x in self._live.values()):
            raise ValueError('duplicate or aliased live resource')
        observation = Observation(self._next, buffer, memory, offset, size, allocation_size)
        self._next += 1
        self._live[buffer] = observation
        return observation

    def check(self, observation: Observation) -> None:
        if self._closed or self._live.get(observation.buffer) != observation:
            raise ValueError('stale or untracked observation')

    def release(self, observation: Observation) -> None:
        self.check(observation)
        del self._live[observation.buffer]

    def close(self) -> None:
        self._closed = True
        self._live.clear()
