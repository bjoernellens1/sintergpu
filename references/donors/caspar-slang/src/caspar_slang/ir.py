from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AccessKind(str, Enum):
    READ = "read"
    WRITE = "write"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class Buffer:
    name: str
    access: AccessKind
    num_alloc_param: bool = True
    scalar_type: str = "float"
    components: int = 1
    index_name: str | None = None
    unique_indices: bool = False
    layout: tuple[tuple[int, ...], ...] = ()


@dataclass
class KernelIR:
    name: str
    buffers: list[Buffer]
    registers: list[str]
    code_lines: list[str]
    block_size: int = 256
    scalar_type: str = "float"
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("kernel name must not be empty")
        if self.scalar_type != "float":
            raise NotImplementedError("Milestone 0 supports float32 only")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        for buffer in self.buffers:
            if buffer.scalar_type not in {"float", "uint"}:
                raise NotImplementedError("Buffers support float32 and uint32 only")
            if buffer.components <= 0:
                raise ValueError("buffer components must be positive")
        names = [b.name for b in self.buffers]
        if len(names) != len(set(names)):
            raise ValueError("buffer names must be unique")
