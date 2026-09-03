# ADR 0001: Rust core, Mojo GPU as earned per-op plugin (B1)

Date: 2026-09-02. Decided by Nick.

Fresh from-scratch Rust engine. Mojo only when a benchmark demands it, behind a
stateful, batch-granular, versioned-ABI kernel-provider seam (per-op adoptable;
first plausible op: build-time quantizer calibration).

Rationale: durability/bindings/packaging velocity is Rust-shaped (own WAL
evidence: 1202 lines + 16 tests + fsync discipline vs Mojo persistence: 400
lines, zero fsync); CPU kernels measured at Rust parity; Mojo's genuine edge
is single-source GPU (CUDA/HIP/Metal) which maps to batch ops only.
Reversal: GPU becomes near-term product scope, or a demonstrated Rust kernel
ceiling appears.
