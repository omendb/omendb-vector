# ADR 0003: Every feature earns its place

Date: 2026-09-02. Decided by Nick.

We maintain what we add. Nothing enters the design to "support" a paper,
a method, or a hypothetical user — only SOTA standing plus broad,
concrete user need justifies the maintenance burden. Novel work must beat
measured SOTA on a reproducible harness, not merely differ from it.

First application: MUVERA removed entirely (not demoted to a track). It
fails on arbitrary user models without STE retraining, and TACHIOM covers
the approximate-multivector future with no model demands. If a model
co-design story ever exists, MUVERA can be re-proposed against this bar.
