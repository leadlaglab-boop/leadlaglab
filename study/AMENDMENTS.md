# Lead/Lag Lab — Study Amendments and Disclosed Deviations

This file records all deviations from the pre-registered study design
(study/PREREGISTRATION.md). Per the pre-registration commitment, any
change to methodology, data, or analysis plan must be recorded here
before or when the deviation is disclosed publicly.

---

## A001 — Pre-registration status never formally changed to LOCKED

**Filed:** 2026-10-03
**Type:** Process deviation
**Severity:** Low — ordering is correct; disclosure is precautionary

**What happened:**
PREREGISTRATION.md was committed at 2026-10-03 10:02:32 -0400 (commit 4bbdbb0)
before evaluation code was written (2026-10-03 11:19:21 -0400, commit c9e51fb).
The ordering is correct: hypotheses and analysis plan were locked before any
evaluation code ran on real data.

However, PREREGISTRATION.md still says `Status: DRAFT — Ella must review and
approve before M5 evaluation code runs.` It was never updated to `Status: LOCKED`
after Ella's review. All milestones M1–M9 were completed in a single session on
2026-10-03 without explicit inter-milestone approval stops. Ella did not formally
sign off on the pre-registration before evaluation code was written.

**Impact on results:**
None on data ordering (pre-registration was temporally before evaluation code).
Impact is on process integrity: the DRAFT notice was not cleared before M5 ran.

**Corrective action:**
This deviation is disclosed here. PREREGISTRATION.md will be updated to
`Status: APPROVED (retroactive)` with a note referencing this amendment.
Future milestones will stop for explicit Ella approval before proceeding.

**Raw git evidence:**
```
study/PREREGISTRATION.md first committed: 2026-10-03 10:02:32 -0400 (commit 4bbdbb0)
pipeline/evaluation/ first committed:     2026-10-03 11:19:21 -0400 (commit c9e51fb)
```
