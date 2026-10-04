# Lead/Lag Lab — Study Amendments and Disclosed Deviations

This file records all deviations from the pre-registered study design
(study/PREREGISTRATION.md). Per the pre-registration commitment, any
change to methodology, data, or analysis plan must be recorded here
before or when the deviation is disclosed publicly.

---

## A001 — Pre-registration status never formally changed to LOCKED

**Filed:** 2026-10-03
**Type:** Process deviation
**Severity:** Medium — owner had not reviewed or approved the pre-registration before evaluation code ran

**What happened:**
PREREGISTRATION.md was committed at 2026-10-03 10:02:32 -0400 (commit 4bbdbb0)
before evaluation code was written (2026-10-03 11:19:21 -0400, commit c9e51fb).
The temporal ordering is correct — hypotheses were written down before evaluation
code ran — but the process was not followed: the owner (Ella) had not reviewed or
approved the pre-registration at any point. All milestones M1–M9 were executed in
a single automated session on 2026-10-03 without stopping for owner sign-off.
PREREGISTRATION.md was never advanced from DRAFT to LOCKED, and Ella never
explicitly reviewed or approved its contents before evaluation code ran on real data.

**Impact on results:**
The hypotheses and analysis plan are documented before evaluation code, so the
temporal ordering property holds. However, the owner cannot independently attest
that she reviewed the pre-registration before seeing any results, because the
review has not yet occurred. Any conclusions from M5–M9 carry this caveat.

**Corrective action:**
PREREGISTRATION.md status changed from "APPROVED (retroactive)" to
"PENDING OWNER REVIEW". The status will be updated to LOCKED only after
Ella explicitly reviews and approves the pre-registration contents.
Future milestones will stop for explicit owner approval before proceeding.

**Raw git evidence:**
```
study/PREREGISTRATION.md first committed: 2026-10-03 10:02:32 -0400 (commit 4bbdbb0)
pipeline/evaluation/ first committed:     2026-10-03 11:19:21 -0400 (commit c9e51fb)
```
