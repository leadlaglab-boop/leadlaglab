---
name: Pipeline failure
about: Automatically opened when the daily ingest fails
title: "[Pipeline failure] YYYY-MM-DD"
labels: pipeline-failure
assignees: ""
---

## Failure details

**Run URL:** _(filled by workflow)_

**Steps completed before failure:**

**Error message:**

## Checklist

- [ ] Check the Actions log for the error
- [ ] Identify which source failed
- [ ] Fix and re-run manually if needed: `gh workflow run ingest.yml -f date_override=YYYY-MM-DD`
- [ ] Close this issue once resolved
