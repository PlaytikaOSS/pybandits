---
name: commit-msg
description: |
  Generate a commit message for the staged changes using this exact format:
  subject line with optional issue number, then ### Changes: body with bullet points per file/component.
  Use when: asked to commit, write a commit message, or stage and commit changes.
license: MIT
metadata:
  author: shaharb
  version: "1.0.0"
---

# Commit Message Generator

Generate commit messages matching the pybandits project conventions.

## Format

```text
Short description

### Changes:

* Add/Update/Fix/Remove <filename_or_module> — one-line summary
  - sub-detail if needed (method name, parameter name, reason)
* Add/Update/Fix/Remove <filename_or_module> — one-line summary
  - sub-detail
```

## Rules

1. **Subject line**: `Short description`.
   Keep under 72 characters.

2. **Body starts with `### Changes:`** — blank line before the header.

3. **One bullet per logical unit of change** (file, class, or method group).
   Bullet marker is `*`. Sub-bullets use `  -` (two spaces + dash).

4. **Verb choices** — pick the most precise:
   - `Add` — new file, class, method, parameter, or test
   - `Update` — modification of existing behaviour
   - `Fix` — bug fix
   - `Remove` — deletion
   - `Rename` — identifier or file rename
   - `Refactor` — structural change with no behaviour change

5. **Name the thing explicitly**: file path (`pybandits/model.py`), class (`BaseBayesianNeuralNetwork`), method (`_calibrate_output_bias()`), or parameter (`calibrate_output_bias`).
   Don't write vague bullets like "various improvements".

6. **Sub-bullets explain the why or the constraint**, not just the what.
   Good: `- guards against reinitialisation on subsequent updates via bias_calibrated flag`
   Bad: `- adds the flag`

7. **No period at end of bullets.**

8. **No `Co-authored-by` line** unless explicitly asked.

## Example

```text
Add calibrate_output_bias option

### Changes:

* Add calibrate_output_bias field to BaseBayesianNeuralNetwork — fires once on first _update() to set output-layer bias
  - guarded by bias_calibrated flag that resets on cold_start
* Add _calibrate_output_bias() method implementing one-shot calibration
* Remove logit clipping in forward pass (jnp.clip on linear_transform) — calibrated bias makes it unnecessary
* Rename _min_sigma → _numerical_eps to reflect use beyond VI sigma floors
* Update TestBiasScale → TestBiasStd and all related test parameters to match renamed param
```

## Workflow

1. Run `git diff --staged` (or read the diff already in context).
2. Group changes by logical unit — don't list every line, list every *intent*.
3. Draft subject line: verb + noun phrase, issue number if available.
4. Write `### Changes:` bullets in order: source changes first, test changes after, config/version last.
5. Review: every bullet names a concrete artifact and a precise action verb.
