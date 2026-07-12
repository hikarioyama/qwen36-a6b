# Corpus Judge v1: Japanese verifiable data rubric

## Purpose and decision rule

Use this rubric after deterministic validation has passed.  Assess whether the
Japanese prompt/response pair is high-quality training signal for natural,
useful Japanese assistance, not whether it merely satisfies a regex or count.

Return exactly `accept`, `reject`, or `borderline`, plus one short reason on
one line.  Judge strictly: **if unsure, reject**.  Use `borderline` only when
the pair is otherwise clearly useful and one small, concrete uncertainty is
left for audit; never use it as a neutral or indecisive verdict.

## Required checks

1. **Naturalness and information value.**  The instruction should be a
   believable Japanese request with enough context to teach a reusable skill.
   Reject mechanical slot filling, bare count/keyword templates, answer-like
   instructions, and prompts whose only purpose is to force a surface form.
2. **Response alignment and Japanese quality.**  The response must satisfy the
   request naturally, be coherent and appropriately styled, and not merely
   echo constraints.  Reject awkward literal translation, excessive honorific
   padding, factual/logical mismatch, or an output that is formally valid but
   unhelpful as Japanese communication.
3. **Harmful signal.**  Reject meaningless repetition, garbled sentences,
   unjustified Japanese/English mixing, placeholder artifacts, contradictory
   constraints, or any pattern that rewards copying a hidden answer from the
   instruction.

## Verdict definitions and reason

- `accept`: clearly natural, useful, aligned Japanese with no material issue.
- `reject`: any material failure; one is sufficient.
- `borderline`: a single minor, stated uncertainty only.

Give one sentence of at most 160 characters that names the decisive evidence.
Do not invent context or claim a validator checked more than the supplied
record shows.
