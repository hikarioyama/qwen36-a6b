# Corpus Judge v1: tool-call data rubric

## Purpose and decision rule

Use this rubric only after programmatic/schema/executor validation.  It is a
second-stage quality filter, not a replacement for those checks.  Judge the
example as training signal for a capable tool-using agent, rather than merely
asking whether its recorded call happens to execute.

Return exactly one of `accept`, `reject`, or `borderline`, plus one short,
specific reason on one line.  The default is strict: **when the evidence is
mixed, uncertain, or insufficient, return `reject`, not `borderline` or
`accept`**.  `borderline` is only for an otherwise useful record with one
minor, clearly identified uncertainty that a later audit may resolve; it is
not a convenience bucket for indecision.  Do not infer facts absent from the
record.

## Required checks

1. **Natural, informative request.**  The user request should resemble a
   plausible task, provide meaningful context and constraints, and require an
   agent decision.  Reject mechanical templates, placeholder-only domains or
   arguments, and requests that spell out the answer such as the exact tool
   name, complete arguments, or prescribed trace/order without a real-world
   reason.  A syntactically natural sentence is still rejectable if it is only
   a transcription exercise.
2. **Correctness and alignment.**  The assistant response/tool calls must
   address the request, use available tools with valid arguments, preserve
   stated constraints, and make sequencing/conditional behaviour intelligible.
   Reject hallucinated, irrelevant, duplicated, inexplicable, or
   instruction-conflicting calls even if an executor accepted them.  A tool
   result must be used coherently where the next call claims to depend on it.
3. **Training-signal safety.**  Reject meaningless repetition, broken prose,
   malformed/placeholder content, unexplained language switching, irrelevant
   boilerplate, leakage of hidden answers, or patterns likely to teach a model
   to imitate a bad interaction.  Be especially alert to synthetic-looking
   names such as generic `field_N` arguments when they carry no semantic task
   meaning.

## Verdict definitions

- `accept`: all required checks are clearly satisfied; the request contains a
  real decision/problem and the response is an accurate, useful demonstration.
- `reject`: any material failure above, including an answer-revealing request
  or a low-value templated interaction.  One material defect is enough.
- `borderline`: all core checks pass, but one minor quality uncertainty remains
  and the record could still be useful.  State that uncertainty precisely.

## Reason format

Write a single sentence of at most 160 characters, naming the decisive
observation rather than a score.  Examples: `Exact tool name and all arguments
are embedded in the request, so this is a transcription task.`  Do not claim
that an example was executed, fact-checked, or decontaminated unless that
evidence is present in the supplied record.

## Tool-call-specific cautions

- A correct trace does not rescue a request that exposes the target call.
- Do not reward extra calls, long traces, or tool count by themselves.
- Conditional recovery is useful only when the condition and recovery action
  make semantic sense; a branch written as a fixed answer template is reject.
- Prefer a concise reject for ambiguity over guessing the intended tool.
