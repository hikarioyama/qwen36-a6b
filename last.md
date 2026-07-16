# Paraphrase quality gate implementation

## Changed files

- `esft/selfgen_toolcall_intent_v1.py`
  - assigns a deterministic style card per seed;
  - adds an English semantic description to every generated schema property and
    treats a missing description as a generator error;
  - sets the default tier mix to zero T1 transcription rows;
  - includes gate context and style metadata in the private paraphrase batch;
  - reruns all deterministic gates during ingest, quarantines failed or missing
    paraphrases in `paraphrase_excluded.jsonl`, and never falls back to a
    transcription.
- `esft/paraphrase_glm52_fireworks.py`
  - prompts with the selected style card, removes the first-person/chat-email
    constraint, and applies the per-row deterministic gates before writeback.
- `esft/paraphrase_gates.py`
  - implements typed-literal fidelity, schema-name leakage, operational
    invention, batch prefix/style quotas, and character-3-gram duplicate gates;
  - provides the offline shadow report command.
- `esft/tests/test_paraphrase_gates.py`
  - covers accepted and rejected examples for every implemented gate.
- `esft/tests/test_paraphrase_driver.py`
  - verifies style-card prompting and the driver's local gate path with a mocked
    API call (no network required).
- `esft/tests/test_selfgen_toolcall_intent_v1.py`
  - verifies failed paraphrases are excluded rather than downgraded to a
    transcription.

## Use

Prepare and emit a private paraphrase batch as usual:

```bash
python3 esft/selfgen_toolcall_intent_v1.py prepare --run-id <run-id> --name-style diverse
python3 esft/selfgen_toolcall_intent_v1.py emit-paraphrase-batch --run-id <run-id>
```

Run an offline fail-rate audit before accepting a batch.  It only reports; it
does not modify the run:

```bash
python3 esft/paraphrase_gates.py --shadow --run-id <run-id> --paraphrases <writeback.jsonl>
```

The normal driver supplies style cards and runs the per-row gates.  It can be
imported and unit-tested without API access; an actual invocation needs the
usual API credential:

```bash
python3 esft/paraphrase_glm52_fireworks.py --batch esft/data/selfgen_toolcall_intent_v1/<run-id>/paraphrase_batch.jsonl --out <writeback.jsonl>
```

Authoritative ingestion repeats all deterministic checks and writes only passed
paraphrases into `seeds.json`; every failure is recorded in the quarantine file:

```bash
python3 esft/selfgen_toolcall_intent_v1.py ingest-paraphrase --run-id <run-id> --input <writeback.jsonl>
```

No output artifact records machine hostnames or absolute input paths.
