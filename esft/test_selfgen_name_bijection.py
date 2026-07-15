import copy
import json
from pathlib import Path
import tempfile

import selfgen_name_bijection as bijection
import selfgen_toolcall_intent_v1 as intent


SEEDS = Path(__file__).parent / "data/selfgen_toolcall_intent_v1/intent_r4_divnames_desc2/seeds.json"


def _seeds():
    return json.loads(SEEDS.read_text(encoding="utf-8"))["seeds"]


def _training_record(seed):
    selected = copy.deepcopy(seed["expected_stages"])
    results = [[intent.mock_execute(call, stage_no, seed["pattern"]) for call in stage]
               for stage_no, stage in enumerate(selected)]
    return {"seed_id": seed["seed_id"], **intent.render_training(seed, selected, results)}


def test_round_trip_restores_seed_namespace():
    original = _seeds()[0]
    mocked, mapping = bijection.mockize_seed(original)
    restored = bijection.restore_record(mocked, mapping)
    assert restored == original


def test_first_fifty_mock_invariants_and_round_trip():
    for original in _seeds()[:50]:
        mocked, mapping = bijection.mockize_seed(original)
        assert mocked["user_request"] == original["user_request"]
        assert len(mocked["tools"]) == len(original["tools"])
        names = [tool["name"] for tool in mocked["tools"]]
        assert len(names) == len(set(names))
        # Visible arguments are unchanged; receipts are recomputed from renamed calls.
        derived = {(item["stage"], item["field"]) for item in original.get("derived_values", [])}
        for stage_no, (old_stage, new_stage) in enumerate(zip(
                original["expected_stages"], mocked["expected_stages"])):
            for old_call, new_call in zip(old_stage, new_stage):
                assert [value for field, value in old_call["arguments"].items() if (stage_no, field) not in derived] == [
                    value for field, value in new_call["arguments"].items()
                    if (stage_no, field) not in {(item["stage"], item["field"])
                                                for item in mocked.get("derived_values", [])}
                ]
        assert bijection.restore_record(mocked, mapping) == original


def test_restore_changes_only_names_in_training_record():
    original = _seeds()[0]
    mocked, mapping = bijection.mockize_seed(original)
    record = _training_record(mocked)
    before = copy.deepcopy(record)
    restored = bijection.restore_record(record, mapping)
    expected = _training_record(original)
    assert restored == expected
    # The original record object is not mutated by restoration.
    assert record == before


def test_recompute_derived_uses_mock_call_receipts_and_reverses_exactly():
    original = next(seed for seed in _seeds() if seed["pattern"] == "long_chain")
    mocked, mapping = bijection.mockize_seed(original)
    stage0 = [intent.mock_execute(call, 0, mocked["pattern"]) for call in mocked["expected_stages"][0]]
    stage1 = mocked["expected_stages"][1][0]["arguments"]
    assert stage1["first_receipt"] == stage0[0]["result"]["receipt"]
    assert stage1["second_receipt"] == stage0[1]["result"]["receipt"]
    assert mocked["expected_stages"][3][0]["arguments"]["recovery_code"] == "UNAVAILABLE"
    assert bijection.restore_record(mocked, mapping) == original


def test_generation_request_covers_every_visible_value_in_all_patterns():
    for pattern in ("single", "parallel", "multi_turn", "error_recovery", "long_chain"):
        original = next(seed for seed in _seeds() if seed["pattern"] == pattern)
        mocked, _ = bijection.mockize_seed(original)
        request = bijection.generation_request(mocked)
        derived = {(item["stage"], item["field"]) for item in mocked.get("derived_values", [])}
        for stage_no, stage in enumerate(mocked["expected_stages"]):
            for call in stage:
                for field, value in call["arguments"].items():
                    if (stage_no, field) not in derived:
                        assert intent._literal(value) in request


def test_cli_mockize_preserves_requests_and_restores_rendered_gold_sample():
    original = next(seed for seed in _seeds() if seed["pattern"] == "long_chain")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        input_path = root / "input.json"
        input_path.write_text(json.dumps({"seeds": [original]}), encoding="utf-8")
        bijection._cmd_mockize(input_path, root / "mocked")
        mocked = json.loads((root / "mocked" / "seeds.json").read_text(encoding="utf-8"))["seeds"][0]
        mapping = json.loads((root / "mocked" / "bijection_maps.json").read_text(encoding="utf-8"))[original["seed_id"]]
    assert mocked["user_request"] == bijection.generation_request(mocked)
    assert mocked["_bijection_original_requests"] == {
        key: original[key] for key in ("user_request", "natural_request", "transcription_request")
    }
    record = _training_record(mocked)
    restored = bijection.restore_record(record, mapping)
    assert restored == _training_record(original)
    assert bijection.restore_record(mocked, mapping) == original


def test_all_five_thousand_seeds_mockize_without_exception():
    for seed in _seeds():
        bijection.mockize_seed(seed)
