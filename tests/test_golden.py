# Golden-file checks on the real test cases in testcases/: the per-layer summary (polygon count, area) of each tool's
# output must stay as stored in tests/golden/. A change that alters a result on purpose updates the files:
#
#     UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py
#
# and the diff of tests/golden/ in the commit shows what changed. Marked "golden" (~30 s): -m "not golden" skips them.
# testcases/ is not in git (.gitignore): the cases come from there or from GDS_TESTCASES, and are skipped if missing.

import json
import os

import pytest

from conftest import REPO, design_summary, run_tool

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
TESTCASES = os.environ.get("GDS_TESTCASES") or os.path.join(REPO, "testcases")
CASES = ["T582_BEOL_I1_2x300u_forEM", "T582_BEOL_D1_300u"]
TOOLS = ["gds_prepare_for_EM", "gds_removefill"]


@pytest.mark.golden
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize("case", CASES)
def test_golden(tmp_path, case, tool):
    src = os.path.join(TESTCASES, case + ".gds")
    if not os.path.isfile(src):
        pytest.skip(f"test case {case}.gds not in {TESTCASES} (testcases/ is not in git)")
    out = tmp_path / "out.gds"
    result = run_tool(tool, src, out)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    summary = {f"{layer}/{datatype}": list(value) for (layer, datatype), value in sorted(design_summary(out).items())}
    path = os.path.join(GOLDEN, f"{case}__{tool}.json")
    if os.environ.get("UPDATE_GOLDEN"):
        os.makedirs(GOLDEN, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
            f.write("\n")
        pytest.skip(f"golden file written: {os.path.basename(path)}")
    with open(path, encoding="utf-8") as f:
        expected = json.load(f)
    changed = {key: (expected.get(key), summary.get(key)) for key in sorted(set(expected) | set(summary))
               if expected.get(key) != summary.get(key)}
    assert not changed, f"layer: (golden [count, area], now) {changed}"
