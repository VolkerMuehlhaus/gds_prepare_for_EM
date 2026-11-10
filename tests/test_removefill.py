# gds_removefill: floating fill (a size that repeats often enough) goes, design metal stays, and a zero-width EM
# port marker no longer crashes its polygon merge (gdspy.boolean returns None for it).

from conftest import design_summary, run_tool, write_gds


def test_removes_fill_keeps_design_and_markers(tmp_path):
    fill = [(8, 0, (100 + 4 * (n % 6), 4 * (n // 6), 101 + 4 * (n % 6), 1 + 4 * (n // 6))) for n in range(30)]
    src = write_gds(tmp_path / "in.gds", {"TOP": [(8, 0, (0, 0, 40, 30)), (201, 0, (5, 5, 5, 8)),   # zero width
                                                  (202, 0, (2, 2, 4, 4))] + fill})
    result = run_tool("gds_removefill", src, tmp_path / "out.gds")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    after = design_summary(tmp_path / "out.gds")
    assert after[(8, 0)] == (1, 1200.0), "the 30 fill squares are gone, the design metal stays"
    assert (202, 0) in after
