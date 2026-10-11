"""Fixture for resume_check_v2 (offline): the current user-stopped state refuses; one future normal-wall invocation with
the single pinned gap passes; a second missing summary, a non-wall stop or a mismatched gap pin refuses."""
import json, shutil, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_check_v2 as g  # noqa: E402

HIST = ["start attempt-line 1 2026-10-10T23:59:01Z", "exit 0 2026-10-11T05:59:36Z", "checker exit 0 2026-10-11T05:59:41Z",
        "start attempt-line 4 2026-10-11T06:16:50Z", "exit 0 2026-10-11T07:28:14Z", "checker exit 0 2026-10-11T07:28:18Z",
        "start attempt-line 7 2026-10-11T07:37:38Z", "exit 0 2026-10-11T10:02:03Z", "checker exit 0 2026-10-11T10:02:08Z",
        "start attempt-line 10 2026-10-11T10:17:32Z", "exit 143 2026-10-11T12:45:56Z", "checker exit 0 2026-10-11T12:46:05Z"]
GAP = {"invocation": 4, "start_line": HIST[9], "exit_line": HIST[10], "wall_s": 8904}
SUMM = [{"stopped": "invocation wall budget 21600.0 s reached"}, {"stopped": "'378|w001_014' failed twice: Earth Engine capacity exceeded."},
        {"stopped": "fatal service error on '572|w001_002': Too many concurrent aggregations."}]


class Gate(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        (self.d / "out").mkdir()
        (self.d / "inputs").mkdir()
        (self.d / "inputs" / "production_manifest.json").write_text(json.dumps({"request_sha256": g.BODY}))
        self.repo = self.d / "repo"
        self.repo.mkdir()
        (self.repo / "a.py").write_text("x")
        self.receipt = {"accepted_summary_gaps": [GAP], "source_sha256": {"a.py": "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"}}

    def tearDown(self):
        shutil.rmtree(self.d)

    def state(self, exit_lines, summaries, checker_done=8280, summed=34523.9):
        (self.d / "invocations.exit.log").write_text("\n".join(exit_lines) + "\n")
        (self.d / "out" / "invocations.log").write_text("x\n" * sum(1 for l in exit_lines if l.startswith("start ")))
        (self.d / "out" / "run_summaries.jsonl").write_text("".join(json.dumps(s) + "\n" for s in summaries))
        for f in self.d.glob("check-*.json"):
            f.unlink()
        (self.d / "check-20261012T000000Z.json").write_text(json.dumps({
            "done": checker_done, "pending": 57801 - checker_done, "done_cells": 6.83e10 * checker_done / 8280,
            "pending_cells": 4.67e11 - 6.83e10 * checker_done / 8280, "call_reservations": 8310 + (checker_done - 8280),
            "call_cap": 59576, "invocations": 0, "max_invocations": 20, "n_violations": 0, "complete": False,
            "summed_invocation_seconds": summed}))
        return g.evaluate(self.d, self.receipt, self.repo, "")

    def test_current_user_stopped_state_refuses(self):
        r = self.state(HIST, SUMM)
        self.assertFalse(r["pass"])
        self.assertTrue(any("pinned user-stopped" in f for f in r["fails"]))
        self.assertFalse(any("unapproved summary" in f for f in r["fails"]))  # the pinned gap itself is accepted

    def test_future_normal_wall_invocation_with_single_pinned_gap_passes(self):
        inv5 = ["start attempt-line 13 2026-10-12T00:00:00Z", "exit 0 2026-10-12T06:00:30Z", "checker exit 0 2026-10-12T06:00:35Z"]
        r = self.state(HIST + inv5, SUMM + [{"stopped": "invocation wall budget 21600.0 s reached"}], checker_done=11000,
                       summed=34523.9 + 21600)
        self.assertTrue(r["pass"], r["fails"])
        self.assertEqual(r["corrected_elapsed_s"], 34523.9 + 21600 + 8904)  # gap time added once, from the receipt

    def test_current_corrected_projection_about_70_hours(self):
        r = self.state(HIST, SUMM)
        self.assertEqual(r["corrected_elapsed_s"], 34523.9 + 8904)
        self.assertAlmostEqual(r["projected_remaining_s"] / 3600, 70.7, delta=1.5)

    def test_second_missing_summary_refuses(self):
        inv5 = ["start attempt-line 13 2026-10-12T00:00:00Z", "exit 0 2026-10-12T06:00:30Z", "checker exit 0 2026-10-12T06:00:35Z"]
        r = self.state(HIST + inv5, SUMM, checker_done=11000)  # invocation 5 wrote no summary
        self.assertTrue(any("unapproved summary" in f for f in r["fails"]))

    def test_second_user_stop_without_summary_refuses(self):
        inv5 = ["start attempt-line 13 2026-10-12T00:00:00Z", "exit 143 2026-10-12T01:00:00Z", "checker exit 0 2026-10-12T01:00:05Z"]
        r = self.state(HIST + inv5, SUMM, checker_done=9000)
        self.assertFalse(r["pass"])
        self.assertTrue(any("unapproved summary" in f for f in r["fails"]))

    def test_non_wall_stop_refuses(self):
        inv5 = ["start attempt-line 13 2026-10-12T00:00:00Z", "exit 0 2026-10-12T02:00:00Z", "checker exit 0 2026-10-12T02:00:05Z"]
        r = self.state(HIST + inv5, SUMM + [{"stopped": "fatal service error on 'x': Too many concurrent aggregations."}], checker_done=9500)
        self.assertTrue(any("wall budget" in f for f in r["fails"]))

    def test_mismatched_gap_pin_refuses(self):
        self.receipt["accepted_summary_gaps"] = [{**GAP, "exit_line": "exit 143 2026-10-11T12:45:57Z"}]
        inv5 = ["start attempt-line 13 2026-10-12T00:00:00Z", "exit 0 2026-10-12T06:00:30Z", "checker exit 0 2026-10-12T06:00:35Z"]
        r = self.state(HIST + inv5, SUMM + [{"stopped": "invocation wall budget 21600.0 s reached"}], checker_done=11000)
        self.assertTrue(any("does not match" in f for f in r["fails"]))


if __name__ == "__main__":
    unittest.main()
