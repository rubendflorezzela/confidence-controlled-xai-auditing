import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import aggregate as A  # noqa: E402
from config import CFG  # noqa: E402


class ReproducibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = A.load_records(CFG.paths.records_dir)
        cls.inst = A.per_instance(cls.df)

    def test_record_counts(self):
        self.assertEqual(len(self.df), 6000)
        self.assertEqual(len(self.inst), 1200)
        self.assertEqual(self.inst["instance_id"].nunique(), 600)

    def test_method_comparison(self):
        out = A.method_comparison(self.inst, "auc")
        row = out.iloc[0]
        self.assertAlmostEqual(row["mean_a"], 0.1117, places=4)
        self.assertAlmostEqual(row["mean_b"], 0.3499, places=4)
        self.assertAlmostEqual(row["median_diff"], -0.1789, places=4)
        self.assertEqual(int(row["n_pairs"]), 600)

    def test_controlled_drise_gaps(self):
        edges = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        out = A.crossdomain_controlled(
            self.inst, "auc", edges, n_boot=200, alpha=0.05
        )
        d = out[out["method"] == "d_rise"].reset_index(drop=True)
        gaps = d["gap_pie_minus_jaad"].to_numpy()
        self.assertTrue(np.allclose(gaps, [0.0051, 0.0341, 0.0429, 0.0481, 0.0115], atol=5e-4))

    def test_global_f0_coupling(self):
        out = A.confound_report(self.inst)
        dr = out[(out["method"] == "d_rise") & (out["scope"] == "ALL") & (out["x"] == "score_start")]
        ec = out[(out["method"] == "eigencam") & (out["scope"] == "ALL") & (out["x"] == "score_start")]
        self.assertAlmostEqual(float(dr.iloc[0]["spearman_rho"]), 0.7620, places=4)
        self.assertAlmostEqual(float(ec.iloc[0]["spearman_rho"]), 0.7444, places=4)


if __name__ == "__main__":
    unittest.main()
