"""Synthetic checks only; no complaint-data or model access."""
import unittest
import numpy as np
from sklearn.metrics import f1_score
from report_final_temporal_test import macro_from_codes

class PairedBootstrapMetrics(unittest.TestCase):
    def test_confusion_macro_matches_sklearn_including_absent_label(self):
        y=np.array([0,0,1,1,1,2]);p=np.array([0,1,1,2,1,0])
        self.assertAlmostEqual(macro_from_codes(y,p,4),f1_score(y,p,labels=range(4),average='macro',zero_division=0))

    def test_paired_identical_predictions_have_zero_difference(self):
        y=np.array([0,0,1,1,1,2]);p=np.array([0,1,1,2,1,0]);rng=np.random.default_rng(20261009)
        for _ in range(100):
            idx=rng.integers(0,len(y),size=len(y))
            self.assertEqual(macro_from_codes(y[idx],p[idx],4)-macro_from_codes(y[idx],p[idx],4),0)

if __name__=='__main__':unittest.main()
