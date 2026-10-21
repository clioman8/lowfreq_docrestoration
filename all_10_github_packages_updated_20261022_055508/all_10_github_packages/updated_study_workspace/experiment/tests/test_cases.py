import unittest
from docrestore.cases import select_cases


class CaseSelectionTests(unittest.TestCase):
    def fixture(self):
        config={"analysis":{"example_seed":0,"threshold":.1,"threshold_adjacent_interval":[.08,.12]},
                "degradation":{"severities":[{"id":"S0"},{"id":"S1"}]}}
        source=[{"sample_id":name,"severity_id":s,"split":"test","gt_normalized":"same text", "cer":v}
                for name,s,v in [("a","S0",0),("a","S1",.11),("b","S0",0),("b","S1",.7)]]
        groups=[[dict(r,seed=0,method=g,cer=r["cer"]/2) for r in source] for g in ("A","B","C")]
        return config,source,groups

    def test_near_and_severe_use_input_not_restored_ranking(self):
        config,source,groups=self.fixture()
        groups[2][1]["cer"]=.9
        report=select_cases(*groups,source,config)
        self.assertEqual(report["cases"]["threshold_adjacent"]["sample_id"],"a")
        self.assertEqual(report["cases"]["severe_input"]["sample_id"],"b")
        self.assertEqual(report["cases"]["largest_C_deterioration_vs_A"]["sample_id"],"a")

    def test_no_failure_is_not_invented(self):
        config,source,groups=self.fixture()
        report=select_cases(*groups,source,config)
        self.assertEqual(report["cases"]["largest_C_deterioration_vs_A"]["status"],"not_found")

    def test_mismatched_targets_are_rejected(self):
        config,source,groups=self.fixture()
        groups[1][0]["gt_normalized"]="wrong text"
        with self.assertRaises(ValueError):
            select_cases(*groups,source,config)
