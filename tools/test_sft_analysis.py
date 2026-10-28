"""Validate scientific closure against a real cached trajectory and fault cases."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from tools.research_core import atomic,read
from tools.sft_analysis import ARMS,METRICS,audit_episode,paired_statistics


class PairedStatistics(unittest.TestCase):
    def test_ever_does_not_hide_wrong_goal_first(self):
        index={};seeds=list(range(20))
        for arm in ARMS:
            for seed in seeds:
                for goal in ('A','B'):
                    first=goal if arm=='adapted' else 'A'
                    row={m:True for m in METRICS}
                    row.update(first_grasp_class=first,correct_first_grasp=first==goal,
                        active_goal_first_completion_without_prior_wrong_goal=arm=='adapted' or goal=='A')
                    index[arm,seed,goal]=row
        result=paired_statistics(index,seeds)
        b=result['adapted_minus_unadapted']['B']
        self.assertEqual(b['active_goal_ever_success']['estimate'],0)
        self.assertEqual(b['correct_first_grasp']['estimate'],1)
        self.assertEqual(b['active_goal_first_completion_without_prior_wrong_goal']['estimate'],1)
        self.assertEqual(result['adapted_minus_unadapted']['L']['estimate'],1)
        self.assertEqual(result['adapted_minus_unadapted']['pair_both_correct']['reset_clusters'],20)
        del index['adapted',0,'B']
        with self.assertRaises(KeyError):paired_statistics(index,seeds)


class SavedTrace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ.get('DSRL_ANALYSIS_FIXTURE','tmp/language-stage-20261027/job-18584'))
        if not (cls.root/'case_manifest.json').is_file():raise unittest.SkipTest('Provide DSRL_ANALYSIS_FIXTURE with first_batch saved traces')
        cls.protocol=read(cls.root/'resolved_protocol.json')['protocol']
        case=next(c for c in read(cls.root/'case_manifest.json') if c['phase']=='e1_dev' and c['policy']=='base' and c['prompt_id']=='A')
        cls.record=read(cls.root/'cases'/(case['id']+'.json'))

    def test_actual_400_step_trace(self):
        trace=audit_episode(self.root,self.record,self.protocol,self.record['code_commit'])
        self.assertEqual(trace['noise'].shape,(20,1,50,32))

    def fixture(self,root):
        record=copy.deepcopy(self.record)
        paths=list(record['artifact_hashes'])+['cases/'+record['case']['id']+'.json',
            'common_states/'+record['common_state']['id']+'.npz']
        for relative in paths:
            target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(self.root/relative,target)
        return record

    def test_changed_saved_actions_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=self.fixture(root)
            path=root/next(k for k in record['artifact_hashes'] if k.endswith('/arrays.npz'))
            with path.open('ab') as stream:stream.write(b'corruption')
            with self.assertRaisesRegex(AssertionError,'Corrupt result'):
                audit_episode(root,record,self.protocol,record['code_commit'])

    def test_reported_grasp_cannot_override_actual_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record=self.fixture(root)
            record['first_grasp_class']='OTHER' if record['first_grasp_class']!='OTHER' else 'NONE'
            atomic(root/'cases'/(record['case']['id']+'.json'),record)
            with self.assertRaises(AssertionError):
                audit_episode(root,record,self.protocol,record['code_commit'])


if __name__=='__main__':unittest.main()
