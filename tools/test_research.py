"""Scientific failure-mode regression checks; no simulator or GPU dependencies."""
import tempfile
from pathlib import Path
import unittest

from tools.research_core import (Budget, BudgetExceeded, GraspTracker, StageTracker, atomic,
    case_plan, completed, file_sha, fixed_schedule, lock, read, validate)
from tools.stage_probe import prompt_for
from tools.research_stats import interval, paired_contrasts


class ResearchTests(unittest.TestCase):
    def setUp(self):
        self.p=read('configs/research_protocol_v1.json')

    def test_pre_registered_matrix_and_no_reserved(self):
        self.assertEqual(validate(self.p)['counts']['e1_confirm'],800)

    def test_first_grasp_is_locked_and_ties_are_multiple(self):
        t=GraspTracker(['cream_cheese_1','butter_1','other'])
        for i in range(1,4):t.update(dict(cream_cheese_1=False,butter_1=True,other=False),i)
        for i in range(4,8):t.update(dict(cream_cheese_1=True,butter_1=False,other=False),i)
        self.assertEqual((t.first_class,t.first_step),('B',3))
        t=GraspTracker(['x','y'])
        for i in range(3):t.update(dict(x=True,y=True),i+1)
        self.assertEqual(t.first_class,'MULTIPLE')

    def test_nonconsecutive_grasp_does_not_count(self):
        t=GraspTracker(['x'])
        for i,value in enumerate([True,True,False,True,True]):t.update({'x':value},i)
        self.assertEqual(t.first_class,'NONE')

    def test_oracle_rollback_uses_current_compound_not_history(self):
        t=StageTracker()
        for i in range(3):t.observe(True,i+1)
        self.assertEqual(t.query(20),1)
        t.observe(False,21);t.observe(False,22)
        self.assertEqual(t.query(22),1)
        t.observe(False,23)
        self.assertEqual(t.query(40),0)
        self.assertEqual(t.first_complete,3)

    def test_400_step_mock_ignores_every_native_done_and_keeps_B(self):
        import numpy as np
        from types import SimpleNamespace
        from tools.research_runner import rollout
        c=next(c for c in case_plan(self.p) if c['phase']=='e1_dev' and c['prompt_id']=='B')
        c=dict(c,reset_seed=1)  # No rendering in this mocked technical check.
        class Env:
            def __init__(self): self.steps=0
            def step(self,action):
                self.steps+=1
                return {},0,True,{}
            def get_sim_state(self):return np.asarray([self.steps],float)
            def close(self):pass
        env=Env();calls=[]
        def infer(obs,text,noise):
            calls.append(text);return np.zeros((50,7))
        def event(env,task):
            return dict(goals=dict(A=env.steps>0,B=False),grasp=dict(cream_cheese_1=False,butter_1=False),
                contact=dict(cream_cheese_1=False,butter_1=False),positions=dict(cream_cheese_1=[0,0,0],butter_1=[0,0,0]),
                official_success=env.steps>0)
        with tempfile.TemporaryDirectory() as d:
            b=SimpleNamespace(p=self.p,output=Path(d),logger=SimpleNamespace(log=lambda *a:None),
                reset=lambda *a:(env,{}),common=lambda *a:dict(id='mock'),events=event,infer=infer,
                audit_prompt=lambda text:dict(tokens=[],mask=[]),noise=lambda *a:(np.zeros((1,50,32)),np.array([0,0])))
            result=rollout(b,c,Budget(Path(d)/'budget.json'),{},0)
            self.assertEqual((env.steps,result['executed_env_steps'],result['vla_queries']),(400,400,20))
            self.assertEqual(result['status'],'POLICY_FAILURE')
            self.assertTrue(result['goal_ever']['A'])
            self.assertFalse(result['active_goal_ever_success'])
            self.assertTrue(all(x==self.p['e1']['goals'][1]['text'] for x in calls))
        c['prompt_id']='empty'
        self.assertEqual(prompt_for(self.p,c,StageTracker(),0,{})[0],'')

    def test_fixed_calibration_clips_rounds_and_ignores_missing(self):
        self.assertEqual(fixed_schedule([121,None,122],200),140)
        self.assertEqual(fixed_schedule([None]*10,300),300)
        self.assertEqual(fixed_schedule([599],200),300)

    def test_budget_interruption_conservatively_charged_and_cap_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'budget.json'
            b=Budget(path,cap=30);b.reserve('interrupted',20,'policy')
            b=Budget(path,cap=30)
            self.assertEqual(b.state['uncertain_steps'],20)
            with self.assertRaises(BudgetExceeded):b.reserve('too_much',11,'policy')
            b.reserve('replay',10,'replay');b.settle('replay',7)
            self.assertEqual(b.state['physical_steps_upper_bound'],27)

    def test_resume_skips_complete_and_rejects_changed_artifact_or_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=root/'actions.npz';a.write_bytes(b'fixed-actions')
            c={'id':'case'};value={'case':c,'artifact_hashes':{'actions.npz':file_sha(a)}}
            atomic(root/'cases/case.json',value)
            self.assertEqual(completed(root,c),value)
            with self.assertRaises(AssertionError):completed(root,dict(id='case',changed=True))
            a.write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):completed(root,c)
            lock(root/'manifest.json',{'x':1})
            with self.assertRaises(AssertionError):lock(root/'manifest.json',{'x':2})

    def test_cluster_statistics_keep_paired_reset_and_missing_bounds(self):
        plan=[c for c in case_plan(self.p) if c['phase']=='e1_confirm']
        records=[dict(case=c,status='COMPLETE',correct_first_grasp=c['policy']!='base',active_goal_ever_success=False)
                 for c in plan if c['reset_seed']<3139]
        stats=paired_contrasts(records,plan,'e1_confirm')
        self.assertEqual(len(stats['complete_clusters']),39)
        effect=stats['contrasts']['correct_first_grasp__dsrl_mean_minus_base']
        self.assertEqual(effect['estimate'],1)
        self.assertTrue(effect['degenerate'])
        self.assertLess(effect['missing_sensitivity'][0],1)
        self.assertEqual(interval([])['estimate'],None)

    def test_full_report_includes_policy_failure_and_complete_cluster_denominators(self):
        from tools.research_stats import generate
        plan=case_plan(self.p)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            atomic(root/'case_manifest.json',plan)
            atomic(root/'run_manifest.json',dict(code_commit='mock'))
            atomic(root/'E0_PASS.json',dict(passed=True))
            for c in plan:
                if c['phase']=='e0':continue
                requested=c['requested_goal_id']
                goals=dict(A=requested=='A',B=requested=='B') if c['task']=='E1' else {'global':False}
                r=dict(case=c,status='POLICY_FAILURE',artifact_hashes={},first_grasp_class=requested or 'NONE',
                    correct_first_grasp=bool(requested),active_goal_ever_success=False if requested or c['task']!='E1' else None,
                    goal_ever=goals,first_stage_completion=100, failure_diagnostics=dict(timeout=True))
                atomic(root/'cases'/(c['id']+'.json'),r)
            decision=generate(root,self.p)
            self.assertEqual(decision['execution_status'],'COMPLETE')
            self.assertEqual(decision['scientific_decision'],'EXECUTION_BOTTLENECK')
            self.assertEqual(decision['completed_cases'],990)
            self.assertEqual(len(read(root/'paired_cluster_statistics.json')['e1_confirm']['complete_clusters']),40)


if __name__=='__main__':unittest.main()
