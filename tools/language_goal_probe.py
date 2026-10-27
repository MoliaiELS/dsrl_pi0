"""Full-rollout frozen VLA backend. Truth is used only by the evaluator."""
import copy
import hashlib
import inspect
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np

from tools.research_core import atomic, file_sha, lock, read, sha


class InvalidConstruction(RuntimeError):
    pass


class Backend:
    def __init__(self, p, output, logger):
        from tools.control_probe import Probe
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        from openpi.models.tokenizer import PaligemmaTokenizer
        from flax import nnx
        import jax

        self.p, self.output, self.logger, self.jax, self.nnx = p, Path(output), logger, jax, nnx
        self.data = Path(os.environ['DSRL_DATA_ROOT'])
        registry = benchmark.get_benchmark_dict()['libero_90']()
        self.tasks, self.paths, taskmap = {}, {}, {}
        for ident, name in [('E1', p['e1']['physical_task_name'])] + [(t['id'], t['task_name']) for t in p['e2']['tasks']]:
            matches = [(i,registry.get_task(i)) for i in range(registry.n_tasks)
                       if Path(registry.get_task(i).bddl_file).stem == name]
            assert len(matches) == 1, (name, matches)
            i, task = matches[0]
            self.tasks[ident] = task
            self.paths[ident] = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
            taskmap[ident] = dict(id=i, name=name, language=task.language, bddl_sha256=file_sha(self.paths[ident]))
        assert taskmap['E1']['id'] == p['e1']['audited_task_id']
        parsed_a = robosuite_parse_problem(str(self.paths['E1']))
        path_b = self.paths['E1'].with_name('LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray.bddl')
        parsed_b = robosuite_parse_problem(str(path_b))
        # Only labels / goal / obj_of_interest may differ. Always instantiate A.
        physical = ['problem_name','domain_name','regions','fixtures','objects','initial_state','scene_properties']
        physical = [k for k in physical if k in parsed_a]
        assert {'regions','objects','fixtures'}.issubset(physical), parsed_a.keys()
        assert all(parsed_a[k] == parsed_b[k] for k in physical), 'A/B official physical scenes differ'
        atomic(self.output / 'task_audit.json', dict(task_name_to_id=taskmap, parsed_A=parsed_a,
            parsed_B=parsed_b, compared_physical_keys=physical, B_bddl_sha256=file_sha(path_b)))
        cfg = dict(task_suite='libero_90', task_id=taskmap['E1']['id'], model_config='pi0_libero',
                   resolution=256, seed=9000, query_freq=20, instructions=[], physics_atol=1e-8,
                   proprio_atol=1e-7, image_atol=0)
        self.probe = Probe(cfg, self.output, logger)
        self.policy = self.probe.policy
        self.policy._sample_kwargs['num_steps'] = 10
        assert self.policy._sample_kwargs == {'num_steps':10}
        signature = inspect.signature(self.policy._sample_actions)
        assert signature.parameters['num_steps'].default == 10, signature
        self.tokenizer = PaligemmaTokenizer(max_len=48)
        self.prompt_audit = {}
        texts = [g['text'] for g in p['e1']['goals']] + [a['text'] for a in p['e1']['auxiliary_prompts']]
        for task in p['e2']['tasks']:
            texts += [p['e2']['prompt_global_template'].format(goal=task['global_goal'])]
            texts += [p['e2']['prompt_stage_template'].format(goal=task['global_goal'], stage=task['stage_'+str(i)]) for i in (0,1)]
        for text in texts:
            self.audit_prompt(text)
        atomic(self.output / 'prompt_audit.json', self.prompt_audit)
        self.actors, self.actor_variants, controllers = {}, {}, {}
        self.logger.log({'phase':'restoring_three_controllers'},0)
        from flax.training import checkpoints
        from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
        for spec in p['e1']['policies']:
            if spec['noise'] != 'trained_sac':
                continue
            source = (self.data / spec['source_run']).resolve()
            assert source.is_relative_to(self.data.resolve())
            config = read(source / 'config.json')
            assert read(source / 'SUCCESS.json')['passed']
            assert config['mode'] == 'train' and config['seed'] == spec['controller_training_seed']
            assert config['task_id'] == taskmap['E1']['id'] and config['task_suite'] == 'libero_90'
            exact = source / ('checkpoint' + str(spec['checkpoint_step']))
            assert exact.is_dir()
            side = config['resize_image']
            assert side == 64
            observations = {'pixels':np.zeros((1,side,side,3,1),np.uint8), 'state':np.zeros((1,8,1),np.float32)}
            kwargs = copy.deepcopy(config['train_kwargs'])
            if kwargs.pop('cosine_decay',False):
                kwargs['decay_steps'] = config['max_steps']
            actor = PixelSACLearner(0,observations,np.zeros((1,1,32),np.float32),**kwargs)
            # Explicit directory, never latest-checkpoint selection.
            restored = checkpoints.restore_checkpoint(str(exact), actor._save_dict)
            actor._actor, actor._critic = restored['actor'], restored['critic']
            actor._target_critic_params, actor._temp = restored['target_critic_params'], restored['temp']
            actor._rng = restored['rng']
            assert int(np.asarray(actor._actor.step)) == 500000
            self.actors[spec['id']] = actor
            self.actor_variants[spec['id']] = SimpleNamespace(env='libero',resize_image=side)
            controllers[spec['id']] = dict(path=str(exact), training_seed=config['seed'], step=500000,
                file_hashes={str(f.relative_to(exact)):file_sha(f) for f in sorted(exact.rglob('*')) if f.is_file()},
                source_run=read(source/'run.json'), source_config_sha256=file_sha(source/'config.json'))
        atomic(self.output/'checkpoint_manifest.json', controllers)
        self.logger.log({'phase':'hashing_frozen_weights_and_assets'},0)
        self.controllers_before = self.controller_hashes()
        self.model_before = self.model_fingerprint()
        modelpath = Path(os.environ['DSRL_MODEL_PATH'])
        download = read(modelpath/'download_manifest.json')
        for item in download['files']:
            assert (modelpath/item['path']).stat().st_size == item['size']
            assert file_sha(modelpath/item['path']) == item['sha256'], item['path']
        self.resolved = dict(task_name_to_id=taskmap, model_path=str(modelpath),
            model_hashes=download, norm_stats_path=str(modelpath/'assets/physical-intelligence/libero/norm_stats.json'),
            controller_paths={k:v['path'] for k,v in controllers.items()},
            prompt_token_audit=self.prompt_audit, sampling_settings=p['shared_eval'],
            bash_setup_path=os.environ['DSRL_BASH_SETUP'], bash_setup_sha256=file_sha(os.environ['DSRL_BASH_SETUP']),
            slurm_profile={k:v for k,v in os.environ.items() if k in ['SLURM_JOB_PARTITION','SLURM_CPUS_PER_TASK','CUDA_VISIBLE_DEVICES']})
        atomic(self.output/'trainable_manifest.json',dict(gradient_updates=0,replay_insertions=0,optimizer_created=False,
            controllers_before=self.controllers_before, vla_state_before=self.model_before))
        self.reference = None
        self.logger.log({'phase':'backend_ready'},0)

    def audit_prompt(self, text):
        clean = text.strip().replace('_',' ').replace('\n',' ')
        raw = self.tokenizer._tokenizer.encode(clean,add_bos=True) + self.tokenizer._tokenizer.encode('\n')
        assert len(raw) <= 48, f'Prompt truncation: {len(raw)} {text!r}'
        tokens, mask = self.tokenizer.tokenize(text)
        assert list(tokens[mask]) == raw
        audit = dict(text=text, untruncated_tokens=len(raw), tokens=tokens.tolist(), mask=mask.tolist(), truncated=False)
        self.prompt_audit[text] = audit
        return audit

    def tree_hash(self, value):
        h = hashlib.sha256()
        structure = str(self.jax.tree_util.tree_structure(value))
        h.update(structure.encode())
        for leaf in self.jax.tree_util.tree_leaves(value):
            a = np.asarray(leaf)
            h.update(str((a.shape,str(a.dtype))).encode())
            h.update(a.tobytes())
        return h.hexdigest()

    def model_fingerprint(self):
        model = self.policy._sample_actions.__wrapped__.__self__
        return self.tree_hash(self.nnx.state(model))

    def controller_hashes(self):
        return {k:self.tree_hash(v._save_dict) for k,v in self.actors.items()}

    def verify_frozen(self):
        assert self.controller_hashes() == self.controllers_before, 'Controller state changed'
        assert self.model_fingerprint() == self.model_before, 'Frozen VLA state changed'
        if self.reference:
            obs,text,noise,expected = self.reference
            np.testing.assert_allclose(self.infer(obs,text,noise),expected,atol=1e-5,rtol=0)
        atomic(self.output/'frozen_check.json',dict(passed=True, controllers=self.controllers_before,
            vla_state=self.model_before, gradient_updates=0,replay_insertions=0))

    def reset(self, task, seed):
        random.seed(seed)
        np.random.seed(seed)
        env = self.probe.env_class(bddl_file_name=str(self.paths[task]),camera_heights=256,camera_widths=256,
                                   ignore_done=True,horizon=1000,control_freq=20)
        try:
            env.seed(seed)
            obs = env.reset()
            assert env.env.ignore_done and env.env.control_freq == 20
            if not np.isfinite(env.get_sim_state()).all():
                raise InvalidConstruction('nonfinite_initial_data')
            if env.check_success():
                raise InvalidConstruction('goal_already_true_at_t0')
            if task == 'E1':
                events = self.events(env,task)
                if events['goals']['A'] or events['goals']['B']:
                    raise InvalidConstruction('goal_already_true_at_t0')
                if any(events['grasp'][g['object']] for g in self.p['e1']['goals']):
                    raise InvalidConstruction('candidate_already_grasped_at_t0')
            if task == 'T2':
                assert self.predicate(env,['open','wooden_cabinet_1_top_region']), 'T2 drawer must start open'
            return env,obs
        except BaseException:
            env.close()
            raise

    def snapshot(self,env,obs):
        return self.probe.snapshot(env,obs,[])

    def check_snapshot(self,expected,actual):
        assert str(expected['model_xml_sha256']) == str(actual['model_xml_sha256']), 'Model XML differs'
        errors={}
        for key,tol in [('physics',1e-8),('proprio',1e-7),('agentview_image',0),('wrist_image',0)]:
            err=float(np.max(np.abs(expected[key].astype(np.float64)-actual[key].astype(np.float64))))
            assert err <= tol, f'common_state_replay_mismatch {key}: {err}'
            errors[key]=err
        return errors

    def common(self,case,env,obs):
        snapshot = self.snapshot(env,obs)
        name = f"{case['task']}__{case['reset_seed']}"
        path = self.output/'common_states'/(name+'.npz')
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            with np.load(path) as saved:
                errors=self.check_snapshot(saved,snapshot)
        else:
            np.savez_compressed(path,**snapshot)
            errors={k:0 for k in ['physics','proprio','agentview_image','wrist_image']}
        return dict(id=name, file_sha256=file_sha(path), physics_sha256=sha(snapshot['physics'].tobytes()), errors=errors,
                    generator='fresh seeded reset; empty recorded prefix; warmup=0',called_set_init_state=False)

    def infer(self,obs,text,noise):
        result=self.probe.infer(copy.deepcopy(obs),text,noise)
        assert result.shape == (50,7), result.shape
        if self.reference is None:
            self.reference=(copy.deepcopy(obs),text,noise.copy(),result.copy())
        return result

    def noise(self,obs,policy,reset,sampling,query):
        from examples.train_utils_sim import obs_to_img
        key=self.jax.random.fold_in(self.jax.random.PRNGKey(73000+reset+100000*sampling),query)
        if policy == 'base':
            value=np.asarray(self.jax.random.normal(key,(1,50,32)))
        elif policy == 'bounded_random':
            value=np.repeat(np.asarray(self.jax.random.uniform(key,(1,1,32),minval=-1,maxval=1)),50,axis=1)
        else:
            actor=self.actors[policy]
            actor_obs=dict(pixels=obs_to_img(obs,self.actor_variants[policy])[None,...,None],
                state=self.probe.to_proprio(copy.deepcopy(obs),self.probe.variant)[None,...,None])
            previous=actor._rng
            try:
                actor._rng=key
                value=np.asarray(actor.sample_actions(actor_obs)).reshape(1,1,32)
            finally:
                actor._rng=previous
            assert np.max(np.abs(value)) <= 1.00001
            value=np.repeat(value,50,axis=1)
        assert value.shape == (1,50,32) and np.isfinite(value).all()
        return value,np.asarray(key)

    @staticmethod
    def predicate(env,expression):
        if expression[0].lower() == 'and':
            return all(Backend.predicate(env,e) for e in expression[1:])
        return bool(env.env._eval_predicate(expression))

    def events(self,env,task):
        objects=env.env.objects_dict
        grasp={k:bool(env.env._check_grasp(env.robots[0].gripper,v.contact_geoms)) for k,v in objects.items()}
        gripper=env.robots[0].gripper
        contact={k:bool(env.env.check_contact(gripper,v)) for k,v in objects.items()}
        positions={k:env.sim.data.body_xpos[env.env.obj_body_id[k]].copy().tolist() for k in objects}
        native=bool(env.check_success())
        if task == 'E1':
            goals={g['id']:self.predicate(env,g['predicate']) for g in self.p['e1']['goals']}
            assert goals['A'] == native
            return dict(goals=goals,grasp=grasp,contact=contact,positions=positions,official_success=native)
        spec=next(t for t in self.p['e2']['tasks'] if t['id']==task)
        stage=self.predicate(env,spec['stage_0_done'])
        if spec.get('stage_0_also_requires_not_grasped'):
            stage=stage and not grasp['butter_1']
        assert self.predicate(env,spec['goal']) == native
        return dict(goals={'global':native},grasp=grasp,contact=contact,positions=positions,
                    stage_0_done=stage,official_success=native,
                    drawer_open=self.predicate(env,['open','wooden_cabinet_1_top_region']),
                    drawer_closed=self.predicate(env,['close','wooden_cabinet_1_top_region']))
