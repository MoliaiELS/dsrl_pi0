"""CPU checks that UUID normalization stays within the allocated device file."""
import unittest

from tools.slurm_gpu import check_gres, select_egl_device, select_gpu


class SlurmGpuBinding(unittest.TestCase):
    def setUp(self):
        self.env = {'SLURM_JOB_ID': '123', 'SLURM_JOB_GPUS': '3', 'CUDA_VISIBLE_DEVICES': '3'}
        # NVML ordinals can differ from device-file minors after a reboot.
        self.gpus = [{'nvml_index': 0, 'minor_number': 3, 'uuid': 'GPU-allocated'},
                     {'nvml_index': 3, 'minor_number': 0, 'uuid': 'GPU-other'}]

    def test_uses_allocated_minor_rather_than_nvml_ordinal(self):
        self.assertEqual(select_gpu(self.env, self.gpus)['uuid'], 'GPU-allocated')

    def test_accepts_cgroup_relative_mask_and_matching_uuid(self):
        for mask in ['0', 'GPU-allocated']:
            with self.subTest(mask=mask):
                self.assertEqual(select_gpu({**self.env, 'CUDA_VISIBLE_DEVICES': mask}, self.gpus)['uuid'], 'GPU-allocated')

    def test_rejects_conflicting_or_expanded_visibility(self):
        for mask in ['GPU-other', '1', '0,3', '']:
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                select_gpu({**self.env, 'CUDA_VISIBLE_DEVICES': mask}, self.gpus)

    def test_rejects_missing_or_multiple_allocations(self):
        for allocation in ['', '0,3']:
            with self.subTest(allocation=allocation), self.assertRaises(ValueError):
                select_gpu({**self.env, 'SLURM_JOB_GPUS': allocation}, self.gpus)
        with self.assertRaises(ValueError):
            select_gpu({**self.env, 'SLURM_JOB_ID': ''}, self.gpus)

    def test_rejects_absent_or_ambiguous_device_identity(self):
        for inventory in [[], self.gpus + [self.gpus[0]]]:
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                select_gpu(self.env, inventory)

    def test_requires_matching_gres_file_mapping(self):
        text = 'NodeName=4090node[1-3] Name=gpu Type=4090 File=/dev/nvidia[0-3]'
        check_gres(text, '4090node2')
        for config in [text.replace('[0-3]', '[1-4]'), text.replace('4090node', '3090node'), '']:
            with self.subTest(config=config), self.assertRaises(ValueError):
                check_gres(config, '4090node2')

    def test_matches_egl_by_pci_rather_than_assuming_same_index(self):
        devices = [{'egl_index': 2, 'pci_bus_id': '0000:CA:00.0'},
                   {'egl_index': 0, 'pci_bus_id': '0000:31:00.0'}]
        self.assertEqual(select_egl_device('00000000:ca:00.0', devices)['egl_index'], 2)

    def test_rejects_missing_or_ambiguous_egl_identity(self):
        target = {'egl_index': 2, 'pci_bus_id': '0000:CA:00.0'}
        for devices in [[], [target, target]]:
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                select_egl_device('00000000:ca:00.0', devices)


if __name__ == '__main__':
    unittest.main()
