"""Checks for the sanitized launch templates; no training libraries required."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicConfigurationTests(unittest.TestCase):
    def test_manifest_uses_relative_paths_and_detects_source_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'scripts').mkdir()
            (root / 'nnunetv2').mkdir()
            script = root / 'scripts/build_umctf_sacl_c2_source_manifest.py'
            shutil.copyfile(ROOT / script.relative_to(root), script)
            source = root / 'nnunetv2/example.py'
            source.write_text('VALUE = 1\n')
            subprocess.run(['python3', str(script), 'write'], check=True, capture_output=True)
            manifest = json.loads((root / 'artifacts/source_manifest_umctf_b0_sacl_c2.json').read_text())
            self.assertIn('nnunetv2/example.py', manifest['files'])
            self.assertTrue(all(not Path(name).is_absolute() for name in manifest['files']))
            subprocess.run(['python3', str(script), 'verify'], check=True, capture_output=True)
            source.write_text('VALUE = 2\n')
            result = subprocess.run(['python3', str(script), 'verify'], capture_output=True)
            self.assertNotEqual(result.returncode, 0)

    def test_final_contracts_need_no_private_environment(self):
        for name in ('run_microbt_umctf_sacl_c2_foldall_scratch300.sh', 'continue_microbt_umctf_sacl_c2_final300_foldall.sh'):
            result = subprocess.run(['bash', str(ROOT / 'scripts' / name), '--print-contract'], check=True, text=True, capture_output=True)
            contract = json.loads(result.stdout)
            self.assertEqual(contract['fold'], 'all')
            self.assertEqual(contract['formal_epochs'], 300)

    def test_training_requires_user_configuration_before_starting(self):
        environment = os.environ.copy()
        for key in ('MICROBT_DATASET_ID', 'MICROBT_DATASET_NAME', 'nnUNet_raw', 'nnUNet_preprocessed', 'MICROBT_PF_CACHE'):
            environment.pop(key, None)
        result = subprocess.run(['bash', str(ROOT / 'scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh')], env=environment, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('MICROBT_DATASET_ID', result.stderr)


if __name__ == '__main__':
    unittest.main()
