import ast
import unittest
from pathlib import Path


class TestRunTrainingEnvironmentContract(unittest.TestCase):
    def test_torchinductor_environment_value_is_a_string(self):
        source_path = Path(__file__).resolve().parents[2] / "run" / "run_training.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == "TORCHINDUCTOR_COMPILE_THREADS"
                for target in node.targets
            )
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].value.value, "1")


if __name__ == "__main__":
    unittest.main()
