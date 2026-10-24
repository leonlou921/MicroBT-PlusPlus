"""Choose augmentation workers without a machine-name configuration table."""
import os


def get_allowed_n_proc_DA():
    """Respect the explicit worker setting, bounded by available CPU count."""
    use_this = int(os.environ.get('nnUNet_n_proc_DA', '12'))
    return min(use_this, os.cpu_count() or 1)
