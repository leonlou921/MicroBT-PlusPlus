bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 996"
bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 997"
bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 998"
bsub -q gpu.legacy -gpu num=1:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test.sh 999"


bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 996"
bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 997"
bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 998"
bsub -q gpu.legacy -gpu num=2:j_exclusive=yes:gmem=1G -L /bin/bash ". /path/to/example && cd /path/to/example && export nnUNet_keep_files_open=True && . nnunetv2/tests/integration_tests/run_integration_test_trainingOnly_DDP.sh 999"
