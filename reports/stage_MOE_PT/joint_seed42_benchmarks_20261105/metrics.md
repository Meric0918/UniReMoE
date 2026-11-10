# Stage MOE evaluation

Checkpoint: `/home/mjp/UniPHT/trained_model/PLM_Stage_MOE_PT/joint_seed42_20261103_192126/best.pt`

| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |
|---|---:|---:|---:|---:|---:|---:|---|
| ph/test_unseen_hla | 47748 | 0.785092 | 0.759485 | 0.397615 | 0.633208 | 0.094010 | hla: 0 seen in training |
| ph/test_unseen_pair | 64 | 0.500000 | 0.472669 | -0.036886 | 0.297872 | 0.094010 | Mixed / combination split |
| ph/test_unseen_peptide | 62764 | 0.967929 | 0.966302 | 0.800591 | 0.903199 | 0.094010 | peptide: 0 seen in training |
| pt/test_unseen_pair | 10274 | 0.812605 | 0.850260 | 0.541052 | 0.720208 | 0.497879 | Mixed / combination split |
| pt/test_unseen_peptide | 144800 | 0.774155 | 0.801193 | 0.391711 | 0.580043 | 0.497879 | peptide: 0 seen in training |
| pt/test_unseen_tcr | 142960 | 0.914119 | 0.912908 | 0.675198 | 0.838152 | 0.497879 | tcr: 5 seen in training |
| pht/test_unseen_hla | 19670 | 0.795538 | 0.641364 | -0.215719 | 0.005911 | 0.000000 | hla: 0 seen in training |
| pht/test_unseen_peptide | 14118 | 0.990289 | 0.989761 | 0.884065 | 0.941579 | 0.000000 | peptide: 0 seen in training |
| pht/test_unseen_tcr | 7698 | 0.999435 | 0.999456 | 0.969125 | 0.984604 | 0.000000 | tcr: 4 seen in training |
| pht/test_unseen_triple | 660 | 0.576244 | 0.719735 | 0.386891 | 0.541322 | 0.000000 | Mixed / combination split |
