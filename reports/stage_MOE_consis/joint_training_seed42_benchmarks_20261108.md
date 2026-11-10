# Stage MoE Consis benchmark results — 联合训练（joint training）

Checkpoint: `/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage_MOE_consis/seed_42/joint_20261106_152222/best.pt`

| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |
|---|---:|---:|---:|---:|---:|---:|---|
| ph/test_unseen_hla | 47748 | 0.883870 | 0.866690 | 0.622298 | 0.807069 | 0.103749 | hla: 0 seen in training |
| ph/test_unseen_pair | 64 | 0.651367 | 0.624872 | 0.134840 | 0.461538 | 0.103749 | Mixed / combination split |
| ph/test_unseen_peptide | 62764 | 0.964462 | 0.959570 | 0.819876 | 0.912066 | 0.103749 | peptide: 0 seen in training |
| pt/test_unseen_pair | 10274 | 0.744122 | 0.825363 | 0.549295 | 0.727396 | 0.505340 | Mixed / combination split |
| pt/test_unseen_peptide | 144800 | 0.724154 | 0.776211 | 0.396014 | 0.582743 | 0.505340 | peptide: 0 seen in training |
| pt/test_unseen_tcr | 142960 | 0.890193 | 0.896733 | 0.638176 | 0.808144 | 0.505340 | tcr: 0 seen in training |
| pht/test_unseen_hla | 19670 | 0.979516 | 0.965378 | 0.863066 | 0.932815 | 0.000000 | hla: 0 seen in training |
| pht/test_unseen_peptide | 14118 | 0.991257 | 0.991092 | 0.894284 | 0.947494 | 0.000000 | peptide: 0 seen in training |
| pht/test_unseen_tcr | 7698 | 0.999363 | 0.999364 | 0.968351 | 0.984221 | 0.000000 | tcr: 3 seen in training |
| pht/test_unseen_triple | 660 | 0.774587 | 0.825998 | 0.436139 | 0.648551 | 0.000000 | Mixed / combination split |

## pht/test_unseen_hla: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 9833 | 0.10312213973355029 |
| 010 | 1 | 0.0 |
| 100 | 1 | 0.0 |
| 111 | 9835 | 0.9642094560244027 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 19670 | 0.4607015760040671 | 0.04442241955634653 |
| hierarchy | applicable | 19669 | 0.4607249987289644 | 0.04442467805548509 |
| hierarchy | positive | 9835 | 0.9156075241484494 | 0.08687321938732735 |
| hierarchy | negative | 9834 | 0.005796217205613179 | 0.0019718202154732915 |
| pt_implication | all | 19670 | 0.015607524148449415 | 0.009790740841972465 |
| pt_implication | applicable | 19669 | 0.015608317657227108 | 0.009791238617194489 |
| pt_implication | positive | 9835 | 0.0035587188612099642 | 0.001506316219136898 |
| pt_implication | negative | 9834 | 0.027659141753101486 | 0.018077003492616164 |

## pht/test_unseen_peptide: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 7057 | 0.060507297718577296 |
| 010 | 1 | 0.0 |
| 100 | 1 | 1.0 |
| 111 | 7059 | 0.954809463096756 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 14118 | 0.0048165462530103415 | 0.001241240627190488 |
| hierarchy | applicable | 14117 | 0.004816887440674364 | 0.0012413285524314876 |
| hierarchy | positive | 7059 | 0.006233177503895736 | 0.0012215966101641364 |
| hierarchy | negative | 7058 | 0.003400396712949844 | 0.0012610632903834926 |
| pt_implication | all | 14118 | 0.04887377815554611 | 0.02865918771579296 |
| pt_implication | applicable | 14117 | 0.048877240206842816 | 0.028661217834636606 |
| pt_implication | positive | 7059 | 0.0872644850545403 | 0.050419321620036144 |
| pt_implication | negative | 7058 | 0.010484556531595354 | 0.006900031291545882 |

## pht/test_unseen_tcr: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 3847 | 0.0283337665713543 |
| 010 | 1 | 0.0 |
| 100 | 1 | 1.0 |
| 111 | 3849 | 0.9966224993504806 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 7698 | 0.005455962587685113 | 0.0016566515506457379 |
| hierarchy | applicable | 7697 | 0.005456671430427439 | 0.0016568667840549422 |
| hierarchy | positive | 3849 | 0.00857365549493375 | 0.0021222914028288515 |
| hierarchy | negative | 3848 | 0.002338877338877339 | 0.001191321212937281 |
| pt_implication | all | 7698 | 0.07261626396466615 | 0.028726741212531624 |
| pt_implication | applicable | 7697 | 0.07262569832402235 | 0.028730473412247427 |
| pt_implication | positive | 3849 | 0.14211483502208366 | 0.055559876846933164 |
| pt_implication | negative | 3848 | 0.0031185031185031187 | 0.001894097679371853 |

## pht/test_unseen_triple: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 1 | 0.0 |
| 010 | 328 | 0.12804878048780488 |
| 100 | 1 | 1.0 |
| 111 | 330 | 0.5424242424242425 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 660 | 0.26666666666666666 | 0.024772106105828494 |
| hierarchy | applicable | 659 | 0.26707132018209406 | 0.024809696555154484 |
| hierarchy | positive | 330 | 0.5333333333333333 | 0.04954421221165699 |
| hierarchy | negative | 329 | 0.0 | 0.0 |
| pt_implication | all | 660 | 0.006060606060606061 | 0.002973723390675855 |
| pt_implication | applicable | 332 | 0.006024096385542169 | 0.0012272814311176897 |
| pt_implication | positive | 330 | 0.006060606060606061 | 0.001234719500397191 |
| pt_implication | negative | 2 | 0.0 | 0.0 |
