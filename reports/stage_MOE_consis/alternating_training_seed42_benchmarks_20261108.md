# Stage MoE Consis benchmark results — 交替训练（sequential training, PH → PT → PHT）

Checkpoint: `/home/mjp/Project/PLM/UnifyImmun/trained_model/PLM_Stage_MOE_consis/seed_42/sequential_20261106_152222/best.pt`

| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |
|---|---:|---:|---:|---:|---:|---:|---|
| ph/test_unseen_hla | 47748 | 0.850953 | 0.852870 | 0.549990 | 0.757347 | 0.066569 | hla: 0 seen in training |
| ph/test_unseen_pair | 64 | 0.722168 | 0.709733 | 0.232945 | 0.528302 | 0.066569 | Mixed / combination split |
| ph/test_unseen_peptide | 62764 | 0.943644 | 0.941599 | 0.757823 | 0.880336 | 0.066569 | peptide: 0 seen in training |
| pt/test_unseen_pair | 10274 | 0.766125 | 0.816180 | 0.494782 | 0.672592 | 0.527925 | Mixed / combination split |
| pt/test_unseen_peptide | 144800 | 0.733064 | 0.765137 | 0.385528 | 0.613409 | 0.527925 | peptide: 0 seen in training |
| pt/test_unseen_tcr | 142960 | 0.800448 | 0.827481 | 0.531572 | 0.716827 | 0.527925 | tcr: 0 seen in training |
| pht/test_unseen_hla | 19670 | 0.994190 | 0.993071 | 0.878808 | 0.940180 | 0.000005 | hla: 0 seen in training |
| pht/test_unseen_peptide | 14118 | 0.986432 | 0.987250 | 0.864989 | 0.929111 | 0.000005 | peptide: 0 seen in training |
| pht/test_unseen_tcr | 7698 | 0.999106 | 0.999200 | 0.975202 | 0.987635 | 0.000005 | tcr: 3 seen in training |
| pht/test_unseen_triple | 660 | 0.985211 | 0.984207 | 0.879823 | 0.937888 | 0.000005 | Mixed / combination split |

## pht/test_unseen_hla: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 9833 | 0.11329197599918642 |
| 010 | 1 | 0.0 |
| 100 | 1 | 0.0 |
| 111 | 9835 | 0.9875953228266395 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 19670 | 0.0018810371123538384 | 0.0006675401636731927 |
| hierarchy | applicable | 19669 | 0.001881132746962225 | 0.0006675741023667548 |
| hierarchy | positive | 9835 | 0.0007117437722419929 | 0.0002879931165893095 |
| hierarchy | negative | 9834 | 0.003050640634533252 | 0.001047193686983511 |
| pt_implication | all | 19670 | 0.01626842907981698 | 0.008210656487045035 |
| pt_implication | applicable | 19669 | 0.016269256189943567 | 0.008211073928525896 |
| pt_implication | positive | 9835 | 0.003863751906456533 | 0.00048024895125133634 |
| pt_implication | negative | 9834 | 0.02867602196461257 | 0.01594268503809427 |

## pht/test_unseen_peptide: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 7057 | 0.03457559869632989 |
| 010 | 1 | 0.0 |
| 100 | 1 | 1.0 |
| 111 | 7059 | 0.8977192236860745 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 14118 | 0.003187420314492138 | 0.000970743529966266 |
| hierarchy | applicable | 14117 | 0.003116809520436353 | 0.0009662763996504398 |
| hierarchy | positive | 7059 | 0.003683241252302026 | 0.0006832557890602401 |
| hierarchy | negative | 7058 | 0.0025502975347123833 | 0.001249337109505389 |
| pt_implication | all | 14118 | 0.0718232044198895 | 0.03624156989548522 |
| pt_implication | applicable | 14117 | 0.07182829213005595 | 0.0362441371243508 |
| pt_implication | positive | 7059 | 0.13627992633517497 | 0.06840803564004624 |
| pt_implication | negative | 7058 | 0.007367526211391329 | 0.004075681524705925 |

## pht/test_unseen_tcr: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 3847 | 0.020795425006498572 |
| 010 | 1 | 0.0 |
| 100 | 1 | 1.0 |
| 111 | 3849 | 0.9961028838659393 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 7698 | 0.0035074045206547155 | 0.0010855219732295162 |
| hierarchy | applicable | 7697 | 0.003507860205274782 | 0.0010856630050566215 |
| hierarchy | positive | 3849 | 0.005715770329955833 | 0.001625240631508004 |
| hierarchy | negative | 3848 | 0.0012993762993762994 | 0.0005459451557293422 |
| pt_implication | all | 7698 | 0.08157963107300598 | 0.02184630221603347 |
| pt_implication | applicable | 7697 | 0.08159022995972456 | 0.021849140503965917 |
| pt_implication | positive | 3849 | 0.15952195375422187 | 0.04215444141346299 |
| pt_implication | negative | 3848 | 0.0036382536382536385 | 0.0015385627491182306 |

## pht/test_unseen_triple: triplet consistency

| Class | Samples | Recall (111) / FPR (negatives) |
|---|---:|---:|
| 000 | 1 | 0.0 |
| 010 | 328 | 0.03353658536585366 |
| 100 | 1 | 1.0 |
| 111 | 330 | 0.9151515151515152 |

| Constraint | Scope | Samples | Violation rate | Mean excess |
|---|---|---:|---:|---:|
| hierarchy | all | 660 | 0.013636363636363636 | 0.0006442177295684816 |
| hierarchy | applicable | 659 | 0.013657056145675266 | 0.0006451952982021212 |
| hierarchy | positive | 330 | 0.024242424242424242 | 0.00017475785631122003 |
| hierarchy | negative | 329 | 0.00303951367781155 | 0.001117062641132204 |
| pt_implication | all | 660 | 0.012121212121212121 | 0.001976314479710929 |
| pt_implication | applicable | 332 | 0.018072289156626505 | 0.0033675394862531173 |
| pt_implication | positive | 330 | 0.01818181818181818 | 0.003387948816472833 |
| pt_implication | negative | 2 | 0.0 | 0.0 |
