# Stage MOE evaluation

Checkpoint: `/home/mjp/UniPHT/trained_model/PLM_Stage_MOE_PT/sequential_seed42_20261103_202153/best.pt`

| Dataset | N | AUROC | AUPRC | MCC | F1 | Threshold | Cold audit |
|---|---:|---:|---:|---:|---:|---:|---|
| ph/test_unseen_hla | 47748 | 0.831504 | 0.813522 | 0.509550 | 0.741761 | 0.163445 | hla: 0 seen in training |
| ph/test_unseen_pair | 64 | 0.630371 | 0.643402 | 0.139010 | 0.440000 | 0.163445 | Mixed / combination split |
| ph/test_unseen_peptide | 62764 | 0.945072 | 0.944384 | 0.742896 | 0.875958 | 0.163445 | peptide: 0 seen in training |
| pt/test_unseen_pair | 10274 | 0.781982 | 0.838997 | 0.568558 | 0.739535 | 0.579876 | Mixed / combination split |
| pt/test_unseen_peptide | 144800 | 0.762624 | 0.797722 | 0.417727 | 0.596495 | 0.579876 | peptide: 0 seen in training |
| pt/test_unseen_tcr | 142960 | 0.911581 | 0.911785 | 0.655271 | 0.817982 | 0.579876 | tcr: 5 seen in training |
| pht/test_unseen_hla | 19670 | 0.864497 | 0.748353 | 0.174990 | 0.344764 | 0.009126 | hla: 0 seen in training |
| pht/test_unseen_peptide | 14118 | 0.986442 | 0.987204 | 0.872767 | 0.933645 | 0.009126 | peptide: 0 seen in training |
| pht/test_unseen_tcr | 7698 | 0.999492 | 0.999517 | 0.978799 | 0.989425 | 0.009126 | tcr: 4 seen in training |
| pht/test_unseen_triple | 660 | 0.917066 | 0.913160 | 0.520911 | 0.675472 | 0.009126 | Mixed / combination split |
