# UnifyImmun: a unified cross-attention model for prediction of antigen binding specificity 
[![DOI image](https://zenodo.org/badge/DOI/10.5281/zenodo.14282419.svg)](https://doi.org/10.5281/zenodo.14282419)

UnifyImmun is an advanced computational model that predicts the binding specificity of antigens to both HLA and TCR molecules. By employing a unified cross-attention mechanism, UnifyImmun provides a comprehensive evaluation of antigen immunogenicity, which is crucial for the development of effective immunotherapies.

## Web Server: http://hliulab.tech/unifylmmun/

## Key features
- **Unified model**: Simultaneously predicts peptide bindings to both HLA and TCR molecules.
- **Cross-attention mechanism**: Integrates the features of peptides and HLA/TCR molecules for model interpretability.
- **Progressive training strategy**: Utilizes a two-phase progressive training to improve feature extraction and model generalizability.
- **Virtual adversarial training**: Enhances model robustness by training on perturbed data.
- **Superior performance**: Outperforms existing methods on both pHLA and pTCR prediction tasks on multiple datasets.

For inquiries or collaborations, please contact: hliu@njtech.edu.cn

## System requirements
- **Linux version**: 4.18.0-193 (Centos confirmed)
- **GPU**: NVIDIA GeForce RTX 4090 (or compatible GPUs)
- **CUDA Version**: 12.4
- **Python**: 3.10
- **PyTorch**: 2.2.1 (model implementation)

## Installation guide
>1. Clone the UnifyImmun repository

` git clone https://github.com/hliulab/unifyimmun.git`

>2. Enter UnifyImmun project folder

` cd unifyimmun/`

>3. Set up the Python environment and install the required packages
   
` pip install -r requirements.txt `

## Instructions for use
The training data for pHLA and pTCR bindings is stored in the <kbd>data</kbd> folder. The source code of UnifyImmun model, as well as the training and testing scripts, are included in the <kbd>source</kbd> folder. The trained models are stored in the <kbd>trained_model</kbd> folder.

### Input data format
The input data should be a CSV file with three columns named `tcr`, `peptide`, and `HLA`, representing the TCR CDR3 sequence, peptide sequence, and HLA sequence, respectively.

### Model training
For the convenience of sequentially running all the training steps, you can use the provided Python script named run_all_phases.py. After ensuring that the required environment and dependencies are installed, execute the following code:

`cd source`

`python run_all_phases.py`

### ESM-C + TCR-BERT stage-1 training

The new unified PLM pipeline is additive and does not replace the legacy scripts. It
uses frozen residue-level ESM-C/TCR-BERT caches, a shared peptide Adapter, two
cross-attention branches, and module-targeted FGM.

See [PLM_STAGE1.md](PLM_STAGE1.md) for checkpoint validation, cache construction,
three-seed SwanLab experiments, resume, and evaluation commands.

### Stage1 with sequence encoders

The independent `stage1_encoder` experiment replaces the three adapters with
Transformer encoders with three layers and eight heads over the same frozen PLM caches.
It supports training, resume, evaluation, task model export, and UMAP snapshots.
See [PLM_STAGE1_ENCODER.md](PLM_STAGE1_ENCODER.md) for parameters and commands.

### ESM-C + TCR-BERT stage-2 fine-tuning

Stage 2 loads a completed unified stage-1 `last` checkpoint, then progressively
unfreezes the top Peptide/HLA ESM-C and TCR-BERT blocks with discriminative
learning rates, gradient accumulation, and alternating pHLA/pTCR rounds.

See [PLM_STAGE2.md](PLM_STAGE2.md) for the exact 2A/2B unfreezing schedule,
dynamic learning rates, full-PLM checkpoint behavior, training, and resume
commands.

### Hyper-PMT heterogeneous hypergraph Stage14

Stage14 is an isolated Stage8-compatible branch that synchronously fuses
peptide, HLA, and TCR residue embeddings through a shared two-layer
Hyper-PMT encoder. See [PLM_STAGE14.md](PLM_STAGE14.md) for the model,
four-condition training commands, checkpoint contract, and benchmark evaluation.

### Stage15 pairwise Shared Hyper-PMT

Stage15 keeps the Stage1 frozen-cache PH/PT training contract while replacing
the two pairwise cross-attention modules with the Shared Hyper-PMT introduced by
Stage14. It trains all downstream modules from scratch and remains independent
of the PHT/prior/constraint branches. See [PLM_STAGE15.md](PLM_STAGE15.md).


### Stage PHT with sequence encoders

Stage PHT trains three random sequence encoders over frozen pretrained ESM-C and
TCR-BERT features, followed by the Stage Adapter PHT sequential cross-attention
path. It uses the same PHT data, balanced supervision, and two-phase training.
See [PLM_STAGE_PHT.md](PLM_STAGE_PHT.md) for configuration and commands.

### Standard MoE ablation of Stage Final

`stage_standard_moe` replaces the Full model's MoE with independent four-expert
branch pools, linear gates and sample-level Top-2 routing. It retains PH-to-PHT
features, PT fusion and consistency supervision, and uses seed 42 sequential
training. See [PLM_STAGE_STANDARD_MOE.md](PLM_STAGE_STANDARD_MOE.md) for the
comparison, matching data configuration, resume and evaluation commands.

### Model testing
Given the fine-tuned model or our trained model (saved in trained_model folder), you can evaluate it on our demo test sets using the following scripts.
>Predict HLA binding specificity using pHLA test set

`cd source`

`python HLA_test.py`

>Evaluate TCR binding specificity using pTCR test set

`cd source`

`python TCR_test.py`


### Output predicted scores
Given the fine-tuned model or our trained model (saved in trained_model folder), you can output predicted scores for the demo test sets using the following scripts.
>Output predicted scores for HLA binding specificity using pHLA test set

`cd source`

`python HLA_output_score.py`

>Output prediction scores for TCR binding specificity using pTCR test set

`cd source`

`python TCR_output_score.py`

> In our practice, the time overhead required to run these two demos above is about 2 minutes when batch_size=8192.


### Hyperparameter adjustment
If transfer the model using your custom dataset, you may need to adjust the hyperparameters within the Python scripts. Hyperparameters include learning rate, batch size, number of epochs, and other model-specific parameters.

Note: Ensure that the file paths and script names provided in the commands match those in your project directory. The source/ directory and script names like HLA_test.py and TCR_test.py are placeholders and should be replaced with the actual paths and filenames used in your implementation.

### Customizing output
To customize the output results, users can modify the parameters within each script. Detailed comments within the code provide descriptions and guidance for parameter adjustments.


## Support
For further assistance, bug reports, or to request new features, please contact us at hliu@njtech.edu.cn or open an issue on the [GitHub repository page](https://github.com/hliulab/unifyimmun).

---

Please replace the placeholder links and information with actual data when the repository is available. Ensure that the instructions are clear and that the repository contains the `requirements.txt` file with all necessary dependencies listed.
