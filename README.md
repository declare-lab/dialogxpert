# [AAAI26] DialogXpert: Driving Intelligent and Emotion-Aware Conversations Through Online Value-Based Reinforcement Learning with LLM Priors

![Description](images/architecture.png)

<p align="center">
   <a href="https://ojs.aaai.org/index.php/AAAI/article/view/40244" target="_blank">Official Paper (AAAI26 Proceedings)</a> &nbsp; | &nbsp;
  <a href="https://drive.google.com/file/d/1ktTEJomjhGOumFx79QQhDnDGQJ4qgRkO/view?usp=sharing" target="_blank">Extended Paper (Appendix)</a>
</p>

**NOTE**: The extended paper is a self-hosted extended version, including appendices and additional experiments. **_The official AAAI paper should be cited for publication purposes._**

## Introduction

Proactive dialogue systems require efficient action selection under large action spaces and evolving conversational dynamics. Hence, this repository implements a proactive dialogue planning framework that integrates:
- **LLM Priors**: to generate a rational set of action candidates.
- **Emotion Trajectory Modeling**: to track conversational dynamics.
- **Q-learning**: to select the optimal action from the subset of action candidates.

## Highlights

- **Action space reduction**: Leverages LLM priors to select top-k candidate actions, reducing the effective action space and improving sample efficiency without requiring fine-tuning.

- **Emotion-aware planning**: Introduces the integration of emotion trajectories into goal-driven dialogue, enabling context-aware and emotionally informed action selection.

- **Deep Q-learning**: Unlike existing state-of-the-art (SOTA) approaches that rely on MCTS or diffusion-based planning, our method uses Deep Q-learning over a reduced action space. This significantly improves efficiency, reducing LLM calls from 30 to 4 while maintaining competitive performance.

- **Multi-dataset approach**: Evaluated across multiple dialogue domains, including emotional support (ESConv, ExTES), negotiation (CB), tutoring (CIMA), and persuasion (P4G). See the dataset section for details.

## Architecture Code Mapping

- LLM Priors: [`env.py`](https://github.com/declare-lab/dialogxpert/blob/master/env.py#L190) (Full function get_prior_actions_llm)
- Emotion trajectory:  [`env.py`](https://github.com/declare-lab/dialogxpert/blob/master/env.py#L427) (Storage done on Line 429)
- Action selection: [`train_model.py`](https://github.com/declare-lab/dialogxpert/blob/master/train_model.py#L180) (Lines 180-196)
- Q-learning: 
    - Buffer storage: [`train_model.py`](https://github.com/declare-lab/dialogxpert/blob/master/train_model.py#L228) (Done for every conversation turn)
    - Training: [`train_model.py`](https://github.com/declare-lab/dialogxpert/blob/master/train_model.py#L255) (Done every epoch)

## Datasets

The following datasets have been utilized: 

- [ESConv](https://aclanthology.org/2021.acl-long.269/): Emotional support conversation.
- [CIMA](https://aclanthology.org/2020.bea-1.5/): Tutoring dialogue dataset.
- [CraigstlistBargain (CB)](https://aclanthology.org/D18-1256/): Neogitation dialogues
- [ExTES](https://aclanthology.org/2024.acl-long.611/): Emotional support conversation (similar to ESConv).
- [P4G](https://aclanthology.org/P19-1566/): Persuasion dialogues.

All datasets (training, validation, and test splits) are included in this repository under the `data/` directory.

## Implementation

### Quick Start

To train the model on a specific dataset, run `python train_model.py --data_name <dataset_name>`

**NOTE**: Dataset-specific prompt configuration is required before training (see below).

### Downloading the LLM weights

Download the LLM model weights locally:

1. Set the desired model name in `download_llm_weights` (Lines 4-5): https://github.com/declare-lab/dialogxpert/blob/master/download_llm_weights.py#L4-5

2. Run `python download_llm_weights.py`

NOTES: 

- Update the `repo_id` in `download_llm_weights.py` to select the desired model.

- Ensure that you are logged into huggingface with the appropriate access token.

### Dataset-Specific Configuration

Due to differences across dialogue domains, prompt and roleplay functions must be configured manually for each dataset.

For a selected dataset `<dataset_name>`, update the following components:

- **Policy Prompt**: Replace with `{dataset_name}_prompt` in: https://github.com/declare-lab/dialogxpert/blob/master/env.py#L196

- **Roleplay Functions**: Replace with {dataset_name}_roleplay:

    - _System roleplay_: https://github.com/declare-lab/dialogxpert/blob/master/env.py#L418

    - _User roleplay_: https://github.com/declare-lab/dialogxpert/blob/master/env.py#L435

    - _Critic_: https://github.com/declare-lab/dialogxpert/blob/master/env.py#L326

    - Example: For the ExTES dataset, replace all relevant functions with `extes_*` variants.

All the available prompt functions can be found in: https://github.com/declare-lab/dialogxpert/blob/master/qwen_prompts.py

### Training

After completing the configuration, run `python train_model.py --data_name <dataset_name>`

> Example: `python train_model.py --data_name ExTES`

Evaluation is performed automatically during training itself:

- Per-epoch metrics: Average turns, success rate (https://github.com/declare-lab/dialogxpert/blob/master/train_model.py#L24)
- Self-play evaluation (turn-level): https://github.com/declare-lab/dialogxpert/blob/master/env.py#L443

### Extra information

Results may vary slightly due to stochastic training, Q-learning, LLM sampling, and hardware differences. We recommend fixing random seeds and using consistent environments for reproducibility.

The current implementation uses manual prompt configuration to support flexibility across diverse dialogue domains. While this requires minor manual setup, it enables flexible adaptation across multiple dialogue domains. Future updates will include automated dataset-specific configuration.

## Citation

If you find our work useful, please cite

```
@inproceedings{rakib2026dialogxpert,
  title={DialogXpert: Driving Intelligent and Emotion-Aware Conversations Through Online Value-Based Reinforcement Learning with LLM Priors},
  author={Abdur Rakib, Tazeek Bin and Mehrish, Ambuj and Soon, Lay-Ki and Lim, Wern Han and Poria, Soujanya},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026},
  doi={10.1609/aaai.v40i36.40244},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/40244}
}
```

## Repository Credits

The following repositories are given credit for their open-source code utilization

```
- PPDPP: https://github.com/dengyang17/PPDPP/tree/main
- DPDP: https://github.com/cs-holder/DPDP/tree/main
- RL-LLM: https://github.com/yanxue7/RL-LLM-Prior/tree/main
```

