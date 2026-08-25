# Reproducing Role-Confusion Linear Probes

This repository is a workspace for reproducing the linear role-probe results from **Prompt Injection as Role Confusion**. The core experiment holds text content fixed, places copies of that content in different chat roles, extracts token-level hidden states, and trains linear classifiers to predict the enclosing role. The probes can then measure whether untrusted text is represented as user-like, assistant-like, or chain-of-thought-like despite its actual role tag.

## Primary references

- [Accessible LessWrong write-up](https://www.lesswrong.com/posts/d8xDGzCEYE639qqEv/a-mechanistic-explanation-of-prompt-injection-and-why-you)
- [Paper: *Prompt Injection as Role Confusion*](https://arxiv.org/abs/2603.12277)
- [Authors' full reproduction repository](https://github.com/role-confusion/prompt-injection-as-role-confusion)
- [Authors' role-probe demonstration notebook](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/master/demo/role-probe-demo.ipynb)

This repository does not currently vendor the authors' code or data. Record the upstream commit used for each run so that results remain traceable as their repository evolves.

## Reproduction target

The initial target is the lightweight `gpt-oss-20b` demonstration:

1. Sample neutral text from C4 and Dolma 3.
2. Render identical samples in the `system`, `user`, `analysis`/CoT, `assistant`, and `tool` roles.
3. Extract token activations at several transformer layers.
4. Remove chat-template tokens and label only content tokens by their true enclosing role.
5. Split by prompt (not by token) and fit L2-regularized multinomial logistic-regression probes.
6. Report held-out accuracy by layer and apply the probes to controlled tagged, untagged, and role-mismatched examples.

The demo uses 150 source sequences, a maximum sequence length of 512, layers `0, 4, 8, 12, 16, 20`, seed `123`, and a four-way probe over `system`, `user`, `cot`, and `assistant`. Treat these as the baseline settings before running ablations.

## Requirements

- Python 3.12+
- A Linux machine with an NVIDIA CUDA GPU
- CUDA 12.8 for the authors' documented environment
- Enough GPU memory for `openai/gpt-oss-20b` plus activation extraction
- Hugging Face access for the model and streamed datasets
- Optional OpenRouter or OpenAI credentials for API-backed examples

The paper's full workflows were run on an NVIDIA H200. The demonstration says a batch size of 32 works on an H100, but smaller GPUs may require a much smaller batch size and may still run out of memory. CPU-only and Apple Silicon execution are not supported by the supplied RAPIDS/cuML workflow.

## Setup

Create and activate an environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Use the authors' current dependency instructions as the source of truth. Their demo requires PyTorch, Transformers 5+, Datasets, pandas, NumPy, tqdm, scikit-learn, Plotly, CuPy, and RAPIDS cuML. PyTorch, CuPy, and RAPIDS wheels must match the machine's CUDA version; follow the [RAPIDS installation guide](https://docs.rapids.ai/install/) instead of installing arbitrary wheels.

Copy or download these two upstream files while preserving the `demo/` layout:

```text
demo/
├── role-probe-demo.ipynb
└── simple_test_helpers.py
```

Then start Jupyter from the repository root so that `demo.simple_test_helpers` is importable:

```bash
python -m pip install jupyterlab python-dotenv
jupyter lab demo/role-probe-demo.ipynb
```

## Credentials

Put local credentials in `.env`; the file is ignored by Git. The prepared variables are:

```dotenv
HF_TOKEN=
OPENROUTER_API_KEY=
OPENAI_API_KEY=
WANDB_API_KEY=
```

Load them in Python rather than pasting secrets into a notebook:

```python
import os
from dotenv import load_dotenv

load_dotenv()
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
```

The upstream demo currently includes a cell with an empty, notebook-local OpenRouter key variable. Replace that assignment with environment loading before using the API-backed example. Never commit executed notebook output containing credentials or sensitive prompts.

## Experimental discipline

For every run, record:

- upstream Git commit;
- model revision and tokenizer revision;
- package lock or environment export;
- GPU model, driver, and CUDA versions;
- random seed, sample count, sequence length, batch size, probed layers, and probe hyperparameters;
- dataset names and revisions;
- held-out accuracy and per-class metrics for each layer.

Keep the train/test split grouped by source prompt. A token-level random split leaks near-duplicate content across roles and will overstate probe performance. Also distinguish probe *decodability* from causal evidence: high linear accuracy shows role information is available in the representation, not by itself that the model uses that direction to decide whether to follow an instruction.

## Suggested milestones

- [ ] Run the upstream demonstration unchanged and save environment metadata.
- [ ] Reproduce held-out role-classification accuracy across layers.
- [ ] Reproduce the gardening conversation under correct, absent, and conflicting role tags.
- [ ] Compare CoTness/Userness against the figures and qualitative trends in the paper.
- [ ] Add multiple seeds and confidence intervals.
- [ ] Only then move to the authors' full role-space and prompt-injection workflows.

## Safety and cost

Some upstream experiments use jailbreak prompts and prompt-injection attacks. Run them only against models and systems you are authorized to test, isolate credentials and tools, and avoid connecting experimental agents to real secrets or consequential services. Hosted-model experiments can incur API charges; set provider spending limits before large sweeps.

## Citation

```bibtex
@inproceedings{ye2026promptinjectionroleconfusion,
  title     = {Prompt Injection as Role Confusion},
  author    = {Ye, Charles and Cui, Jasmine and Hadfield-Menell, Dylan},
  booktitle = {International Conference on Machine Learning},
  year      = {2026},
  url       = {https://arxiv.org/abs/2603.12277}
}
```
