#!/usr/bin/env python3
"""Generate a local, reproducible copy of the vendored upstream demo."""

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_NOTEBOOK = ROOT / "vendor/upstream/demo/role-probe-demo.ipynb"
UPSTREAM_HELPER = ROOT / "vendor/upstream/demo/simple_test_helpers.py"
DEMO_DIR = ROOT / "demo"
OUTPUT_NOTEBOOK = DEMO_DIR / "role-probe-demo.ipynb"
UPSTREAM_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {description} match, found {count}")
    return source.replace(old, new)


def main() -> None:
    notebook = json.loads(UPSTREAM_NOTEBOOK.read_text())

    provenance = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Local reproduction copy\n",
            "\n",
            f"Generated from upstream commit `{UPSTREAM_COMMIT}` by "
            "`scripts/prepare_demo.py`. Local changes load configuration from "
            "`.env`, allow small smoke-test settings, pin the fetched prompt URL, "
            "and make the probe split grouping configurable. Do not edit this generated "
            "notebook directly; change the generator instead.\n",
        ],
    }
    notebook["cells"].insert(0, provenance)

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))

        if "import torch\nfrom datasets import load_dataset" in source:
            source = replace_once(
                source,
                "import torch\n",
                "import os\nimport torch\nfrom dotenv import load_dotenv\n\n"
                "load_dotenv()\n"
                "SPLIT_GROUP = os.environ.get('ROLE_PROBE_SPLIT_GROUP', 'prompt_ix')\n"
                "MODEL_REVISION = os.environ.get('ROLE_PROBE_MODEL_REVISION', '6cee5e81ee83917806bbde320786a8fb61efebee')\n"
                "C4_REVISION = os.environ.get('ROLE_PROBE_C4_REVISION', 'f3b95a11ff318ce8b651afc7eb8e7bd2af469c10')\n",
                "imports",
            )

        if "CACHE_DIR = '/workspace/hf'" in source:
            source = replace_once(
                source,
                "CACHE_DIR = '/workspace/hf' # or None if uncached",
                "CACHE_DIR = os.environ.get('HF_HOME') or None",
                "model cache",
            )
            source = source.replace(
                "cache_dir = CACHE_DIR, attn_implementation",
                "cache_dir = CACHE_DIR, revision = MODEL_REVISION, attn_implementation",
            )
            source = source.replace(
                "cache_dir = CACHE_DIR, add_eos_token",
                "cache_dir = CACHE_DIR, revision = MODEL_REVISION, add_eos_token",
            )

        if "N_SAMPLES = 150" in source:
            source = replace_once(
                source,
                "N_SAMPLES = 150",
                "N_SAMPLES = int(os.environ.get('ROLE_PROBE_N_SAMPLES', '150'))",
                "sample count",
            )
            source = source.replace(
                "load_dataset('allenai/c4', 'en', split = 'validation', streaming = True)",
                "load_dataset('allenai/c4', data_dir = 'en', split = 'validation', revision = C4_REVISION, streaming = True)",
            )

        if "BATCH_SIZE = 32" in source:
            source = replace_once(
                source,
                "BATCH_SIZE = 32 # 32 works fine for an H100 with this model and seq len, but adjust as needed",
                "BATCH_SIZE = int(os.environ.get('ROLE_PROBE_BATCH_SIZE', '32')) # Reduce for a smoke test or lower-memory GPU",
                "batch size",
            )

        if "probe_sample_df = (\n    label_gptoss_content_roles(sample_df)" in source:
            source = replace_once(
                source,
                ".pipe(lambda df: df[(df['is_content'] == True) & (df['role'].notna())]) # Drop non-content tags\n)",
                ".pipe(lambda df: df[(df['is_content'] == True) & (df['role'].notna())]) # Drop non-content tags\n"
                "    .merge(input_df[['prompt_ix', 'base_seq_ix']], on='prompt_ix', how='left', validate='many_to_one')\n)",
                "base sequence merge",
            )

        if "prompt_ix_train, prompt_ix_test = cuml.train_test_split" in source:
            source = replace_once(
                source,
                "prompt_ix_train, prompt_ix_test = cuml.train_test_split(sample_df['prompt_ix'].unique(), test_size = 0.1, random_state = seed)\n"
                "    train_df = sample_df[sample_df['prompt_ix'].isin(prompt_ix_train)]\n"
                "    test_df = sample_df[sample_df['prompt_ix'].isin(prompt_ix_test)]",
                "group_train, group_test = cuml.train_test_split(sample_df[SPLIT_GROUP].unique(), test_size = 0.1, random_state = seed)\n"
                "    train_df = sample_df[sample_df[SPLIT_GROUP].isin(group_train)]\n"
                "    test_df = sample_df[sample_df[SPLIT_GROUP].isin(group_test)]",
                "configurable train/test split",
            )
            source += """

# Save the trained probes and a small accuracy table to persistent storage.
import pickle
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get('ROLE_PROBE_OUTPUT_DIR', 'outputs'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with (OUTPUT_DIR / 'role-probes.pkl').open('wb') as handle:
    pickle.dump(all_probes, handle)
pd.DataFrame(
    [{'layer_ix': item['layer_ix'], 'accuracy': float(item['acc'])} for item in all_probes]
).to_csv(OUTPUT_DIR / 'probe-accuracy.csv', index=False)
print(f'Saved probe artifacts to {OUTPUT_DIR.resolve()}')
"""

        if "OPENROUTER_API_KEY = ''" in source:
            source = replace_once(
                source,
                "OPENROUTER_API_KEY = ''",
                "OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')",
                "OpenRouter key",
            )
            source = source.replace(
                "refs/heads/master/experiments/cot-forgery-chat-evals/prompts/forgery-prompt-openai.yaml",
                f"{UPSTREAM_COMMIT}/experiments/cot-forgery-chat-evals/prompts/forgery-prompt-openai.yaml",
            )

        cell["source"] = source.splitlines(keepends=True)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n")
    shutil.copy2(UPSTREAM_HELPER, DEMO_DIR / "simple_test_helpers.py")
    (DEMO_DIR / "__init__.py").write_text(
        '"""Generated demo files for the role-probe reproduction."""\n'
    )
    print(f"Generated {OUTPUT_NOTEBOOK.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
