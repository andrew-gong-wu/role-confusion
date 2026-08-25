# Vendored upstream snapshot

- Repository: `https://github.com/role-confusion/prompt-injection-as-role-confusion`
- Branch observed: `master`
- Commit: `ec333c40fd43fe991e1ebf66765051b6d7e35784`
- Commit date: 2026-05-31
- Commit subject: `Cleanup`
- Retrieved: 2026-08-25

The files in this directory are preserved without local edits. See the
upstream `LICENSE.md` for their MIT license. Run `scripts/prepare_demo.py` to
create the locally adapted copy in `demo/`.

At this snapshot, `demo/role-probe-demo.ipynb` explicitly requires
Transformers major version 5, while `setup_python.sh` installs
`transformers==4.57.5`. The local H100 setup follows the notebook requirement
and records the installed version for reproducibility.

