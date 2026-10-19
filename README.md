# cudasynth

Exact Clifford+T circuit synthesis by GPU branch-and-bound: enumerate gate sequences level by level, prune every
prefix whose remaining work provably exceeds the depth budget, and either find a circuit or prove that none of
the given depth exists. Targets: unitaries, state maps, and repeat-until-success (RUS) circuits without
measurement. See [docs/PLAN.md](docs/PLAN.md).

Status: early development. The CPU reference prototype lives in milq (`misc/prototypes/prune_probe.py`).
