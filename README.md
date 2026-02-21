# TCRMD — TCR-pMHC Ternary Complex Molecular Dynamics Pipeline

TCRMD is a fully automated, four-stage pipeline for modelling the T-cell receptor (TCR) /
peptide–MHC (pMHC) ternary complex and extracting binding-relevant observables from a
short molecular dynamics simulation.

```
Sequences (JSON)
      │
      ▼
┌─────────────────────────────┐
│  Stage 1 · Sequence-to-     │  CDR extraction → Boltz-1 inference
│           Structure         │  → template superimposition
└────────────────┬────────────┘
                 │ predicted_complex.pdb
                 ▼
┌─────────────────────────────┐
│  Stage 2 · System           │  PDBFixer cleaning → PROPKA protonation
│           Preparation       │  → OpenMM solvation (TIP3P-FB, 0.15 M NaCl)
└────────────────┬────────────┘
                 │ solvated.pdb
                 ▼
┌─────────────────────────────┐
│  Stage 3 · Simulation       │  Energy minimisation → NVT/NPT equilibration
│                             │  (AMBER14SB, HMR 4 fs timestep, CPU)
└────────────────┬────────────┘
                 │ trajectory.dcd
                 ▼
┌─────────────────────────────┐
│  Stage 4 · Inference &      │  RMSF · H-bond map · Contact map
│           Analytics         │  · Buried surface area · COM distance
└─────────────────────────────┘
```

## Quick-start

### Prerequisites

Each stage has its own conda environment under `envs/` and a matching Docker image under
`docker/`.  Install the full set of dependencies with:

```bash
conda env create -f envs/sequence_to_structure.yaml   # Stage 1
conda env create -f envs/system_preparation.yaml      # Stage 2
conda env create -f envs/simulate.yaml                # Stage 3
conda env create -f envs/inference_analytics.yaml     # Stage 4
```

Alternatively, build the Docker images:

```bash
docker build -t tcrmd-s2s       -f docker/sequence_to_structure/Dockerfile .
docker build -t tcrmd-sysprep   -f docker/system_preparation/Dockerfile    .
docker build -t tcrmd-simulate  -f docker/simulate/Dockerfile               .
docker build -t tcrmd-analytics -f docker/inference_analytics/Dockerfile    .
```

### Running the full pipeline

```bash
python run_pipeline.py \
    --sequences  tests/data/sequences.json \
    --template   my_template.pdb \
    --checkpoint boltz1.ckpt \
    --output     results/
```

Key CLI flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--sequences` | *(required)* | JSON file mapping chain keys to amino-acid sequences |
| `--template` | `None` | Reference PDB for MHC backbone superimposition (optional) |
| `--checkpoint` | `boltz1.ckpt` | Boltz model checkpoint |
| `--output` | `results/` | Top-level output directory |
| `--num-steps` | `50000` | MD equilibration steps (≈ 200 ps with default HMR 4 fs/step) |
| `--temperature` | `300.0` | Simulation temperature in K |
| `--platform` | `CPU` | OpenMM platform (`CPU`, `CUDA`, `OpenCL`) |
| `--ph` | `7.4` | pH for protonation-state assignment |
| `--no-hmr` | off | Disable Hydrogen Mass Repartitioning |
| `--skip-inference` | off | Skip Boltz and use `--template` directly |

All intermediate files are written to `<output>/intermediate/`.  Final analytics are
written to `<output>/analytics/`.  A JSON summary of all output paths is written to
`<output>/pipeline_summary.json`.

## Sequence input format

```json
{
  "alpha":     "METDTLLL...",
  "beta":      "MGAGVSQS...",
  "peptide":   "SIINFEKL",
  "mhc_alpha": "GSHSMRYF...",
  "mhc_beta":  "MIQRTPKI..."
}
```

Chain keys map to PDB chain IDs: `alpha`→A, `beta`→B, `mhc_alpha`→C, `mhc_beta`→D,
`peptide`→P.

## Documentation

Detailed, step-by-step vignettes live in the `docs/` folder:

| Vignette | Stage | Topics |
|----------|-------|--------|
| [`docs/vignette_01_sequence_to_structure.md`](docs/vignette_01_sequence_to_structure.md) | 1 | CDR extraction, Boltz input formatting, structural inference, template alignment |
| [`docs/vignette_02_system_preparation.md`](docs/vignette_02_system_preparation.md) | 2 | PDB cleaning, protonation-state assignment, solvation |
| [`docs/vignette_03_simulation.md`](docs/vignette_03_simulation.md) | 3 | Force-field setup, energy minimisation, NVT/NPT equilibration |
| [`docs/vignette_04_inference_analytics.md`](docs/vignette_04_inference_analytics.md) | 4 | RMSF, hydrogen-bond map, contact map, BSA, COM distance |
| [`docs/hpc_apptainer_guide.md`](docs/hpc_apptainer_guide.md) | all | Building Apptainer images, serial HPC execution, SLURM scripts |

## Running tests

```bash
pip install -e ".[test]"
pytest -v
```

Each stage has its own test module under `tests/` and a GitHub Actions workflow under
`.github/workflows/` that builds and tests the matching Docker image.

## Project layout

```
tcrmd/
├── sequence_to_structure.py   # Stage 1 – CDR extraction, Boltz, alignment
├── system_preparation.py      # Stage 2 – PDBFixer, PROPKA, solvation
├── simulate.py                # Stage 3 – OpenMM MD
└── inference_analytics.py     # Stage 4 – RMSF, H-bonds, contacts, BSA, COM

run_pipeline.py                # End-to-end orchestration script
tests/                         # pytest test suite (one file per stage)
envs/                          # Conda environment YAML files
docker/                        # Stage-specific Dockerfiles
docs/                          # Detailed vignettes and HPC guide
```

## Naming conventions

| Scope | Convention |
|-------|-----------|
| Exported functions | PascalCase (`ExtractCDRLoops`, `RunSimulation`) |
| Public arguments | camelCase (`chainType`, `platformName`) |
| Internal variables | snake_case |
| CLI flags | kebab-case (`--num-steps`, `--skip-inference`) |

## License

MIT — see [LICENSE](LICENSE).