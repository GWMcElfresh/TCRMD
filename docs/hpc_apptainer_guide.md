# HPC & Apptainer Guide

This guide describes how to run the TCRMD pipeline on an HPC cluster using
[Apptainer](https://apptainer.org) (formerly Singularity), executing the four
stages in serial inside a single allocation or across separate jobs.

---

## Why Apptainer?

Most HPC clusters prohibit Docker for security reasons but support Apptainer,
which builds portable, read-only Singularity Image Format (`.sif`) container
images.  Each TCRMD stage has an Apptainer definition file (`.def`) under
`apptainer/` that installs the required conda environment from scratch.

---

## Step 1 — Build Apptainer images

Choose the approach that matches your HPC environment.

### Option A: Build directly on HPC (internet access required)

This is the recommended approach for clusters that allow outbound internet
connections from compute or build nodes.  No Docker installation is needed.

```bash
cd /scratch/$USER/tcrmd   # or wherever you cloned the repo

# Build all four images
apptainer build tcrmd-s2s.sif       apptainer/sequence_to_structure.def
apptainer build tcrmd-sysprep.sif   apptainer/system_preparation.def
apptainer build tcrmd-simulate.sif  apptainer/simulate.def
apptainer build tcrmd-analytics.sif apptainer/inference_analytics.def
```

If your cluster requires `--fakeroot` or `--sandbox` instead of root:

```bash
# fakeroot (most common on multi-user HPC; requires sysadmin to enable)
apptainer build --fakeroot tcrmd-simulate.sif apptainer/simulate.def

# Writable sandbox → convert to .sif when done
apptainer build --sandbox tcrmd-simulate/ apptainer/simulate.def
apptainer build tcrmd-simulate.sif tcrmd-simulate/
```

Each `.def` file:
- Bootstraps from `mambaorg/micromamba:1.5.8` (pulled from Docker Hub)
- Installs the matching conda environment from `envs/<stage>.yaml`
- Copies `tcrmd/` and `tests/` into the image
- Includes a `%test` section that can be invoked with `apptainer test`

```bash
# Verify the image after building
apptainer test tcrmd-simulate.sif
```

### Option B: Build on a local machine with Docker and transfer

Use this approach on air-gapped clusters where build nodes have no internet
access.

```bash
# On your local machine (requires Docker)
docker build -t tcrmd-s2s       -f docker/sequence_to_structure/Dockerfile .
docker build -t tcrmd-sysprep   -f docker/system_preparation/Dockerfile    .
docker build -t tcrmd-simulate  -f docker/simulate/Dockerfile               .
docker build -t tcrmd-analytics -f docker/inference_analytics/Dockerfile    .

apptainer build tcrmd-s2s.sif       docker-daemon://tcrmd-s2s:latest
apptainer build tcrmd-sysprep.sif   docker-daemon://tcrmd-sysprep:latest
apptainer build tcrmd-simulate.sif  docker-daemon://tcrmd-simulate:latest
apptainer build tcrmd-analytics.sif docker-daemon://tcrmd-analytics:latest

# Transfer to HPC
scp tcrmd-*.sif user@hpc.example.edu:/scratch/user/tcrmd/
```

---

## Step 2 — Verify images on the cluster

```bash
# Smoke-test: run the sequence-to-structure unit tests
apptainer exec \
    --bind /scratch/user/tcrmd:/workspace \
    /scratch/user/tcrmd/tcrmd-s2s.sif \
    python -m pytest /workspace/tests/test_sequence_to_structure.py -v --tb=short
```

---

## Step 3 — Serial execution scripts

The recommended HPC workflow runs each stage in its own job step so that
you can inspect intermediate outputs before proceeding.

### 3a · Stage 1 — Sequence-to-Structure

```bash
#!/usr/bin/env bash
# stage1_s2s.sh
#SBATCH --job-name=tcrmd_s2s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s2s_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-s2s.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
import json
from tcrmd.sequence_to_structure import (
    ExtractCDRLoops,
    FormatBoltzInput,
    RunBoltzInference,
    AlignToTemplate,
)

with open("/workspace/sequences.json") as fh:
    sequences = json.load(fh)

for chain in ("alpha", "beta"):
    cdrs = ExtractCDRLoops(sequences, chainType=chain)
    print(f"CDR loops ({chain}):", cdrs)

boltz_yaml = FormatBoltzInput(
    sequences,
    templatePdbPath="/workspace/template.pdb",
    outputDir="/workspace/intermediate",
)

predicted_pdb = RunBoltzInference(
    boltz_yaml,
    checkpointPath="/workspace/boltz1.ckpt",
    outputDir="/workspace/intermediate",
)

rmsd = AlignToTemplate(
    predicted_pdb,
    "/workspace/template.pdb",
    "/workspace/intermediate/aligned_complex.pdb",
)
print(f"Alignment RMSD: {rmsd:.3f} Å")
PYTHON
```

### 3b · Stage 2 — System Preparation

```bash
#!/usr/bin/env bash
# stage2_sysprep.sh
#SBATCH --job-name=tcrmd_sysprep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sysprep_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-sysprep.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
from tcrmd.system_preparation import CleanPDB, AssignProtonationStates, SolvateSystem

CleanPDB(
    "/workspace/intermediate/aligned_complex.pdb",
    "/workspace/intermediate/cleaned.pdb",
    ph=7.4,
)

try:
    AssignProtonationStates(
        "/workspace/intermediate/cleaned.pdb",
        "/workspace/intermediate/protonated.pdb",
        ph=7.4,
    )
    ready = "/workspace/intermediate/protonated.pdb"
except ImportError:
    print("propka unavailable; using cleaned PDB")
    ready = "/workspace/intermediate/cleaned.pdb"

SolvateSystem(
    ready,
    "/workspace/intermediate/solvated.pdb",
    padding=1.0,
    ionicStrength=0.15,
)
PYTHON
```

### 3c · Stage 3 — Simulation

```bash
#!/usr/bin/env bash
# stage3_simulate.sh
#SBATCH --job-name=tcrmd_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/simulate_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-simulate.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
import json
from tcrmd.simulate import RunSimulation

results = RunSimulation(
    pdbPath="/workspace/intermediate/solvated.pdb",
    outputDir="/workspace/simulation",
    numSteps=500_000,      # 500 000 × 4 fs = 2 ns (HMR)
    temperature=300.0,
    hmr=True,
    platformName="CPU",
    minimizeFirst=True,
)

with open("/workspace/simulation/sim_results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print("Minimised PDB :", results["minimized_pdb"])
print("Trajectory    :", results["trajectory"])
print("Energy (kJ/mol):", results["potential_energy"])
PYTHON
```

> **GPU clusters:** Replace `platformName="CPU"` with `"CUDA"` and add
> `#SBATCH --gres=gpu:1` to the SLURM header.  No other code changes are needed.

### 3d · Stage 4 — Inference & Analytics

```bash
#!/usr/bin/env bash
# stage4_analytics.sh
#SBATCH --job-name=tcrmd_analytics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analytics_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-analytics.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
import os
from tcrmd.inference_analytics import (
    ComputeRMSF,
    ComputeHydrogenBondMap,
    ComputeContactMap,
    ComputeBuriedSurfaceArea,
    ComputeCOMDistance,
)

topo = "/workspace/simulation/minimized.pdb"
traj = "/workspace/simulation/trajectory.dcd"
odir = "/workspace/analytics"
os.makedirs(odir, exist_ok=True)

ComputeRMSF(topo, traj, outputPath=f"{odir}/rmsf.csv")

ComputeHydrogenBondMap(
    topo, traj,
    donorSelection="segid A or segid B",
    acceptorSelection="segid P",
    outputPath=f"{odir}/hbond_map.npy",
)

ComputeContactMap(
    topo, traj,
    selectionA="segid A or segid B",
    selectionB="segid P",
    outputPath=f"{odir}/contact_map.npy",
)

ComputeBuriedSurfaceArea(
    topo, traj,
    selectionA="segid A or segid B",
    selectionB="segid C or segid D or segid P",
    outputPath=f"{odir}/bsa.csv",
)

ComputeCOMDistance(
    topo, traj,
    selectionA="segid A or segid B",
    selectionB="segid C or segid D",
    outputPath=f"{odir}/com_distance.csv",
)

print("Analytics complete. Files written to", odir)
PYTHON
```

---

## Step 4 — Submitting the serial pipeline

Chain the four jobs with `--dependency=afterok` so each stage only starts
when the preceding stage succeeds:

```bash
JOB1=$(sbatch --parsable stage1_s2s.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 stage2_sysprep.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 stage3_simulate.sh)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 stage4_analytics.sh)

echo "Jobs submitted: $JOB1 → $JOB2 → $JOB3 → $JOB4"
```

---

## Step 5 — Running the end-to-end pipeline in a single job

For smaller systems or rapid exploration, all four stages can be run inside
a single SLURM job using the orchestration script:

```bash
#!/usr/bin/env bash
# run_full_pipeline.sh
#SBATCH --job-name=tcrmd_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/full_pipeline_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd

# Use the simulate image; it contains openmm, pdbfixer, and numpy.
# For the analytics step we switch to the analytics image.
SIF_SIMULATE=$WORKDIR/tcrmd-simulate.sif
SIF_ANALYTICS=$WORKDIR/tcrmd-analytics.sif

# Stages 1-3 (all require openmm/pdbfixer; s2s only needs boltz):
apptainer exec --bind $WORKDIR:/workspace $SIF_SIMULATE \
    python /workspace/run_pipeline.py \
        --sequences  /workspace/sequences.json \
        --template   /workspace/template.pdb \
        --checkpoint /workspace/boltz1.ckpt \
        --output     /workspace/results \
        --num-steps  500000 \
        --temperature 300.0 \
        --platform   CPU \
        --ph         7.4 \
        --skip-inference   # omit if boltz is available in the image
```

---

## Apptainer bind-mount cheat sheet

Apptainer does not automatically bind-mount user directories from the host.
Always pass `--bind <host_path>:<container_path>` for every directory the
container needs to read or write:

```bash
apptainer exec \
    --bind /scratch/$USER/tcrmd:/workspace \
    --bind /scratch/$USER/boltz_weights:/weights \
    tcrmd-simulate.sif \
    python run_pipeline.py \
        --checkpoint /weights/boltz1.ckpt \
        --sequences  /workspace/sequences.json \
        --output     /workspace/results
```

Common bind mounts:

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `/scratch/$USER/tcrmd` | `/workspace` | Working directory (inputs + outputs) |
| `/scratch/$USER/boltz_weights` | `/weights` | Boltz checkpoint files |
| `/tmp` | `/tmp` | Temporary files (often already auto-bound) |

---

## Troubleshooting

### "No space left on device" during solvation

The solvated system can be 200–400 MB.  Ensure `$TMPDIR` (used by PDBFixer
internally) points to a filesystem with at least 2 GB free:

```bash
export TMPDIR=/scratch/$USER/tmp
mkdir -p $TMPDIR
```

### OpenMM crashes with "CUDA error"

Switch to `platformName="CPU"` for debugging, or check that the SLURM job
requested a GPU (`--gres=gpu:1`) and that the CUDA module is loaded.

### Apptainer "permission denied" on `.sif`

SIF files must be world-readable on shared filesystems:

```bash
chmod 644 tcrmd-*.sif
```

### PROPKA import error

`propka` is only available in the `tcrmd-sysprep` image.  The `run_pipeline.py`
orchestrator handles this gracefully — it logs a warning and skips the
`AssignProtonationStates` step if `propka` is absent.

---

## Expected outputs

After all four stages complete successfully:

```
results/
├── intermediate/
│   ├── boltz_input.yaml
│   ├── predicted_complex.pdb
│   ├── aligned_complex.pdb
│   ├── cleaned.pdb
│   ├── protonated.pdb
│   └── solvated.pdb
├── simulation/
│   ├── minimized.pdb
│   ├── trajectory.dcd
│   ├── equil_log.csv
│   └── checkpoint_*.xml
├── analytics/
│   ├── rmsf.csv
│   ├── hbond_map.npy
│   ├── contact_map.npy
│   ├── bsa.csv
│   └── com_distance.csv
└── pipeline_summary.json
```


```bash
# Smoke-test: run the sequence-to-structure unit tests
apptainer exec \
    --bind /scratch/user/tcrmd:/workspace \
    /scratch/user/tcrmd/tcrmd-s2s.sif \
    python -m pytest /workspace/tests/test_sequence_to_structure.py -v --tb=short
```

---

## Step 3 — Serial execution scripts

The recommended HPC workflow runs each stage in its own job step so that
you can inspect intermediate outputs before proceeding.

### 3a · Stage 1 — Sequence-to-Structure

```bash
#!/usr/bin/env bash
# stage1_s2s.sh
#SBATCH --job-name=tcrmd_s2s
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/s2s_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-s2s.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
import json
from tcrmd.sequence_to_structure import (
    ExtractCDRLoops,
    FormatBoltzInput,
    RunBoltzInference,
    AlignToTemplate,
)

with open("/workspace/sequences.json") as fh:
    sequences = json.load(fh)

for chain in ("alpha", "beta"):
    cdrs = ExtractCDRLoops(sequences, chainType=chain)
    print(f"CDR loops ({chain}):", cdrs)

boltz_yaml = FormatBoltzInput(
    sequences,
    templatePdbPath="/workspace/template.pdb",
    outputDir="/workspace/intermediate",
)

predicted_pdb = RunBoltzInference(
    boltz_yaml,
    checkpointPath="/workspace/boltz1.ckpt",
    outputDir="/workspace/intermediate",
)

rmsd = AlignToTemplate(
    predicted_pdb,
    "/workspace/template.pdb",
    "/workspace/intermediate/aligned_complex.pdb",
)
print(f"Alignment RMSD: {rmsd:.3f} Å")
PYTHON
```

### 3b · Stage 2 — System Preparation

```bash
#!/usr/bin/env bash
# stage2_sysprep.sh
#SBATCH --job-name=tcrmd_sysprep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sysprep_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-sysprep.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
from tcrmd.system_preparation import CleanPDB, AssignProtonationStates, SolvateSystem

CleanPDB(
    "/workspace/intermediate/aligned_complex.pdb",
    "/workspace/intermediate/cleaned.pdb",
    ph=7.4,
)

try:
    AssignProtonationStates(
        "/workspace/intermediate/cleaned.pdb",
        "/workspace/intermediate/protonated.pdb",
        ph=7.4,
    )
    ready = "/workspace/intermediate/protonated.pdb"
except ImportError:
    print("propka unavailable; using cleaned PDB")
    ready = "/workspace/intermediate/cleaned.pdb"

SolvateSystem(
    ready,
    "/workspace/intermediate/solvated.pdb",
    padding=1.0,
    ionicStrength=0.15,
)
PYTHON
```

### 3c · Stage 3 — Simulation

```bash
#!/usr/bin/env bash
# stage3_simulate.sh
#SBATCH --job-name=tcrmd_sim
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/simulate_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-simulate.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
import json
from tcrmd.simulate import RunSimulation

results = RunSimulation(
    pdbPath="/workspace/intermediate/solvated.pdb",
    outputDir="/workspace/simulation",
    numSteps=500_000,      # 500 000 × 4 fs = 2 ns (HMR)
    temperature=300.0,
    hmr=True,
    platformName="CPU",
    minimizeFirst=True,
)

with open("/workspace/simulation/sim_results.json", "w") as fh:
    json.dump(results, fh, indent=2)

print("Minimised PDB :", results["minimized_pdb"])
print("Trajectory    :", results["trajectory"])
print("Energy (kJ/mol):", results["potential_energy"])
PYTHON
```

> **GPU clusters:** Replace `platformName="CPU"` with `"CUDA"` and add
> `#SBATCH --gres=gpu:1` to the SLURM header.  No other code changes are needed.

### 3d · Stage 4 — Inference & Analytics

```bash
#!/usr/bin/env bash
# stage4_analytics.sh
#SBATCH --job-name=tcrmd_analytics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/analytics_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd
SIF=$WORKDIR/tcrmd-analytics.sif

apptainer exec \
    --bind $WORKDIR:/workspace \
    $SIF python - <<'PYTHON'
import os
from tcrmd.inference_analytics import (
    ComputeRMSF,
    ComputeHydrogenBondMap,
    ComputeContactMap,
    ComputeBuriedSurfaceArea,
    ComputeCOMDistance,
)

topo = "/workspace/simulation/minimized.pdb"
traj = "/workspace/simulation/trajectory.dcd"
odir = "/workspace/analytics"
os.makedirs(odir, exist_ok=True)

ComputeRMSF(topo, traj, outputPath=f"{odir}/rmsf.csv")

ComputeHydrogenBondMap(
    topo, traj,
    donorSelection="segid A or segid B",
    acceptorSelection="segid P",
    outputPath=f"{odir}/hbond_map.npy",
)

ComputeContactMap(
    topo, traj,
    selectionA="segid A or segid B",
    selectionB="segid P",
    outputPath=f"{odir}/contact_map.npy",
)

ComputeBuriedSurfaceArea(
    topo, traj,
    selectionA="segid A or segid B",
    selectionB="segid C or segid D or segid P",
    outputPath=f"{odir}/bsa.csv",
)

ComputeCOMDistance(
    topo, traj,
    selectionA="segid A or segid B",
    selectionB="segid C or segid D",
    outputPath=f"{odir}/com_distance.csv",
)

print("Analytics complete. Files written to", odir)
PYTHON
```

---

## Step 4 — Submitting the serial pipeline

Chain the four jobs with `--dependency=afterok` so each stage only starts
when the preceding stage succeeds:

```bash
JOB1=$(sbatch --parsable stage1_s2s.sh)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 stage2_sysprep.sh)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 stage3_simulate.sh)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 stage4_analytics.sh)

echo "Jobs submitted: $JOB1 → $JOB2 → $JOB3 → $JOB4"
```

---

## Step 5 — Running the end-to-end pipeline in a single job

For smaller systems or rapid exploration, all four stages can be run inside
a single SLURM job using the orchestration script:

```bash
#!/usr/bin/env bash
# run_full_pipeline.sh
#SBATCH --job-name=tcrmd_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/full_pipeline_%j.out

set -euo pipefail

WORKDIR=/scratch/$USER/tcrmd

# Use the simulate image; it contains openmm, pdbfixer, and numpy.
# For the analytics step we switch to the analytics image.
SIF_SIMULATE=$WORKDIR/tcrmd-simulate.sif
SIF_ANALYTICS=$WORKDIR/tcrmd-analytics.sif

# Stages 1-3 (all require openmm/pdbfixer; s2s only needs boltz):
apptainer exec --bind $WORKDIR:/workspace $SIF_SIMULATE \
    python /workspace/run_pipeline.py \
        --sequences  /workspace/sequences.json \
        --template   /workspace/template.pdb \
        --checkpoint /workspace/boltz1.ckpt \
        --output     /workspace/results \
        --num-steps  500000 \
        --temperature 300.0 \
        --platform   CPU \
        --ph         7.4 \
        --skip-inference   # omit if boltz is available in the image
```

---

## Apptainer bind-mount cheat sheet

Apptainer does not automatically bind-mount user directories from the host.
Always pass `--bind <host_path>:<container_path>` for every directory the
container needs to read or write:

```bash
apptainer exec \
    --bind /scratch/$USER/tcrmd:/workspace \
    --bind /scratch/$USER/boltz_weights:/weights \
    tcrmd-simulate.sif \
    python run_pipeline.py \
        --checkpoint /weights/boltz1.ckpt \
        --sequences  /workspace/sequences.json \
        --output     /workspace/results
```

Common bind mounts:

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `/scratch/$USER/tcrmd` | `/workspace` | Working directory (inputs + outputs) |
| `/scratch/$USER/boltz_weights` | `/weights` | Boltz checkpoint files |
| `/tmp` | `/tmp` | Temporary files (often already auto-bound) |

---

## Troubleshooting

### "No space left on device" during solvation

The solvated system can be 200–400 MB.  Ensure `$TMPDIR` (used by PDBFixer
internally) points to a filesystem with at least 2 GB free:

```bash
export TMPDIR=/scratch/$USER/tmp
mkdir -p $TMPDIR
```

### OpenMM crashes with "CUDA error"

Switch to `platformName="CPU"` for debugging, or check that the SLURM job
requested a GPU (`--gres=gpu:1`) and that the CUDA module is loaded.

### Apptainer "permission denied" on `.sif`

SIF files must be world-readable on shared filesystems:

```bash
chmod 644 tcrmd-*.sif
```

### PROPKA import error

`propka` is only available in the `tcrmd-sysprep` image.  The `run_pipeline.py`
orchestrator handles this gracefully — it logs a warning and skips the
`AssignProtonationStates` step if `propka` is absent.

---

## Expected outputs

After all four stages complete successfully:

```
results/
├── intermediate/
│   ├── boltz_input.yaml
│   ├── predicted_complex.pdb
│   ├── aligned_complex.pdb
│   ├── cleaned.pdb
│   ├── protonated.pdb
│   └── solvated.pdb
├── simulation/
│   ├── minimized.pdb
│   ├── trajectory.dcd
│   ├── equil_log.csv
│   └── checkpoint_*.xml
├── analytics/
│   ├── rmsf.csv
│   ├── hbond_map.npy
│   ├── contact_map.npy
│   ├── bsa.csv
│   └── com_distance.csv
└── pipeline_summary.json
```
