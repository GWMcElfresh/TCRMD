# Vignette 03 — Simulation

## Overview

Stage 3 runs molecular dynamics (MD) on the solvated TCR-pMHC system using
OpenMM.  Three composable functions are provided in `tcrmd/simulate.py`:

| Step | Function | Purpose |
|------|----------|---------|
| 3a | `SetupSystem` | Build the OpenMM `Simulation` object (force field, HMR, integrator) |
| 3b | `MinimizeEnergy` | Steepest-descent / L-BFGS energy minimisation |
| 3c | `RunEquilibration` | NVT → NPT equilibration; writes DCD trajectory and checkpoints |
| — | `RunSimulation` | Convenience wrapper that chains all three steps |

### Biological context

A short equilibration run (100 ps – 1 ns) after solvation allows the water
shell and ions to relax around the complex before production sampling begins.
Key checks after equilibration:

- **Potential energy** should be negative and stable (no clashes).
- **Temperature** should converge to the target (kinetic energy thermostat).
- **RMSD** of the protein backbone should plateau, indicating structural
  stability.

---

## Environment

```bash
conda activate tcrmd-simulate
# or, inside Docker:
docker run --rm -v $PWD:/workspace tcrmd-simulate bash
```

Dependencies: `openmm>=8.1`, `pdbfixer>=1.9`, `numpy`.

---

## Force field

The default force field is **AMBER14SB** (`amber14-all.xml`) paired with the
**TIP3P-FB** water model (`amber14/tip3pfb.xml`).  These files are bundled with
OpenMM and require no separate download.

### Hydrogen Mass Repartitioning (HMR)

HMR transfers mass from heavy atoms onto their bonded hydrogens, effectively
raising the hydrogen mass from ~1 Da to ~4 Da.  This allows the timestep to be
doubled from 2 fs to 4 fs without loss of accuracy, halving the wall-clock time
for a given simulation length.  HMR is **enabled by default** (`hmr=True`).

Disable HMR with `--no-hmr` (CLI) or `hmr=False` (`RunSimulation`) when
comparing results with older trajectories generated at 2 fs/step.

---

## 3a · Setting up the system

`SetupSystem` creates an OpenMM `Simulation` ready for energy minimisation or
dynamics.

```python
from tcrmd.simulate import SetupSystem

simulation, modeller = SetupSystem(
    pdbPath="intermediate/solvated.pdb",
    forceFieldFiles=("amber14-all.xml", "amber14/tip3pfb.xml"),
    nonbondedMethod="PME",      # particle-mesh Ewald for PBC electrostatics
    constraintType="HBonds",    # constrain X-H bonds → longer timestep
    hmr=True,
    platformName="CPU",         # or "CUDA" / "OpenCL" on a GPU node
)
```

The returned `simulation` object has positions already set from the PDB file.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdbPath` | `str` | — | Path to the solvated PDB |
| `forceFieldFiles` | `tuple[str, ...] \| None` | AMBER ff19SB + TIP3P-FB | Force-field XML files |
| `nonbondedMethod` | `str` | `"PME"` | Electrostatics treatment (`"PME"`, `"NoCutoff"`, etc.) |
| `constraintType` | `str` | `"HBonds"` | Bond constraints (`"HBonds"`, `"AllBonds"`, `"None"`) |
| `hmr` | `bool` | `True` | Apply Hydrogen Mass Repartitioning |
| `platformName` | `str` | `"CPU"` | OpenMM platform |

### Returns

Tuple `(simulation, modeller)` — an `openmm.app.Simulation` and the
`openmm.app.Modeller` that holds the solvated topology and positions.

### Raises

- `FileNotFoundError` — `pdbPath` does not exist
- `ImportError` — `openmm` is not installed

### Platform notes

| Platform | Use case |
|----------|----------|
| `"CPU"` | GitHub runners, HPC CPU nodes, laptops |
| `"CUDA"` | NVIDIA GPU nodes (requires CUDA toolkit) |
| `"OpenCL"` | AMD or other OpenCL-capable GPUs |

On a 16-core CPU node with HMR enabled, 50 000 steps of a ~50 000-atom system
takes approximately 5–15 minutes depending on the CPU frequency.

---

## 3b · Energy minimisation

After solvation there are invariably steric clashes introduced by the placement
of explicit water molecules.  `MinimizeEnergy` uses OpenMM's L-BFGS minimiser
to remove these clashes before dynamics begins.

```python
from tcrmd.simulate import MinimizeEnergy

potential_energy = MinimizeEnergy(
    simulation=simulation,
    outputPdbPath="simulation/minimized.pdb",
    maxIterations=0,    # 0 → iterate until convergence (recommended)
    tolerance=10.0,     # kJ/mol/nm convergence threshold
)
print(f"Minimised potential energy: {potential_energy:.2f} kJ/mol")
```

A well-minimised ~50 000-atom system typically converges to a potential energy
in the range of −10⁶ to −10⁷ kJ/mol.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulation` | `openmm.app.Simulation` | — | Configured simulation with positions set |
| `outputPdbPath` | `str` | — | Path for the minimised structure |
| `maxIterations` | `int` | `0` | Maximum steps; `0` = converge fully |
| `tolerance` | `float` | `10.0` | Convergence in kJ/mol/nm |

### Returns

Potential energy in kJ/mol after minimisation (`float`).

---

## 3c · Equilibration

`RunEquilibration` integrates the equations of motion and writes a DCD
trajectory and checkpoint files.

```python
from tcrmd.simulate import RunEquilibration

trajectory_path = RunEquilibration(
    simulation=simulation,
    outputDir="simulation/",
    numSteps=50_000,        # 50 000 steps × 4 fs/step (HMR) = 200 ps
    temperature=300.0,      # Kelvin
    reportInterval=1_000,   # write frame + log every 1 000 steps
    checkpointInterval=10_000,
)
print("Trajectory:", trajectory_path)
# → simulation/trajectory.dcd
```

Velocities are re-initialised from a Maxwell-Boltzmann distribution at the
target temperature before the first integration step.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `simulation` | `openmm.app.Simulation` | — | Minimised simulation object |
| `outputDir` | `str` | — | Directory for trajectory and logs |
| `numSteps` | `int` | `50000` | Total integration steps (≈ 200 ps with HMR 4 fs/step; ≈ 100 ps without HMR) |
| `temperature` | `float` | `300.0` | Simulation temperature in K |
| `reportInterval` | `int` | `1000` | Frames/log lines written every N steps |
| `checkpointInterval` | `int` | `10000` | Restart checkpoint written every N steps |
| `solventAtomIndices` | `list \| None` | `None` | Atom indices to exclude from DCD (solvent atoms to save disk space) |

### Returns

Path to `<outputDir>/trajectory.dcd`.

### Output files produced by `RunEquilibration`

| File | Content |
|------|---------|
| `trajectory.dcd` | DCD binary trajectory (all atoms unless `solventAtomIndices` given) |
| `equil_log.csv` | Step, potential energy, temperature, density |
| `checkpoint_<N>.xml` | OpenMM restart checkpoint every `checkpointInterval` steps |

---

## 3d · `RunSimulation` — convenience wrapper

For the common case where you simply want to go from a solvated PDB to a
trajectory without manually chaining the three steps:

```python
from tcrmd.simulate import RunSimulation

results = RunSimulation(
    pdbPath="intermediate/solvated.pdb",
    outputDir="simulation/",
    numSteps=50_000,
    temperature=300.0,
    hmr=True,
    platformName="CPU",
    minimizeFirst=True,
)

print("Minimised PDB    :", results["minimized_pdb"])
print("Trajectory       :", results["trajectory"])
print("Potential energy :", results["potential_energy"], "kJ/mol")
```

### Returns

Dictionary with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"minimized_pdb"` | `str \| None` | Path to minimised PDB, or `None` if `minimizeFirst=False` |
| `"trajectory"` | `str` | Path to `trajectory.dcd` |
| `"potential_energy"` | `float \| None` | Post-minimisation energy, or `None` |

---

## Estimating run time

| System size | Steps | HMR | Platform | Approx. wall time |
|------------|-------|-----|----------|-------------------|
| ~10 000 atoms | 50 000 | on | CPU (8 cores) | ~2 min |
| ~50 000 atoms | 50 000 | on | CPU (8 cores) | ~20 min |
| ~50 000 atoms | 500 000 | on | CUDA A100 | ~10 min |

These estimates are highly system-dependent.  Reduce `numSteps` for quick
validation runs; increase for production-quality sampling.

---

## Running the tests

```bash
pytest tests/test_simulate.py -v

# Inside the Docker image
docker run --rm -v $PWD:/workspace -w /workspace tcrmd-simulate \
    python -m pytest tests/test_simulate.py -v --tb=short
```

## Output files

| File | Description |
|------|-------------|
| `simulation/minimized.pdb` | Energy-minimised structure |
| `simulation/trajectory.dcd` | MD trajectory |
| `simulation/equil_log.csv` | Per-step energy, temperature, density |
| `simulation/checkpoint_*.xml` | Restart checkpoints |
