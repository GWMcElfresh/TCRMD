# Vignette 02 — System Preparation

## Overview

Stage 2 converts a raw predicted (or experimental) PDB into a simulation-ready,
solvated system.  The three functions in `tcrmd/system_preparation.py` are
designed to run in sequence:

| Step | Function | Tool | Purpose |
|------|----------|------|---------|
| 2a | `CleanPDB` | PDBFixer | Add missing atoms/residues; remove artefacts |
| 2b | `AssignProtonationStates` | PROPKA + PDBFixer | Set His/Glu/Asp/Cys states at target pH |
| 2c | `SolvateSystem` | OpenMM Modeller | Wrap in a water box with physiological ionic strength |

### Biological context

AI-predicted structures and crystallographic PDBs alike often contain missing
side-chain atoms, non-standard residues, or incomplete N/C termini.  Before
starting a simulation you must:

1. **Repair** the structure so every residue has a complete heavy-atom set
   (step 2a).
2. **Protonate** titratable residues (His, Glu, Asp, Lys, Arg, Tyr, Cys)
   correctly for the intended pH — in the TCR/pMHC interface, pH-sensitive
   histidine contacts in the MHC groove can substantially affect binding
   (step 2b).
3. **Solvate** the complex in explicit solvent.  A rectangular TIP3P-FB water
   box with 0.15 M NaCl mimics physiological conditions and allows the use of
   periodic-boundary-condition (PBC) electrostatics (step 2c).

---

## Environment

```bash
conda activate tcrmd-sysprep
# or, inside Docker:
docker run --rm -v $PWD:/workspace tcrmd-sysprep bash
```

Dependencies: `openmm>=8.1`, `pdbfixer>=1.9`, `propka>=3.5`, `numpy`.

---

## 2a · Cleaning the PDB

`CleanPDB` uses PDBFixer to add missing atoms and residues, replace non-standard
residues, and optionally add missing hydrogens.

```python
from tcrmd.system_preparation import CleanPDB

cleaned_pdb = CleanPDB(
    inputPdbPath="intermediate/aligned_complex.pdb",
    outputPdbPath="intermediate/cleaned.pdb",
    addMissingHydrogens=True,
    addMissingResidues=True,
    removeHeterogens=False,   # keep crystal waters by default
    ph=7.4,
)
print("Cleaned PDB:", cleaned_pdb)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputPdbPath` | `str` | — | Path to the raw input PDB |
| `outputPdbPath` | `str` | — | Destination path for the cleaned PDB |
| `addMissingHydrogens` | `bool` | `True` | Add H atoms (required for MD) |
| `addMissingResidues` | `bool` | `True` | Model missing residue fragments |
| `removeHeterogens` | `bool` | `False` | Strip all HETATM records (ligands, waters) |
| `ph` | `float` | `7.4` | pH used by PDBFixer when adding missing H |

### Returns

Absolute path to the written cleaned PDB (`str`).

### Raises

- `FileNotFoundError` — `inputPdbPath` does not exist
- `ImportError` — `pdbfixer` or `openmm` is not installed

### Notes

- **Non-standard residues** (e.g. selenomethionine, phosphotyrosine) are
  automatically replaced with their standard equivalents.
- Set `removeHeterogens=True` if you want a protein-only system with no bound
  ligands or crystallographic waters.

---

## 2b · Assigning protonation states

`AssignProtonationStates` runs PROPKA to compute pKa values for all titratable
residues and then calls PDBFixer's `addMissingHydrogens` to place protons
consistent with those predictions at the target pH.

```python
from tcrmd.system_preparation import AssignProtonationStates

protonated_pdb = AssignProtonationStates(
    inputPdbPath="intermediate/cleaned.pdb",
    outputPdbPath="intermediate/protonated.pdb",
    ph=7.4,
)
print("Protonated PDB:", protonated_pdb)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputPdbPath` | `str` | — | Path to the cleaned input PDB |
| `outputPdbPath` | `str` | — | Destination path |
| `ph` | `float` | `7.4` | Target pH for protonation-state assignment |

### Returns

Absolute path to the written protonated PDB (`str`).

### Raises

- `FileNotFoundError` — `inputPdbPath` does not exist
- `ImportError` — `propka`, `pdbfixer`, or `openmm` is not installed

### Notes

- PROPKA writes a `.propka` sidecar file.  The implementation copies the
  input to a temporary directory and changes the working directory there so
  PROPKA's sidecar never touches your data directories.
- If `propka` is unavailable (e.g. on a machine that only has `pdbfixer`),
  `run_pipeline.py` logs a warning and falls back to using the cleaned PDB
  directly.

---

## 2c · Solvating the system

`SolvateSystem` embeds the protonated structure in a rectangular water box,
adds counter-ions to neutralise the system charge, and brings the salt
concentration to the requested value.

```python
from tcrmd.system_preparation import SolvateSystem

solvated_pdb = SolvateSystem(
    inputPdbPath="intermediate/protonated.pdb",
    outputPdbPath="intermediate/solvated.pdb",
    padding=1.0,           # nm; water extends ≥1 nm beyond every solute atom
    ionicStrength=0.15,    # mol/L; 0.15 M NaCl (physiological)
    waterModel="tip3p",    # PBC-compatible 3-point water model
    positiveIon="Na+",
    negativeIon="Cl-",
)
print("Solvated PDB:", solvated_pdb)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputPdbPath` | `str` | — | Path to the protonated input PDB |
| `outputPdbPath` | `str` | — | Destination path |
| `padding` | `float` | `1.0` | Minimum distance from solute to box edge (nm) |
| `ionicStrength` | `float` | `0.15` | Target salt concentration (mol/L) |
| `waterModel` | `str` | `"tip3p"` | OpenMM water model identifier (`"tip3p"`, `"tip3pfb"`, `"opc"`) |
| `positiveIon` | `str` | `"Na+"` | Cation species |
| `negativeIon` | `str` | `"Cl-"` | Anion species |

### Returns

Absolute path to the written solvated PDB (`str`).

### Raises

- `FileNotFoundError` — `inputPdbPath` does not exist
- `ImportError` — `openmm` is not installed

### Notes on water models

`SolvateSystem` defaults to `"tip3p"` for box building.  The simulation stage
(Stage 3) independently loads force-field XML files and defaults to
`amber14/tip3pfb.xml` (TIP3P-FB).  For a fully consistent force field,
pass `waterModel="tip3pfb"` to `SolvateSystem` so the box geometry and
simulation forcefield match:

```python
SolvateSystem(ready_pdb, "intermediate/solvated.pdb", waterModel="tip3pfb")
```

### Notes on system size

The TCR-pMHC complex is ~650 residues.  With `padding=1.0 nm` and TIP3P-FB
water the box will contain roughly 40 000–60 000 atoms.  Reducing `padding` to
`0.5 nm` (as used in the test suite) cuts the atom count by ~60 % and is
appropriate for exploratory runs.

For production simulations a minimum padding of 1.0 nm is recommended to avoid
self-interaction artefacts under PBC.

---

## End-to-end example

```python
from tcrmd.system_preparation import CleanPDB, AssignProtonationStates, SolvateSystem

# 2a — clean
CleanPDB(
    "intermediate/aligned_complex.pdb",
    "intermediate/cleaned.pdb",
    ph=7.4,
)

# 2b — protonate (falls back gracefully if propka is absent)
try:
    AssignProtonationStates(
        "intermediate/cleaned.pdb",
        "intermediate/protonated.pdb",
        ph=7.4,
    )
    ready_pdb = "intermediate/protonated.pdb"
except ImportError:
    print("propka not available; using cleaned PDB for solvation")
    ready_pdb = "intermediate/cleaned.pdb"

# 2c — solvate
SolvateSystem(
    ready_pdb,
    "intermediate/solvated.pdb",
    padding=1.0,
    ionicStrength=0.15,
)
```

---

## Running the tests

```bash
pytest tests/test_system_preparation.py -v

# Inside the Docker image
docker run --rm -v $PWD:/workspace -w /workspace tcrmd-sysprep \
    python -m pytest tests/test_system_preparation.py -v --tb=short
```

## Output files

| File | Description |
|------|-------------|
| `intermediate/cleaned.pdb` | PDBFixer-repaired structure |
| `intermediate/protonated.pdb` | pH-adjusted protonation states |
| `intermediate/solvated.pdb` | Solvated simulation-ready structure |
