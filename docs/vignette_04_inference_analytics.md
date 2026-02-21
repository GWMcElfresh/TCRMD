# Vignette 04 — Inference & Analytics

## Overview

Stage 4 extracts binding-relevant observables from the MD trajectory using the
functions in `tcrmd/inference_analytics.py`.  All heavy computation is delegated
to **MDAnalysis** and **NumPy**; no GPU is required.

| Function | Observable | Output |
|----------|-----------|--------|
| `ComputeRMSF` | Per-residue Cα fluctuation | CSV |
| `ComputeHydrogenBondMap` | TCR–peptide H-bond persistence | NumPy `.npy` |
| `ComputeContactMap` | Residue–residue contact frequencies | NumPy `.npy` |
| `ComputeBuriedSurfaceArea` | Per-frame buried surface area (BSA) | CSV |
| `ComputeCOMDistance` | Centre-of-mass distance, TCR vs. MHC | CSV |

### Biological context

After a short equilibration simulation the following observables are commonly
used to characterise TCR–pMHC binding:

- **RMSF** — high values in CDR3 loops indicate conformational flexibility;
  low values reflect a well-docked, stable interface.
- **H-bond persistence** — hydrogen bonds sustained in ≥30 % of frames are
  considered stable and likely contribute to binding free energy.
- **Contact map** — the peptide–CDR3 contact map shows which pMHC residues are
  "seen" by each CDR loop, helping to identify key recognition motifs.
- **BSA** — buried surface area is correlated with binding affinity.  A typical
  TCR–pMHC interface buries 1 200–2 000 Å².
- **COM distance** — the centre-of-mass (COM) distance between the TCR V-domain
  and the MHC α-helices provides a global measure of association/dissociation
  during the trajectory.

---

## Environment

```bash
conda activate tcrmd-analytics
# or, inside Docker:
docker run --rm -v $PWD:/workspace tcrmd-analytics bash
```

Dependencies: `mdanalysis>=2.0`, `numpy>=1.25`.

---

## Common inputs

Every analytics function accepts the same two positional arguments:

| Argument | Description |
|----------|-------------|
| `topologyPath` | Path to a PDB (or PSF) file that defines atom names and connectivity |
| `trajectoryPath` | Path to a DCD (or XTC) trajectory file |

Use `simulation/minimized.pdb` as the topology and `simulation/trajectory.dcd`
as the trajectory.  If `minimized.pdb` is unavailable (e.g. `minimizeFirst=False`),
use `intermediate/solvated.pdb` instead.

---

## 4a · Root Mean Square Fluctuation (RMSF)

RMSF measures how much each residue's Cα atom deviates from its time-averaged
position over the trajectory.

```python
import numpy as np
from tcrmd.inference_analytics import ComputeRMSF

resids, rmsf_values = ComputeRMSF(
    topologyPath="simulation/minimized.pdb",
    trajectoryPath="simulation/trajectory.dcd",
    selection="protein and name CA",   # all Cα atoms
    outputPath="analytics/rmsf.csv",
)

# Find the most flexible residue
most_flexible = resids[np.argmax(rmsf_values)]
print(f"Most flexible residue: {most_flexible}  RMSF = {rmsf_values.max():.2f} Å")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topologyPath` | `str` | — | Topology PDB/PSF |
| `trajectoryPath` | `str` | — | Trajectory DCD/XTC |
| `selection` | `str` | `"protein and name CA"` | MDAnalysis selection string |
| `outputPath` | `str \| None` | `None` | CSV output path (`resid,resname,rmsf_angstrom`) |

### Returns

Tuple `(resids, rmsf_values)` — two 1-D NumPy arrays of length equal to the
number of selected residues.

### Interpreting CDR-loop RMSF

To compute RMSF only for the CDR3α loop (IMGT positions 105–117, chain A):

```python
cdr3_alpha_rmsf = ComputeRMSF(
    topologyPath="simulation/minimized.pdb",
    trajectoryPath="simulation/trajectory.dcd",
    selection="segid A and resid 105:117 and name CA",
    outputPath="analytics/cdr3_alpha_rmsf.csv",
)
```

---

## 4b · Hydrogen-bond persistence map

`ComputeHydrogenBondMap` counts, for every donor–acceptor residue pair, the
fraction of trajectory frames in which a hydrogen bond is present.

The geometric criteria are:
- Donor–acceptor distance ≤ `distanceCutoff` Å (default 3.5 Å)
- D–H···A angle ≥ `angleCutoff` degrees (default 150°)

```python
from tcrmd.inference_analytics import ComputeHydrogenBondMap

persistence_matrix = ComputeHydrogenBondMap(
    topologyPath="simulation/minimized.pdb",
    trajectoryPath="simulation/trajectory.dcd",
    donorSelection="segid A or segid B",    # TCR α+β chains
    acceptorSelection="segid P",             # peptide
    distanceCutoff=3.5,
    angleCutoff=150.0,
    outputPath="analytics/hbond_map.npy",
)
print(f"H-bond matrix shape: {persistence_matrix.shape}")
# → (N_TCR_residues, N_peptide_residues)
```

> **Note:** Hydrogen-bond analysis requires explicit H atoms in the topology.
> If no H atoms are detected, the function logs a warning and returns a
> zero-filled matrix rather than raising an exception.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topologyPath` | `str` | — | Topology file |
| `trajectoryPath` | `str` | — | Trajectory file |
| `donorSelection` | `str` | `"protein"` | MDAnalysis selection for H-bond donors |
| `acceptorSelection` | `str` | `"protein"` | MDAnalysis selection for H-bond acceptors |
| `distanceCutoff` | `float` | `3.5` | Maximum donor–acceptor distance (Å) |
| `angleCutoff` | `float` | `150.0` | Minimum D–H···A angle (degrees) |
| `outputPath` | `str \| None` | `None` | Path for NumPy `.npy` output |

### Returns

2-D NumPy array of shape `(N_donors, N_acceptors)` with values in `[0, 1]`.

### Loading and plotting

```python
import numpy as np
import matplotlib.pyplot as plt

mat = np.load("analytics/hbond_map.npy")
plt.figure(figsize=(10, 8))
plt.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="hot_r")
plt.colorbar(label="H-bond persistence fraction")
plt.xlabel("Peptide residue index")
plt.ylabel("TCR residue index")
plt.title("TCR–peptide hydrogen-bond persistence")
plt.tight_layout()
plt.savefig("analytics/hbond_map.png", dpi=150)
```

---

## 4c · Residue–residue contact map

`ComputeContactMap` reports the fraction of frames in which any heavy atom from
residue-pair (A_i, B_j) is within `cutoff` Å.

```python
from tcrmd.inference_analytics import ComputeContactMap

resids_a, resids_b, contact_matrix = ComputeContactMap(
    topologyPath="simulation/minimized.pdb",
    trajectoryPath="simulation/trajectory.dcd",
    selectionA="segid A or segid B",   # TCR (CDR loops)
    selectionB="segid P",               # peptide
    cutoff=4.5,                         # Å; standard heavy-atom contact threshold
    outputPath="analytics/contact_map.npy",
)
print(f"Contact matrix shape: {contact_matrix.shape}")
# → (N_TCR_residues, N_peptide_residues)

# Residues with any contact in >50 % of frames
high_contact_tcr = resids_a[contact_matrix.max(axis=1) > 0.5]
print("TCR residues with >50% peptide contact:", high_contact_tcr)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topologyPath` | `str` | — | Topology file |
| `trajectoryPath` | `str` | — | Trajectory file |
| `selectionA` | `str` | — | MDAnalysis selection for group A |
| `selectionB` | `str` | — | MDAnalysis selection for group B |
| `cutoff` | `float` | `4.5` | Heavy-atom distance cutoff (Å) |
| `outputPath` | `str \| None` | `None` | Path for NumPy `.npy` output |

### Returns

Tuple `(resids_A, resids_B, contact_matrix)`.

---

## 4d · Buried Surface Area (BSA)

BSA is computed per frame using the Shrake-Rupley rolling-sphere algorithm
implemented entirely in NumPy (no SASA library required).

```
BSA = ½ × (SASA_A + SASA_B − SASA_complex)
```

A positive BSA means surface that was solvent-exposed in the unbound chains is
buried upon complex formation — a direct proxy for binding affinity.

```python
from tcrmd.inference_analytics import ComputeBuriedSurfaceArea

frame_indices, bsa_values = ComputeBuriedSurfaceArea(
    topologyPath="simulation/minimized.pdb",
    trajectoryPath="simulation/trajectory.dcd",
    selectionA="segid A or segid B",             # TCR
    selectionB="segid C or segid D or segid P",  # pMHC
    probeRadius=1.4,                              # Å (water probe)
    outputPath="analytics/bsa.csv",
)

print(f"Mean BSA: {bsa_values.mean():.1f} Å²")
print(f"Max  BSA: {bsa_values.max():.1f} Å²")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topologyPath` | `str` | — | Topology file |
| `trajectoryPath` | `str` | — | Trajectory file |
| `selectionA` | `str` | — | Binding partner A (e.g. TCR) |
| `selectionB` | `str` | — | Binding partner B (e.g. pMHC) |
| `probeRadius` | `float` | `1.4` | Solvent probe radius (Å) |
| `outputPath` | `str \| None` | `None` | CSV path (`frame,bsa_angstrom2`) |

### Returns

Tuple `(frame_indices, bsa_values)` — two 1-D NumPy arrays of length equal to
the number of trajectory frames.

### Performance note

The pure-NumPy Shrake-Rupley implementation has O(N²) per-atom cost.  For the
full ~650-residue complex with default 92 sphere points per atom, expect
approximately 30–60 s per frame on a single core.  Reduce `n_sphere_points`
(internal parameter) or select only interface atoms to speed things up for
exploratory analyses.

---

## 4e · Centre-of-Mass (COM) distance

`ComputeCOMDistance` tracks the separation between the mass-weighted centroids
of two molecular groups over the trajectory.  A decreasing trend during the
simulation may indicate progressive TCR docking; an abrupt increase can signal
transient unbinding.

```python
from tcrmd.inference_analytics import ComputeCOMDistance

frame_indices, com_distances = ComputeCOMDistance(
    topologyPath="simulation/minimized.pdb",
    trajectoryPath="simulation/trajectory.dcd",
    selectionA="segid A or segid B",   # TCR V-alpha/V-beta domains
    selectionB="segid C or segid D",   # MHC α-helices
    outputPath="analytics/com_distance.csv",
)

print(f"Mean COM distance: {com_distances.mean():.2f} Å")
print(f"Min  COM distance: {com_distances.min():.2f} Å")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `topologyPath` | `str` | — | Topology file |
| `trajectoryPath` | `str` | — | Trajectory file |
| `selectionA` | `str` | — | First molecular group |
| `selectionB` | `str` | — | Second molecular group |
| `outputPath` | `str \| None` | `None` | CSV path (`frame,com_distance_angstrom`) |

### Returns

Tuple `(frame_indices, distances)` — two 1-D NumPy arrays.

---

## End-to-end analytics block

```python
import os
from tcrmd.inference_analytics import (
    ComputeRMSF,
    ComputeHydrogenBondMap,
    ComputeContactMap,
    ComputeBuriedSurfaceArea,
    ComputeCOMDistance,
)

topo  = "simulation/minimized.pdb"
traj  = "simulation/trajectory.dcd"
odir  = "analytics"
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
```

---

## Running the tests

```bash
pytest tests/test_inference_analytics.py -v

# Inside the Docker image
docker run --rm -v $PWD:/workspace -w /workspace tcrmd-analytics \
    python -m pytest tests/test_inference_analytics.py -v --tb=short
```

## Output files

| File | Format | Columns / shape |
|------|--------|-----------------|
| `analytics/rmsf.csv` | CSV | `resid`, `resname`, `rmsf_angstrom` |
| `analytics/hbond_map.npy` | NumPy | `(N_donors, N_acceptors)` float64 |
| `analytics/contact_map.npy` | NumPy | `(N_A_residues, N_B_residues)` float64 |
| `analytics/bsa.csv` | CSV | `frame`, `bsa_angstrom2` |
| `analytics/com_distance.csv` | CSV | `frame`, `com_distance_angstrom` |
