# Vignette 01 — Sequence-to-Structure

## Overview

Stage 1 turns amino-acid sequences into a 3-D ternary complex structure.  It
comprises three operations that are exposed as independent Python functions in
`tcrmd/sequence_to_structure.py`:

| Step | Function | Purpose |
|------|----------|---------|
| 1a | `ExtractCDRLoops` | Identify CDR1/2/3 from IMGT-numbered TCR sequences |
| 1b | `FormatBoltzInput` | Write the YAML/JSON input file consumed by `boltz predict` |
| 1c | `RunBoltzInference` | Invoke the Boltz-1 AI structure predictor |
| 1d | `AlignToTemplate` | Kabsch superimposition onto a high-resolution PDB template |

### Biological context

The T-cell receptor recognises antigenic peptides displayed in the groove of
the major histocompatibility complex (MHC).  The three complementarity-
determining regions (CDR1, CDR2, CDR3) of each TCR chain (α and β) form the
primary contact surface.  CDR3 is hypervariable and dominates antigen
specificity.

IMGT-standardised numbering (Lefranc et al.) assigns positions 27–38 to CDR1,
56–65 to CDR2, and 105–117 to CDR3 for both chains.

---

## Environment

```bash
conda activate tcrmd-s2s
# or, inside Docker:
docker run --rm -v $PWD:/workspace tcrmd-s2s bash
```

Dependencies: `numpy`, `boltz>=0.4`.

---

## 1a · Extracting CDR loops

```python
import json
from tcrmd.sequence_to_structure import ExtractCDRLoops

with open("tests/data/sequences.json") as fh:
    sequences = json.load(fh)

alpha_cdrs = ExtractCDRLoops(sequences, chainType="alpha")
beta_cdrs  = ExtractCDRLoops(sequences, chainType="beta")

print("Alpha CDR loops:")
for name, seq in alpha_cdrs.items():
    print(f"  {name}: {seq}")
```

**Expected output** (truncated for readability):

```
Alpha CDR loops:
  CDR1: VQTPKFQVLKTG
  CDR2: SVGAGITDQ
  CDR3: CASRIRDDKIIF
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequenceData` | `dict[str, str]` | — | Mapping of chain names to AA sequences |
| `chainType` | `str` | `"alpha"` | `"alpha"` or `"beta"` (case-insensitive) |

### Returns

`dict[str, str]` — keys `"CDR1"`, `"CDR2"`, `"CDR3"`, values are the
extracted subsequences (may be empty strings for very short sequences).

### Raises

- `ValueError` — unknown `chainType`, or sequence contains non-standard characters
- `KeyError` — `sequenceData` does not contain the requested chain

---

## 1b · Formatting Boltz input

`FormatBoltzInput` serialises the sequence dictionary and an optional template
PDB path into the JSON/YAML format expected by `boltz predict`.

```python
from tcrmd.sequence_to_structure import FormatBoltzInput

boltz_yaml = FormatBoltzInput(
    sequences,
    templatePdbPath="reference/hla_a0201.pdb",  # optional; pass None to omit
    outputDir="intermediate/",
)
print("Boltz input written to:", boltz_yaml)
```

The output file is always named `boltz_input.yaml` inside `outputDir`.

### Chain ID mapping

| Sequence key | PDB chain ID |
|---|---|
| `alpha` | A |
| `beta` | B |
| `mhc_alpha` | C |
| `mhc_beta` | D |
| `peptide` | P |
| any other key | first character, uppercased |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequenceData` | `dict[str, str]` | — | Chain-to-sequence mapping |
| `templatePdbPath` | `str \| None` | — | Absolute path to reference PDB, or `None` |
| `outputDir` | `str` | — | Directory for the output file (created if absent) |

### Returns

Absolute path to the written `boltz_input.yaml` file.

### Raises

- `ValueError` — `sequenceData` is empty
- `FileNotFoundError` — `templatePdbPath` given but does not exist

---

## 1c · Running Boltz inference

> **Note:** `boltz` must be installed and the checkpoint file downloaded before
> this step.  See the [Boltz documentation](https://github.com/jwohlwend/boltz)
> for installation instructions.

```python
from tcrmd.sequence_to_structure import RunBoltzInference

predicted_pdb = RunBoltzInference(
    inputPath="intermediate/boltz_input.yaml",
    checkpointPath="boltz1.ckpt",
    outputDir="intermediate/",
    extraArgs=["--num_workers", "4"],   # forwarded verbatim to boltz predict
)
print("Predicted complex:", predicted_pdb)
# → intermediate/predicted_complex.pdb
```

Internally this calls:

```bash
boltz predict \
    --checkpoint boltz1.ckpt \
    --sequences  intermediate/boltz_input.yaml \
    --output_dir intermediate/
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputPath` | `str` | — | Path to `boltz_input.yaml` from step 1b |
| `checkpointPath` | `str` | — | Path to `boltz1.ckpt` model weights |
| `outputDir` | `str` | — | Directory for Boltz output files |
| `extraArgs` | `list[str] \| None` | `None` | Extra CLI flags forwarded to `boltz predict` |

### Returns

Path to `<outputDir>/predicted_complex.pdb`.

### Raises

- `FileNotFoundError` — `inputPath` or `checkpointPath` does not exist
- `subprocess.CalledProcessError` — `boltz predict` exits non-zero

### Skipping inference

If you already have a high-quality structure and want to bypass Boltz, set
`--skip-inference` on the CLI (or pass `skipInference=True` to `RunPipeline`).
The value of `--template` is used directly as the predicted complex.

---

## 1d · Aligning to a template

After Boltz generates a structure, `AlignToTemplate` superimposes the
predicted MHC backbone onto a reference PDB using the Kabsch algorithm over
common Cα atoms.  This corrects the absolute orientation of the complex so
downstream solvation and simulation start from a physically consistent pose.

```python
from tcrmd.sequence_to_structure import AlignToTemplate

rmsd = AlignToTemplate(
    predictedPdbPath="intermediate/predicted_complex.pdb",
    templatePdbPath="reference/hla_a0201.pdb",
    outputPdbPath="intermediate/aligned_complex.pdb",
)
print(f"Aligned complex RMSD: {rmsd:.3f} Å")
```

### Algorithm

1. Parse Cα coordinates from both PDBs.
2. Find the set of residues common to both structures (by residue number).
3. Centre both point clouds.
4. Compute the optimal rotation matrix via SVD (Kabsch 1976), ensuring a
   right-handed coordinate system.
5. Apply the rotation and translation to **all** ATOM/HETATM records in the
   predicted PDB.
6. Write the transformed structure to `outputPdbPath`.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictedPdbPath` | `str` | — | AI-predicted complex PDB |
| `templatePdbPath` | `str` | — | High-resolution reference PDB |
| `outputPdbPath` | `str` | — | Destination for the superimposed structure |
| `selectionString` | `str` | `"backbone"` | Atom selection label (informational; Cα-only alignment is always used) |

### Returns

RMSD in Å over the aligned Cα atoms (`float`).

### Raises

- `FileNotFoundError` — either input PDB does not exist
- `RuntimeError` — no residue numbers are shared between the two structures

---

## End-to-end example

```python
import json
from tcrmd.sequence_to_structure import (
    ExtractCDRLoops,
    FormatBoltzInput,
    RunBoltzInference,
    AlignToTemplate,
)

with open("tests/data/sequences.json") as fh:
    sequences = json.load(fh)

# 1a — log CDR loops (not required for the pipeline)
for chain in ("alpha", "beta"):
    cdrs = ExtractCDRLoops(sequences, chainType=chain)
    print(f"CDR loops ({chain}):", cdrs)

# 1b — write Boltz input
boltz_yaml = FormatBoltzInput(
    sequences,
    templatePdbPath="reference/hla_a0201.pdb",
    outputDir="intermediate/",
)

# 1c — run inference
predicted_pdb = RunBoltzInference(
    boltz_yaml,
    checkpointPath="boltz1.ckpt",
    outputDir="intermediate/",
)

# 1d — align to template
rmsd = AlignToTemplate(
    predicted_pdb,
    "reference/hla_a0201.pdb",
    "intermediate/aligned_complex.pdb",
)
print(f"Template RMSD: {rmsd:.3f} Å")
```

---

## Running the tests

```bash
# All sequence-to-structure tests (no boltz CLI required for most)
pytest tests/test_sequence_to_structure.py -v

# Inside the Docker image
docker run --rm -v $PWD:/workspace -w /workspace tcrmd-s2s \
    python -m pytest tests/test_sequence_to_structure.py -v --tb=short
```

## Output files

| File | Description |
|------|-------------|
| `intermediate/boltz_input.yaml` | Boltz YAML/JSON input |
| `intermediate/predicted_complex.pdb` | Raw Boltz prediction |
| `intermediate/aligned_complex.pdb` | Template-aligned structure (if template provided) |
