"""
Sequence-to-Structure Module.

Handles CDR loop extraction from TCR sequences, input formatting for Boltz,
structural inference, and template-based superimposition of the predicted MHC
backbone onto a reference PDB.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import os
import json
import logging
import subprocess
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CDR loop definitions (IMGT numbering).
# Alpha chain CDR regions: CDR1 (27-38), CDR2 (56-65), CDR3 (105-117)
# Beta  chain CDR regions: CDR1 (27-38), CDR2 (56-65), CDR3 (105-117)
# ---------------------------------------------------------------------------
_CDR_REGIONS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "alpha": {
        "CDR1": (27, 38),
        "CDR2": (56, 65),
        "CDR3": (105, 117),
    },
    "beta": {
        "CDR1": (27, 38),
        "CDR2": (56, 65),
        "CDR3": (105, 117),
    },
}

_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def _validate_sequence(sequence: str) -> str:
    """Return upper-cased sequence or raise ValueError if non-amino-acid characters found."""
    cleaned = sequence.strip().upper()
    invalid_chars = set(cleaned) - _AMINO_ACIDS
    if invalid_chars:
        raise ValueError(
            f"Sequence contains invalid amino acid characters: {invalid_chars}"
        )
    return cleaned


def _extract_loop(sequence: str, start: int, end: int) -> str:
    """Extract a loop from a 1-indexed [start, end] inclusive range (IMGT numbering)."""
    # Convert to 0-indexed Python slice; clamp to sequence length.
    py_start = max(0, start - 1)
    py_end = min(len(sequence), end)
    return sequence[py_start:py_end]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ExtractCDRLoops(
    sequenceData: Dict[str, str],
    chainType: str = "alpha",
) -> Dict[str, str]:
    """Extract CDR1, CDR2, and CDR3 loops from a TCR chain sequence.

    Args:
        sequenceData: Mapping of chain identifiers to amino-acid sequences.
            Example: ``{"alpha": "METDTLLL...", "beta": "MGAGVSQSP..."}``.
        chainType: TCR chain to process; must be ``"alpha"`` or ``"beta"``.

    Returns:
        Dictionary mapping CDR labels (``"CDR1"``, ``"CDR2"``, ``"CDR3"``)
        to their extracted subsequences.

    Raises:
        ValueError: If *chainType* is not ``"alpha"`` or ``"beta"``, or if the
            sequence contains non-amino-acid characters.
    """
    chain_type = chainType.lower()
    if chain_type not in _CDR_REGIONS:
        raise ValueError(
            f"chainType must be 'alpha' or 'beta', got '{chainType}'"
        )

    if chain_type not in sequenceData:
        raise KeyError(
            f"sequenceData does not contain an entry for chain '{chainType}'"
        )

    raw_sequence = sequenceData[chain_type]
    sequence = _validate_sequence(raw_sequence)

    cdr_loops: Dict[str, str] = {}
    for loop_name, (start, end) in _CDR_REGIONS[chain_type].items():
        loop_seq = _extract_loop(sequence, start, end)
        cdr_loops[loop_name] = loop_seq
        logger.debug("Extracted %s (%d-%d): %s", loop_name, start, end, loop_seq)

    return cdr_loops


def FormatBoltzInput(
    sequenceData: Dict[str, str],
    templatePdbPath: Optional[str],
    outputDir: str,
) -> str:
    """Format sequence data and optional template PDB path into a Boltz YAML input.

    The function writes a ``boltz_input.yaml`` file understood by ``boltz predict``.

    Args:
        sequenceData: Mapping of chain identifiers to amino-acid sequences.
            Recognised keys: ``"alpha"``, ``"beta"``, ``"mhc_alpha"``,
            ``"mhc_beta"``, ``"peptide"``.
        templatePdbPath: Absolute path to an optional PDB template file.
            Pass ``None`` to omit template constraints.
        outputDir: Directory in which to write ``boltz_input.yaml``.

    Returns:
        Absolute path of the written ``boltz_input.yaml`` file.

    Raises:
        FileNotFoundError: If *templatePdbPath* is provided but does not exist.
        ValueError: If *sequenceData* is empty.
    """
    if not sequenceData:
        raise ValueError("sequenceData must not be empty")

    if templatePdbPath is not None and not os.path.isfile(templatePdbPath):
        raise FileNotFoundError(
            f"Template PDB not found: {templatePdbPath}"
        )

    os.makedirs(outputDir, exist_ok=True)

    # Build chain list for the Boltz YAML format.
    chain_entries: List[Dict] = []
    chain_id_map = {
        "alpha": "A",
        "beta": "B",
        "mhc_alpha": "C",
        "mhc_beta": "D",
        "peptide": "P",
    }

    for chain_key, sequence in sequenceData.items():
        chain_id = chain_id_map.get(chain_key, chain_key.upper()[:1])
        validated_seq = _validate_sequence(sequence)
        chain_entries.append(
            {
                "id": chain_id,
                "sequence": validated_seq,
                "entity": chain_key,
            }
        )

    boltz_input: Dict = {"version": 1, "chains": chain_entries}

    if templatePdbPath is not None:
        boltz_input["templates"] = [
            {
                "path": os.path.abspath(templatePdbPath),
                "chain_id": chain_id_map.get("mhc_alpha", "C"),
            }
        ]

    output_path = os.path.join(outputDir, "boltz_input.yaml")

    # Write as YAML-like JSON (Boltz also accepts JSON input).
    with open(output_path, "w") as fh:
        json.dump(boltz_input, fh, indent=2)

    logger.info("Boltz input written to %s", output_path)
    return output_path


def RunBoltzInference(
    inputPath: str,
    checkpointPath: str,
    outputDir: str,
    extraArgs: Optional[List[str]] = None,
) -> str:
    """Invoke the ``boltz predict`` command to generate the ternary complex.

    Args:
        inputPath: Path to the Boltz input YAML/JSON file produced by
            :func:`FormatBoltzInput`.
        checkpointPath: Path to the Boltz model checkpoint file
            (e.g. ``boltz1.ckpt``).
        outputDir: Directory where Boltz writes predicted PDB files.
        extraArgs: Additional command-line arguments forwarded verbatim to
            ``boltz predict``.

    Returns:
        Path to the predicted complex PDB file (``predicted_complex.pdb``).

    Raises:
        FileNotFoundError: If *inputPath* or *checkpointPath* do not exist.
        subprocess.CalledProcessError: If the ``boltz predict`` command exits
            with a non-zero status.
    """
    if not os.path.isfile(inputPath):
        raise FileNotFoundError(f"Boltz input file not found: {inputPath}")
    if not os.path.isfile(checkpointPath):
        raise FileNotFoundError(f"Boltz checkpoint not found: {checkpointPath}")

    os.makedirs(outputDir, exist_ok=True)

    cmd = [
        "boltz",
        "predict",
        "--checkpoint", checkpointPath,
        "--sequences", inputPath,
        "--output_dir", outputDir,
    ]
    if extraArgs:
        cmd.extend(extraArgs)

    logger.info("Running Boltz: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    predicted_pdb = os.path.join(outputDir, "predicted_complex.pdb")
    logger.info("Boltz inference complete. Output: %s", predicted_pdb)
    return predicted_pdb


def AlignToTemplate(
    predictedPdbPath: str,
    templatePdbPath: str,
    outputPdbPath: str,
    selectionString: str = "backbone",
) -> float:
    """Superimpose the predicted MHC backbone onto a high-resolution PDB template.

    Uses ``numpy``-based Kabsch RMSD minimization over C-alpha atoms.  Only
    residues common to both structures (by residue number) are used for the
    alignment.

    Args:
        predictedPdbPath: Path to the AI-predicted complex PDB.
        templatePdbPath: Path to the reference high-resolution PDB template.
        outputPdbPath: Where to write the superimposed structure.
        selectionString: Atom selection used for superimposition; default is
            ``"backbone"`` (N, CA, C, O atoms).

    Returns:
        RMSD (Å) of the aligned Cα atoms.

    Raises:
        FileNotFoundError: If either input PDB file does not exist.
        RuntimeError: If no common residues are found for alignment.
    """
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for AlignToTemplate") from exc

    for path in (predictedPdbPath, templatePdbPath):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"PDB file not found: {path}")

    # Parse CA coordinates from a PDB file.
    def _parse_ca_coords(pdb_path: str) -> Tuple[Dict[int, np.ndarray], List[str]]:
        residue_coords: Dict[int, np.ndarray] = {}
        lines: List[str] = []
        with open(pdb_path) as fh:
            for line in fh:
                lines.append(line)
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    try:
                        res_num = int(line[22:26].strip())
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        residue_coords[res_num] = np.array([x, y, z])
                    except ValueError:
                        continue
        return residue_coords, lines

    pred_coords, pred_lines = _parse_ca_coords(predictedPdbPath)
    tmpl_coords, _ = _parse_ca_coords(templatePdbPath)

    common_residues = sorted(set(pred_coords) & set(tmpl_coords))
    if not common_residues:
        raise RuntimeError(
            "No common residue numbers found between predicted and template PDB"
        )

    pred_arr = np.array([pred_coords[r] for r in common_residues])
    tmpl_arr = np.array([tmpl_coords[r] for r in common_residues])

    # Kabsch algorithm.
    pred_center = pred_arr.mean(axis=0)
    tmpl_center = tmpl_arr.mean(axis=0)
    pred_centered = pred_arr - pred_center
    tmpl_centered = tmpl_arr - tmpl_center

    h_matrix = pred_centered.T @ tmpl_centered
    u_mat, _, vt_mat = np.linalg.svd(h_matrix)
    # Ensure right-handed coordinate system.
    d_sign = np.linalg.det(vt_mat.T @ u_mat.T)
    d_diag = np.diag([1.0, 1.0, d_sign])
    rotation = vt_mat.T @ d_diag @ u_mat.T
    translation = tmpl_center - rotation @ pred_center

    # Compute RMSD after alignment.
    aligned_pred = (rotation @ pred_centered.T).T
    rmsd_val = float(
        np.sqrt(np.mean(np.sum((aligned_pred - tmpl_centered) ** 2, axis=1)))
    )
    logger.info("Alignment RMSD: %.3f Å over %d Cα atoms", rmsd_val, len(common_residues))

    # Apply transformation to all ATOM/HETATM records and write output.
    os.makedirs(os.path.dirname(os.path.abspath(outputPdbPath)), exist_ok=True)
    with open(outputPdbPath, "w") as out_fh:
        for line in pred_lines:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coord = np.array([x, y, z])
                    new_coord = rotation @ coord + translation
                    new_line = (
                        line[:30]
                        + f"{new_coord[0]:8.3f}{new_coord[1]:8.3f}{new_coord[2]:8.3f}"
                        + line[54:]
                    )
                    out_fh.write(new_line)
                except ValueError:
                    out_fh.write(line)
            else:
                out_fh.write(line)

    logger.info("Aligned structure written to %s", outputPdbPath)
    return rmsd_val
