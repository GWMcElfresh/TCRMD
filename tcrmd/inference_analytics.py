"""
Inference & Analytics Module.

Post-simulation analysis for the TCR-pMHC ternary complex:
  * RMSF of CDR3 loops.
  * Hydrogen-bond persistence maps between TCR and peptide.
  * Residue–residue contact maps at the peptide–CDR3 interface.
  * Buried Surface Area (BSA) as a proxy for binding affinity.
  * Centre-of-Mass (COM) distance between TCR V-domains and MHC helices.

All heavy analysis is delegated to MDAnalysis and NumPy.  Matplotlib is used
for figure output.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import logging
import os
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _load_universe(topologyPath: str, trajectoryPath: Optional[str]):
    """Return an MDAnalysis Universe; raise if files are missing."""
    if not os.path.isfile(topologyPath):
        raise FileNotFoundError(f"Topology file not found: {topologyPath}")

    if trajectoryPath is not None and not os.path.isfile(trajectoryPath):
        raise FileNotFoundError(f"Trajectory file not found: {trajectoryPath}")

    try:
        import MDAnalysis as mda
    except ImportError as exc:
        raise ImportError("MDAnalysis is required for analytics functions") from exc

    if trajectoryPath is not None:
        return mda.Universe(topologyPath, trajectoryPath)
    return mda.Universe(topologyPath)


def ComputeRMSF(
    topologyPath: str,
    trajectoryPath: str,
    selection: str = "protein and name CA",
    outputPath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-residue Root Mean Square Fluctuations (RMSF) of Cα atoms.

    RMSF quantifies the dynamic flexibility of each residue over the
    trajectory.  High RMSF values in the CDR3 loops indicate regions of
    conformational plasticity at the TCR–peptide interface.

    Args:
        topologyPath: Path to the topology PDB or PSF file.
        trajectoryPath: Path to the trajectory file (DCD/XTC).
        selection: MDAnalysis atom selection string (default: all Cα atoms in
            protein chains).
        outputPath: If provided, a CSV file ``<outputPath>`` is written with
            columns ``resid,resname,rmsf_angstrom``.

    Returns:
        A tuple ``(resids, rmsf_values)`` where both are 1-D NumPy arrays of
        length equal to the number of selected residues.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ImportError: If ``MDAnalysis`` is not installed.
    """
    # Check file existence before attempting any MDAnalysis import.
    if not os.path.isfile(topologyPath):
        raise FileNotFoundError(f"Topology file not found: {topologyPath}")
    if not os.path.isfile(trajectoryPath):
        raise FileNotFoundError(f"Trajectory file not found: {trajectoryPath}")

    try:
        from MDAnalysis.analysis import rms
    except ImportError as exc:
        raise ImportError("MDAnalysis is required for ComputeRMSF") from exc

    universe = _load_universe(topologyPath, trajectoryPath)
    atoms = universe.select_atoms(selection)

    if len(atoms) == 0:
        raise ValueError(f"Selection '{selection}' matched no atoms.")

    rmsf_analysis = rms.RMSF(atoms).run()
    rmsf_values = rmsf_analysis.results.rmsf
    resids = atoms.resids

    logger.info(
        "RMSF computed for %d residues. Mean=%.3f Å, Max=%.3f Å",
        len(resids),
        float(rmsf_values.mean()),
        float(rmsf_values.max()),
    )

    if outputPath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(outputPath)), exist_ok=True)
        resnames = atoms.resnames
        header = "resid,resname,rmsf_angstrom"
        data = np.column_stack(
            [resids, resnames, rmsf_values.round(4)]
        )
        np.savetxt(
            outputPath,
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%s",
        )
        logger.info("RMSF data written to %s", outputPath)

    return resids, rmsf_values


def ComputeHydrogenBondMap(
    topologyPath: str,
    trajectoryPath: str,
    donorSelection: str = "protein",
    acceptorSelection: str = "protein",
    distanceCutoff: float = 3.5,
    angleCutoff: float = 150.0,
    outputPath: Optional[str] = None,
) -> np.ndarray:
    """Build a hydrogen-bond persistence map between TCR and peptide residues.

    For each donor–acceptor residue pair, reports the fraction of trajectory
    frames in which a hydrogen bond is present (donor–acceptor distance ≤
    *distanceCutoff* Å **and** D–H···A angle ≥ *angleCutoff* degrees).

    Args:
        topologyPath: Path to the topology file.
        trajectoryPath: Path to the trajectory file.
        donorSelection: MDAnalysis selection for H-bond donors (default:
            ``"protein"``).
        acceptorSelection: MDAnalysis selection for H-bond acceptors (default:
            ``"protein"``).
        distanceCutoff: Maximum donor–acceptor distance in Å (default 3.5).
        angleCutoff: Minimum D–H···A angle in degrees (default 150.0).
        outputPath: If provided, a NumPy ``.npy`` file is saved at this path.

    Returns:
        A 2-D NumPy array of shape ``(N_donors, N_acceptors)`` with persistence
        fractions in ``[0, 1]``.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ImportError: If ``MDAnalysis`` is not installed.
    """
    try:
        from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
    except ImportError as exc:
        raise ImportError("MDAnalysis is required for ComputeHydrogenBondMap") from exc

    universe = _load_universe(topologyPath, trajectoryPath)

    # Build residue maps from the donor/acceptor selections.
    donor_atoms = universe.select_atoms(donorSelection)
    acceptor_atoms = universe.select_atoms(acceptorSelection)

    donor_resids = np.unique(donor_atoms.resids)
    acceptor_resids = np.unique(acceptor_atoms.resids)
    n_donors = len(donor_resids)
    n_acceptors = len(acceptor_resids)
    n_frames = universe.trajectory.n_frames

    persistence_matrix = np.zeros((n_donors, n_acceptors), dtype=np.float64)

    # H-bond analysis requires explicit hydrogen atoms and bond topology.
    # If neither is present (e.g. a coarse heavy-atom PDB) we return the
    # zero matrix and emit a warning rather than crashing.
    h_atoms = universe.select_atoms("element H or name H*")
    if len(h_atoms) == 0:
        logger.warning(
            "No hydrogen atoms found in topology '%s'. "
            "H-bond analysis requires explicit H atoms; returning zero persistence matrix.",
            topologyPath,
        )
    else:
        try:
            hb_analysis = HydrogenBondAnalysis(
                universe,
                donors_sel=donorSelection,
                acceptors_sel=acceptorSelection,
                d_a_cutoff=distanceCutoff,
                d_h_a_angle_cutoff=angleCutoff,
            )
            hb_analysis.run()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "HydrogenBondAnalysis failed (%s: %s); returning zero persistence matrix.",
                type(exc).__name__,
                exc,
            )
        else:
            if n_frames > 0 and hb_analysis.results.hbonds.size > 0:
                donor_resid_map = {rid: i for i, rid in enumerate(donor_resids)}
                acceptor_resid_map = {rid: i for i, rid in enumerate(acceptor_resids)}

                for row in hb_analysis.results.hbonds:
                    _, donor_idx, _, acceptor_idx, _, _ = row
                    donor_atom = universe.atoms[int(donor_idx)]
                    acceptor_atom = universe.atoms[int(acceptor_idx)]
                    d_rid = donor_atom.resid
                    a_rid = acceptor_atom.resid
                    if d_rid in donor_resid_map and a_rid in acceptor_resid_map:
                        di = donor_resid_map[d_rid]
                        ai = acceptor_resid_map[a_rid]
                        persistence_matrix[di, ai] += 1.0

                persistence_matrix /= n_frames

    logger.info(
        "H-bond persistence map: %d donor residues × %d acceptor residues",
        n_donors,
        n_acceptors,
    )

    if outputPath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(outputPath)), exist_ok=True)
        np.save(outputPath, persistence_matrix)
        logger.info("H-bond map saved to %s", outputPath)

    return persistence_matrix


def ComputeContactMap(
    topologyPath: str,
    trajectoryPath: str,
    selectionA: str,
    selectionB: str,
    cutoff: float = 4.5,
    outputPath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a residue–residue contact map for the peptide–CDR3 interface.

    A contact is defined as any pair of heavy atoms (one from group A, one from
    group B) within *cutoff* Å.  The fraction of frames in which each residue
    pair is in contact is stored in the output matrix.

    Args:
        topologyPath: Path to the topology file.
        trajectoryPath: Path to the trajectory file.
        selectionA: MDAnalysis atom selection for the first group (e.g. CDR3
            loops of the TCR).
        selectionB: MDAnalysis atom selection for the second group (e.g.
            peptide + MHC groove residues).
        cutoff: Distance cutoff in Å (default 4.5).
        outputPath: If provided, a NumPy ``.npy`` file is saved at this path.

    Returns:
        A tuple ``(resids_A, resids_B, contact_matrix)`` where *contact_matrix*
        has shape ``(len(resids_A), len(resids_B))`` and values in ``[0, 1]``.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ValueError: If either selection matches no atoms.
        ImportError: If ``MDAnalysis`` is not installed.
    """
    universe = _load_universe(topologyPath, trajectoryPath)

    group_a = universe.select_atoms(selectionA)
    group_b = universe.select_atoms(selectionB)

    if len(group_a) == 0:
        raise ValueError(f"selectionA '{selectionA}' matched no atoms.")
    if len(group_b) == 0:
        raise ValueError(f"selectionB '{selectionB}' matched no atoms.")

    resids_a = np.unique(group_a.resids)
    resids_b = np.unique(group_b.resids)
    n_a = len(resids_a)
    n_b = len(resids_b)
    n_frames = universe.trajectory.n_frames

    contact_matrix = np.zeros((n_a, n_b), dtype=np.float64)

    resid_a_map = {rid: i for i, rid in enumerate(resids_a)}
    resid_b_map = {rid: i for i, rid in enumerate(resids_b)}

    for _ in universe.trajectory:
        pos_a = group_a.positions
        pos_b = group_b.positions
        # Pairwise distances (N_a_atoms × N_b_atoms).
        diff = pos_a[:, np.newaxis, :] - pos_b[np.newaxis, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=-1))

        in_contact = dist <= cutoff  # shape (N_a_atoms, N_b_atoms)

        for atom_i, atom_a in enumerate(group_a):
            for atom_j, atom_b in enumerate(group_b):
                if in_contact[atom_i, atom_j]:
                    ri = resid_a_map.get(atom_a.resid)
                    rj = resid_b_map.get(atom_b.resid)
                    if ri is not None and rj is not None:
                        contact_matrix[ri, rj] += 1.0

    if n_frames > 0:
        contact_matrix /= n_frames

    logger.info(
        "Contact map: %d × %d residues, cutoff=%.1f Å",
        n_a,
        n_b,
        cutoff,
    )

    if outputPath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(outputPath)), exist_ok=True)
        np.save(outputPath, contact_matrix)
        logger.info("Contact map saved to %s", outputPath)

    return resids_a, resids_b, contact_matrix


def ComputeBuriedSurfaceArea(
    topologyPath: str,
    trajectoryPath: str,
    selectionA: str,
    selectionB: str,
    probeRadius: float = 1.4,
    outputPath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate Buried Surface Area (BSA) per trajectory frame.

    BSA = ½ × (SASA_A + SASA_B − SASA_complex).  A positive BSA indicates
    solvent-accessible surface that is buried upon TCR binding.

    Args:
        topologyPath: Path to the topology file.
        trajectoryPath: Path to the trajectory file.
        selectionA: MDAnalysis selection for binding partner A (e.g. TCR).
        selectionB: MDAnalysis selection for binding partner B (e.g. pMHC).
        probeRadius: Solvent probe radius in Å (default 1.4 for water).
        outputPath: If provided, a CSV with columns ``frame,bsa_angstrom2`` is
            written at this path.

    Returns:
        A tuple ``(frame_indices, bsa_values)`` where *bsa_values* contains the
        per-frame BSA in Å².

    Raises:
        FileNotFoundError: If either input file does not exist.
        ImportError: If ``MDAnalysis`` is not installed.
    """
    try:
        import MDAnalysis  # noqa: F401
    except ImportError as exc:
        raise ImportError("MDAnalysis is required for ComputeBuriedSurfaceArea") from exc

    try:
        from MDAnalysis.analysis.solvent_accessibility import ShrakeRupley
    except ImportError as exc:
        raise ImportError(
            "MDAnalysis>=2.4 with ShrakeRupley is required for ComputeBuriedSurfaceArea"
        ) from exc

    universe = _load_universe(topologyPath, trajectoryPath)

    group_a = universe.select_atoms(selectionA)
    group_b = universe.select_atoms(selectionB)
    complex_group = group_a | group_b

    # Run each ShrakeRupley analysis over the full trajectory in a separate
    # pass.  Calling ShrakeRupley.run() inside a "for ts in trajectory:" loop
    # would reset the trajectory position on every frame, causing incorrect
    # frame ordering.  Three independent passes are correct and deterministic.
    sr_a = ShrakeRupley(group_a, probe_radius=probeRadius)
    sr_a.run()

    sr_b = ShrakeRupley(group_b, probe_radius=probeRadius)
    sr_b.run()

    sr_complex = ShrakeRupley(complex_group, probe_radius=probeRadius)
    sr_complex.run()

    n_frames = len(universe.trajectory)
    frame_indices = list(range(n_frames))
    bsa_values = []
    for i in range(n_frames):
        sasa_a = float(sr_a.results.areas[i].sum())
        sasa_b = float(sr_b.results.areas[i].sum())
        sasa_complex = float(sr_complex.results.areas[i].sum())
        bsa_values.append(0.5 * (sasa_a + sasa_b - sasa_complex))

    frame_arr = np.array(frame_indices, dtype=int)
    bsa_arr = np.array(bsa_values, dtype=float)

    logger.info(
        "BSA: mean=%.1f Å², max=%.1f Å² over %d frames",
        float(bsa_arr.mean()),
        float(bsa_arr.max()),
        len(bsa_arr),
    )

    if outputPath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(outputPath)), exist_ok=True)
        header = "frame,bsa_angstrom2"
        np.savetxt(
            outputPath,
            np.column_stack([frame_arr, bsa_arr.round(2)]),
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d", "%.2f"],
        )
        logger.info("BSA data written to %s", outputPath)

    return frame_arr, bsa_arr


def ComputeCOMDistance(
    topologyPath: str,
    trajectoryPath: str,
    selectionA: str,
    selectionB: str,
    outputPath: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track the Centre-of-Mass (COM) distance between two molecular groups.

    Measures the separation between the TCR V-alpha/V-beta domains and the MHC
    helices over the simulation.  A decreasing COM distance may indicate TCR
    docking; a sudden increase can signal unbinding.

    Args:
        topologyPath: Path to the topology file.
        trajectoryPath: Path to the trajectory file.
        selectionA: MDAnalysis selection for the first group (e.g. TCR
            V-alpha/V-beta).
        selectionB: MDAnalysis selection for the second group (e.g. MHC
            alpha-1/alpha-2 helices).
        outputPath: If provided, a CSV with columns ``frame,com_distance_angstrom``
            is written at this path.

    Returns:
        A tuple ``(frame_indices, distances)`` where *distances* contains per-frame
        COM distances in Å.

    Raises:
        FileNotFoundError: If either input file does not exist.
        ValueError: If either selection matches no atoms.
        ImportError: If ``MDAnalysis`` is not installed.
    """
    universe = _load_universe(topologyPath, trajectoryPath)

    group_a = universe.select_atoms(selectionA)
    group_b = universe.select_atoms(selectionB)

    if len(group_a) == 0:
        raise ValueError(f"selectionA '{selectionA}' matched no atoms.")
    if len(group_b) == 0:
        raise ValueError(f"selectionB '{selectionB}' matched no atoms.")

    frame_indices = []
    distances = []

    for ts in universe.trajectory:
        com_a = group_a.center_of_mass()
        com_b = group_b.center_of_mass()
        dist = float(np.linalg.norm(com_a - com_b))
        frame_indices.append(ts.frame)
        distances.append(dist)

    frame_arr = np.array(frame_indices, dtype=int)
    dist_arr = np.array(distances, dtype=float)

    logger.info(
        "COM distance: mean=%.2f Å, min=%.2f Å, max=%.2f Å over %d frames",
        float(dist_arr.mean()),
        float(dist_arr.min()),
        float(dist_arr.max()),
        len(dist_arr),
    )

    if outputPath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(outputPath)), exist_ok=True)
        header = "frame,com_distance_angstrom"
        np.savetxt(
            outputPath,
            np.column_stack([frame_arr, dist_arr.round(4)]),
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d", "%.4f"],
        )
        logger.info("COM distance data written to %s", outputPath)

    return frame_arr, dist_arr
