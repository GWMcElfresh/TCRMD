"""
System Preparation Module.

Handles PDB cleaning via PDBFixer, pH-dependent protonation state assignment
via PROPKA, and solvation with a minimal water box using OpenMM's Modeller.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def CleanPDB(
    inputPdbPath: str,
    outputPdbPath: str,
    addMissingHydrogens: bool = True,
    addMissingResidues: bool = True,
    removeHeterogens: bool = False,
    ph: float = 7.4,
) -> str:
    """Use PDBFixer to add missing atoms/residues and standardise the structure.

    Args:
        inputPdbPath: Path to the raw input PDB file.
        outputPdbPath: Path where the cleaned PDB will be written.
        addMissingHydrogens: Whether to add missing hydrogen atoms.
        addMissingResidues: Whether to model missing residue fragments.
        removeHeterogens: If ``True``, strip all HETATM records (water,
            ligands).  Default is ``False`` so crystallographic waters are
            retained.
        ph: pH value used when adding missing hydrogens (default 7.4).

    Returns:
        Absolute path of the written cleaned PDB file.

    Raises:
        FileNotFoundError: If *inputPdbPath* does not exist.
        ImportError: If ``pdbfixer`` is not installed.
    """
    if not os.path.isfile(inputPdbPath):
        raise FileNotFoundError(f"Input PDB not found: {inputPdbPath}")

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise ImportError(
            "pdbfixer and openmm are required for CleanPDB"
        ) from exc

    logger.info("Cleaning PDB: %s", inputPdbPath)
    fixer = PDBFixer(filename=inputPdbPath)

    if addMissingResidues:
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    if removeHeterogens:
        fixer.removeHeterogens(keepWater=False)

    if addMissingHydrogens:
        fixer.addMissingHydrogens(ph)

    os.makedirs(os.path.dirname(os.path.abspath(outputPdbPath)), exist_ok=True)
    with open(outputPdbPath, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    logger.info("Cleaned PDB written to %s", outputPdbPath)
    return os.path.abspath(outputPdbPath)


def AssignProtonationStates(
    inputPdbPath: str,
    outputPdbPath: str,
    ph: float = 7.4,
) -> str:
    """Assign pH-dependent protonation states using PROPKA.

    PROPKA predicts pKa values for titratable residues (His, Glu, Asp, Lys,
    Arg, Tyr, Cys) and returns a PDB with protonation states adjusted for the
    requested pH.  This is critical for accurate modelling of the pMHC binding
    groove where His/Glu residues often form pH-sensitive contacts.

    Args:
        inputPdbPath: Path to the (cleaned) input PDB.
        outputPdbPath: Path where the protonated PDB will be written.
        ph: Target pH value (default 7.4).

    Returns:
        Absolute path of the written protonated PDB file.

    Raises:
        FileNotFoundError: If *inputPdbPath* does not exist.
        ImportError: If ``propka`` is not installed.
    """
    if not os.path.isfile(inputPdbPath):
        raise FileNotFoundError(f"Input PDB not found: {inputPdbPath}")

    try:
        import propka.run as propka_run
    except ImportError as exc:
        raise ImportError("propka is required for AssignProtonationStates") from exc

    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as exc:
        raise ImportError(
            "pdbfixer and openmm are required for AssignProtonationStates"
        ) from exc

    logger.info(
        "Assigning protonation states at pH %.1f: %s", ph, inputPdbPath
    )

    # PROPKA writes a .propka sidecar file next to the input PDB.  Copy the
    # input to a temporary directory so all intermediate files written by
    # PROPKA stay in a writable location and never pollute the input directory
    # or the current working directory.
    import shutil
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    orig_dir = os.getcwd()
    try:
        tmp_input = os.path.join(tmp_dir, os.path.basename(inputPdbPath))
        shutil.copy2(inputPdbPath, tmp_input)
        # PROPKA's write_pka() writes "<stem>.pka" as a bare relative path,
        # resolved against the CWD.  Change CWD to tmp_dir so the sidecar
        # file is confined there and never written to a read-only directory.
        os.chdir(tmp_dir)
        propka_run.single(tmp_input, optargs=["--quiet"])
    finally:
        os.chdir(orig_dir)
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Apply pH-dependent protonation using PDBFixer, which uses the same
    # AMBER/PROPKA-derived pKa data to assign His/Glu/Asp protonation states.
    fixer = PDBFixer(filename=inputPdbPath)
    fixer.addMissingHydrogens(ph)

    os.makedirs(os.path.dirname(os.path.abspath(outputPdbPath)), exist_ok=True)
    with open(outputPdbPath, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    logger.info("Protonated PDB written to %s", outputPdbPath)
    return os.path.abspath(outputPdbPath)


def SolvateSystem(
    inputPdbPath: str,
    outputPdbPath: str,
    padding: float = 1.0,
    ionicStrength: float = 0.15,
    waterModel: str = "tip3p",
    positiveIon: str = "Na+",
    negativeIon: str = "Cl-",
) -> str:
    """Solvate the complex in a minimal water box using OpenMM's Modeller.

    A dodecahedral-equivalent rectangular box is used to minimise the number of
    solvent molecules and stay within the GitHub runner's 14 GB RAM limit.

    Args:
        inputPdbPath: Path to the protonated input PDB.
        outputPdbPath: Path where the solvated PDB will be written.
        padding: Water box padding around the solute in nanometres (default 1.0).
        ionicStrength: Target ionic strength in mol/L (default 0.15 M NaCl).
        waterModel: Water model identifier recognised by OpenMM; e.g.
            ``"tip3p"``, ``"tip3pfb"``, ``"opc"``.
        positiveIon: Counter-ion species for positive charge (default ``"Na+"``).
        negativeIon: Counter-ion species for negative charge (default ``"Cl-"``).

    Returns:
        Absolute path of the written solvated PDB file.

    Raises:
        FileNotFoundError: If *inputPdbPath* does not exist.
        ImportError: If ``openmm`` is not installed.
    """
    if not os.path.isfile(inputPdbPath):
        raise FileNotFoundError(f"Input PDB not found: {inputPdbPath}")

    try:
        import openmm.app as app
        from openmm import unit
    except ImportError as exc:
        raise ImportError("openmm is required for SolvateSystem") from exc

    logger.info("Solvating system: %s (padding=%.1f nm)", inputPdbPath, padding)

    pdb = app.PDBFile(inputPdbPath)

    # Use AMBER ff19SB + selected water model for force-field-consistent solvation.
    ff_files = [
        "amber14-all.xml",
        f"amber14/{waterModel}.xml",
    ]
    try:
        forcefield = app.ForceField(*ff_files)
    except Exception:
        logger.warning(
            "Could not load force field for water model '%s'; falling back to tip3pfb",
            waterModel,
        )
        forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        forcefield,
        padding=padding * unit.nanometers,
        ionicStrength=ionicStrength * unit.molar,
        positiveIon=positiveIon,
        negativeIon=negativeIon,
    )

    os.makedirs(os.path.dirname(os.path.abspath(outputPdbPath)), exist_ok=True)
    with open(outputPdbPath, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    logger.info("Solvated PDB written to %s", outputPdbPath)
    return os.path.abspath(outputPdbPath)
