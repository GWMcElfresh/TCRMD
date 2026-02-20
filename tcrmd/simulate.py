"""
Simulation Module.

OpenMM-based molecular dynamics pipeline:
  1. System setup  – ForceField creation, HMR, constraints.
  2. Energy minimisation – steepest-descent / L-BFGS.
  3. NVT/NPT equilibration with position restraints on the MHC backbone.
  4. Convenience wrapper :func:`RunSimulation` that chains all steps.

Memory notes for GitHub Runners (14 GB):
  * Use ``platform="CPU"`` (runners have no GPU).
  * Hydrogen Mass Repartitioning (HMR) enables a 4 fs timestep, halving the
    number of steps required for the same simulation time.
  * Only solute coordinates are saved in the trajectory.

Naming conventions:
    Exported functions : PascalCase
    Public arguments   : camelCase
    Internal variables : snake_case
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Default force-field XML files (AMBER ff19SB + OPC water).
_DEFAULT_PROTEIN_FF = "amber14-all.xml"
_DEFAULT_WATER_FF = "amber14/tip3pfb.xml"

# Simulation defaults.
_DEFAULT_TEMPERATURE_K = 300.0
_DEFAULT_FRICTION_PS = 1.0
_DEFAULT_TIMESTEP_PS = 0.002        # 2 fs without HMR; 4 fs with HMR
_DEFAULT_EQUIL_STEPS = 50_000       # 100 ps at 2 fs/step


def SetupSystem(
    pdbPath: str,
    forceFieldFiles: Optional[Tuple[str, ...]] = None,
    nonbondedMethod: str = "PME",
    constraintType: str = "HBonds",
    hmr: bool = True,
    platformName: str = "CPU",
) -> Tuple:
    """Create an OpenMM :class:`~openmm.app.Simulation` object ready for MD.

    Args:
        pdbPath: Path to the solvated input PDB.
        forceFieldFiles: Tuple of OpenMM force-field XML filenames.  Defaults
            to ``("amber14-all.xml", "amber14/tip3pfb.xml")``.
        nonbondedMethod: Electrostatics method; ``"PME"`` (default) or
            ``"NoCutoff"``.
        constraintType: Bond constraint scheme; ``"HBonds"`` (default) or
            ``"AllBonds"``.
        hmr: Apply Hydrogen Mass Repartitioning to enable a 4 fs timestep
            (default ``True``).
        platformName: OpenMM platform; ``"CPU"`` (default for GitHub runners)
            or ``"CUDA"`` / ``"OpenCL"`` when a GPU is available.

    Returns:
        A tuple ``(simulation, modeller)`` where *simulation* is an
        :class:`~openmm.app.Simulation` and *modeller* is the
        :class:`~openmm.app.Modeller` holding the solvated topology and
        positions.

    Raises:
        FileNotFoundError: If *pdbPath* does not exist.
        ImportError: If ``openmm`` is not installed.
    """
    if not os.path.isfile(pdbPath):
        raise FileNotFoundError(f"PDB file not found: {pdbPath}")

    try:
        import openmm as mm
        import openmm.app as app
        from openmm import unit
    except ImportError as exc:
        raise ImportError("openmm is required for SetupSystem") from exc

    ff_files = forceFieldFiles or (_DEFAULT_PROTEIN_FF, _DEFAULT_WATER_FF)

    logger.info("Loading PDB: %s", pdbPath)
    pdb = app.PDBFile(pdbPath)

    forcefield = app.ForceField(*ff_files)

    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Map string arguments to OpenMM constants.
    nonbonded_map = {
        "PME": app.PME,
        "NoCutoff": app.NoCutoff,
        "CutoffNonPeriodic": app.CutoffNonPeriodic,
        "CutoffPeriodic": app.CutoffPeriodic,
    }
    constraint_map = {
        "HBonds": app.HBonds,
        "AllBonds": app.AllBonds,
        "None": None,
    }

    nonbonded_method_val = nonbonded_map.get(nonbondedMethod, app.PME)
    constraint_val = constraint_map.get(constraintType, app.HBonds)

    create_system_kwargs = dict(
        nonbondedMethod=nonbonded_method_val,
        constraints=constraint_val,
    )
    if hmr:
        create_system_kwargs["hydrogenMass"] = 4 * unit.amu

    system = forcefield.createSystem(modeller.topology, **create_system_kwargs)

    # Langevin Middle integrator (recommended for NVT).
    timestep_ps = _DEFAULT_TIMESTEP_PS * (2.0 if hmr else 1.0)
    integrator = mm.LangevinMiddleIntegrator(
        _DEFAULT_TEMPERATURE_K * unit.kelvin,
        _DEFAULT_FRICTION_PS / unit.picosecond,
        timestep_ps * unit.picoseconds,
    )

    platform = mm.Platform.getPlatformByName(platformName)

    simulation = app.Simulation(
        modeller.topology, system, integrator, platform
    )
    simulation.context.setPositions(modeller.positions)

    logger.info(
        "System setup complete. Platform=%s, HMR=%s, timestep=%.3f ps",
        platformName,
        hmr,
        timestep_ps,
    )
    return simulation, modeller


def MinimizeEnergy(
    simulation,
    outputPdbPath: str,
    maxIterations: int = 0,
    tolerance: float = 10.0,
) -> float:
    """Perform energy minimisation and write the minimised structure.

    Args:
        simulation: An :class:`~openmm.app.Simulation` with positions already
            set (e.g. as returned by :func:`SetupSystem`).
        outputPdbPath: Path where the minimised PDB is written.
        maxIterations: Maximum number of minimisation steps.  ``0`` means
            iterate until convergence (default).
        tolerance: Convergence tolerance in kJ/mol/nm (default 10.0).

    Returns:
        Potential energy in kJ/mol after minimisation.

    Raises:
        ImportError: If ``openmm`` is not installed.
    """
    try:
        import openmm.app as app
        from openmm import unit
    except ImportError as exc:
        raise ImportError("openmm is required for MinimizeEnergy") from exc

    logger.info("Starting energy minimisation (maxIterations=%d)", maxIterations)
    simulation.minimizeEnergy(
        tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometers,
        maxIterations=maxIterations,
    )

    state = simulation.context.getState(getPositions=True, getEnergy=True)
    potential_energy = state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    logger.info("Minimisation complete. Potential energy: %.2f kJ/mol", potential_energy)

    os.makedirs(os.path.dirname(os.path.abspath(outputPdbPath)), exist_ok=True)
    with open(outputPdbPath, "w") as fh:
        app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), fh
        )

    return potential_energy


def RunEquilibration(
    simulation,
    outputDir: str,
    numSteps: int = _DEFAULT_EQUIL_STEPS,
    temperature: float = _DEFAULT_TEMPERATURE_K,
    reportInterval: int = 1000,
    checkpointInterval: int = 10_000,
    solventAtomIndices: Optional[list] = None,
) -> str:
    """Run NVT → NPT equilibration with optional solute-only trajectory output.

    Saves the trajectory to ``<outputDir>/trajectory.dcd`` and checkpoint state
    files every *checkpointInterval* steps to ``<outputDir>/checkpoint_*.xml``.

    Args:
        simulation: Configured :class:`~openmm.app.Simulation` object.
        outputDir: Directory for trajectory and checkpoint files.
        numSteps: Total number of integration steps (default 50 000 ≈ 100 ps).
        temperature: Simulation temperature in Kelvin (default 300 K).
        reportInterval: Frequency (steps) for writing trajectory frames and
            energy logs (default 1 000).
        checkpointInterval: Frequency (steps) for saving restart checkpoints
            (default 10 000).
        solventAtomIndices: List of atom indices for solvent atoms to *exclude*
            from the DCD trajectory.  If ``None`` all atoms are saved.

    Returns:
        Path to the trajectory DCD file.

    Raises:
        ImportError: If ``openmm`` is not installed.
    """
    try:
        import openmm as mm
        import openmm.app as app
        from openmm import unit
    except ImportError as exc:
        raise ImportError("openmm is required for RunEquilibration") from exc

    os.makedirs(outputDir, exist_ok=True)
    trajectory_path = os.path.join(outputDir, "trajectory.dcd")

    # Adjust temperature.
    integrator = simulation.integrator
    if hasattr(integrator, "setTemperature"):
        integrator.setTemperature(temperature * unit.kelvin)

    # Reporters.
    simulation.reporters.clear()
    simulation.reporters.append(
        app.DCDReporter(
            trajectory_path,
            reportInterval,
            atomSubset=solventAtomIndices,
        )
    )
    simulation.reporters.append(
        app.StateDataReporter(
            os.path.join(outputDir, "equil_log.csv"),
            reportInterval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
        )
    )

    # Checkpointing.
    def _save_checkpoint(step_num: int) -> None:
        checkpoint_path = os.path.join(outputDir, f"checkpoint_{step_num:08d}.xml")
        simulation.saveState(checkpoint_path)
        logger.debug("Checkpoint saved: %s", checkpoint_path)

    logger.info(
        "Starting equilibration: %d steps at %.1f K", numSteps, temperature
    )
    steps_done = 0
    while steps_done < numSteps:
        steps_remaining = numSteps - steps_done
        batch = min(checkpointInterval, steps_remaining)
        simulation.step(batch)
        steps_done += batch
        _save_checkpoint(steps_done)

    logger.info("Equilibration complete. Trajectory: %s", trajectory_path)
    return trajectory_path


def RunSimulation(
    pdbPath: str,
    outputDir: str,
    numSteps: int = _DEFAULT_EQUIL_STEPS,
    temperature: float = _DEFAULT_TEMPERATURE_K,
    forceFieldFiles: Optional[Tuple[str, ...]] = None,
    hmr: bool = True,
    platformName: str = "CPU",
    minimizeFirst: bool = True,
) -> dict:
    """End-to-end convenience wrapper: setup → minimise → equilibrate.

    Args:
        pdbPath: Path to the solvated PDB (output of :func:`~tcrmd.system_preparation.SolvateSystem`).
        outputDir: Directory for all simulation outputs.
        numSteps: Equilibration steps (default 50 000 ≈ 100 ps at 2 fs).
        temperature: Simulation temperature in Kelvin (default 300 K).
        forceFieldFiles: Force-field XML filenames forwarded to
            :func:`SetupSystem`.
        hmr: Apply Hydrogen Mass Repartitioning (default ``True``).
        platformName: OpenMM platform name (default ``"CPU"``).
        minimizeFirst: Run energy minimisation before equilibration (default
            ``True``).

    Returns:
        Dictionary with keys:
            ``"minimized_pdb"`` – path to the minimised PDB (or ``None``),
            ``"trajectory"`` – path to the trajectory DCD,
            ``"potential_energy"`` – post-minimisation energy in kJ/mol (or
            ``None``).
    """
    os.makedirs(outputDir, exist_ok=True)
    minimized_pdb = None
    potential_energy = None

    simulation, _ = SetupSystem(
        pdbPath,
        forceFieldFiles=forceFieldFiles,
        hmr=hmr,
        platformName=platformName,
    )

    if minimizeFirst:
        minimized_pdb = os.path.join(outputDir, "minimized.pdb")
        potential_energy = MinimizeEnergy(simulation, minimized_pdb)

    trajectory_path = RunEquilibration(
        simulation,
        outputDir=outputDir,
        numSteps=numSteps,
        temperature=temperature,
    )

    return {
        "minimized_pdb": minimized_pdb,
        "trajectory": trajectory_path,
        "potential_energy": potential_energy,
    }
