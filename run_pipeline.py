"""
run_pipeline.py – End-to-end orchestration of the TCR-pMHC modeling pipeline.

Stages:
  1. Sequence-to-Structure : CDR extraction → Boltz inference → template alignment.
  2. System Preparation    : PDB cleaning → protonation → solvation.
  3. Simulation            : Energy minimisation → NVT/NPT equilibration.
  4. Inference & Analytics : RMSF, H-bond map, contact map, BSA, COM distance.

Usage (inside Docker container or with dependencies installed)::

    python run_pipeline.py --sequences sequences.json \\
                           --template template.pdb \\
                           --checkpoint boltz1.ckpt \\
                           --output results/

All intermediate files are written under ``<outputDir>/intermediate/``.

Naming conventions:
    Public arguments   : camelCase (CLI flags are kebab-case equivalents)
    Internal variables : snake_case
"""

import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TCR-pMHC Ternary Complex Modeling Pipeline"
    )
    parser.add_argument(
        "--sequences",
        required=True,
        help="Path to JSON file mapping chain keys to amino-acid sequences.",
    )
    parser.add_argument(
        "--template",
        default=None,
        help="Path to a reference PDB template (optional).",
    )
    parser.add_argument(
        "--checkpoint",
        default="boltz1.ckpt",
        help="Path to Boltz model checkpoint (default: boltz1.ckpt).",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Top-level output directory (default: results/).",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50_000,
        help="Equilibration steps (default: 50 000 ≈ 100 ps).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Simulation temperature in K (default: 300).",
    )
    parser.add_argument(
        "--platform",
        default="CPU",
        help="OpenMM platform (default: CPU).",
    )
    parser.add_argument(
        "--ph",
        type=float,
        default=7.4,
        help="pH for protonation state assignment (default: 7.4).",
    )
    parser.add_argument(
        "--no-hmr",
        action="store_true",
        help="Disable Hydrogen Mass Repartitioning.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip Boltz inference and use --template as the predicted complex.",
    )
    return parser.parse_args(argv)


def RunPipeline(
    sequencesPath: str,
    outputDir: str,
    templatePath: str = None,
    checkpointPath: str = "boltz1.ckpt",
    numSteps: int = 50_000,
    temperature: float = 300.0,
    platformName: str = "CPU",
    ph: float = 7.4,
    hmr: bool = True,
    skipInference: bool = False,
) -> dict:
    """Run the full TCR-pMHC modeling pipeline.

    Args:
        sequencesPath: Path to a JSON file mapping chain identifiers to
            amino-acid sequences.
        outputDir: Top-level directory for all results.
        templatePath: Optional path to a reference PDB template.
        checkpointPath: Path to Boltz model checkpoint.
        numSteps: Number of MD equilibration steps.
        temperature: MD temperature in Kelvin.
        platformName: OpenMM platform name.
        ph: pH for protonation state assignment.
        hmr: Apply Hydrogen Mass Repartitioning.
        skipInference: If ``True``, skip Boltz inference and treat
            *templatePath* as the predicted complex.

    Returns:
        Dictionary summarising output file paths:
            ``"predicted_pdb"``, ``"cleaned_pdb"``, ``"protonated_pdb"``,
            ``"solvated_pdb"``, ``"minimized_pdb"``, ``"trajectory"``,
            ``"rmsf_csv"``, ``"hbond_npy"``, ``"contact_npy"``,
            ``"bsa_csv"``, ``"com_csv"``.
    """
    from tcrmd.sequence_to_structure import (
        ExtractCDRLoops,
        FormatBoltzInput,
        RunBoltzInference,
        AlignToTemplate,
    )
    from tcrmd.system_preparation import (
        CleanPDB,
        AssignProtonationStates,
        SolvateSystem,
    )
    from tcrmd.simulate import RunSimulation
    from tcrmd.inference_analytics import (
        ComputeRMSF,
        ComputeHydrogenBondMap,
        ComputeContactMap,
        ComputeBuriedSurfaceArea,
        ComputeCOMDistance,
    )

    os.makedirs(outputDir, exist_ok=True)
    intermediate_dir = os.path.join(outputDir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 1: Sequence-to-Structure                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 1: Sequence-to-Structure ===")

    with open(sequencesPath) as fh:
        sequence_data = json.load(fh)

    # Extract CDR loops for logging / downstream filtering.
    for chain in ("alpha", "beta"):
        if chain in sequence_data:
            cdr_loops = ExtractCDRLoops(sequence_data, chainType=chain)
            logger.info("CDR loops (%s): %s", chain, cdr_loops)

    if skipInference:
        if templatePath is None:
            raise ValueError(
                "--skip-inference requires --template to point to an existing PDB."
            )
        predicted_pdb = templatePath
        logger.info("Skipping Boltz inference; using template: %s", templatePath)
    else:
        boltz_input_path = FormatBoltzInput(
            sequence_data,
            templatePdbPath=templatePath,
            outputDir=intermediate_dir,
        )
        predicted_pdb = RunBoltzInference(
            boltz_input_path,
            checkpointPath=checkpointPath,
            outputDir=intermediate_dir,
        )

    # Template alignment (only if a separate template was provided).
    aligned_pdb = predicted_pdb
    if templatePath is not None and not skipInference:
        aligned_pdb = os.path.join(intermediate_dir, "aligned_complex.pdb")
        rmsd = AlignToTemplate(predicted_pdb, templatePath, aligned_pdb)
        logger.info("Aligned to template. RMSD=%.3f Å", rmsd)

    # ------------------------------------------------------------------ #
    # Stage 2: System Preparation                                          #
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 2: System Preparation ===")

    cleaned_pdb = os.path.join(intermediate_dir, "cleaned.pdb")
    CleanPDB(aligned_pdb, cleaned_pdb, ph=ph)

    protonated_pdb = os.path.join(intermediate_dir, "protonated.pdb")
    try:
        AssignProtonationStates(cleaned_pdb, protonated_pdb, ph=ph)
    except ImportError:
        logger.warning(
            "propka not available; skipping protonation assignment."
        )
        protonated_pdb = cleaned_pdb

    solvated_pdb = os.path.join(intermediate_dir, "solvated.pdb")
    SolvateSystem(protonated_pdb, solvated_pdb)

    # ------------------------------------------------------------------ #
    # Stage 3: Simulation                                                  #
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 3: Simulation ===")

    sim_output_dir = os.path.join(outputDir, "simulation")
    sim_results = RunSimulation(
        solvated_pdb,
        outputDir=sim_output_dir,
        numSteps=numSteps,
        temperature=temperature,
        hmr=hmr,
        platformName=platformName,
    )
    minimized_pdb = sim_results["minimized_pdb"]
    trajectory_path = sim_results["trajectory"]

    # ------------------------------------------------------------------ #
    # Stage 4: Inference & Analytics                                       #
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 4: Inference & Analytics ===")

    analytics_dir = os.path.join(outputDir, "analytics")
    os.makedirs(analytics_dir, exist_ok=True)

    topology_for_analysis = minimized_pdb or solvated_pdb

    rmsf_csv = os.path.join(analytics_dir, "rmsf.csv")
    ComputeRMSF(topology_for_analysis, trajectory_path, outputPath=rmsf_csv)

    hbond_npy = os.path.join(analytics_dir, "hbond_map.npy")
    ComputeHydrogenBondMap(
        topology_for_analysis,
        trajectory_path,
        outputPath=hbond_npy,
    )

    contact_npy = os.path.join(analytics_dir, "contact_map.npy")
    ComputeContactMap(
        topology_for_analysis,
        trajectory_path,
        selectionA="segid A or segid B",  # TCR chains
        selectionB="segid P",              # Peptide
        outputPath=contact_npy,
    )

    bsa_csv = os.path.join(analytics_dir, "bsa.csv")
    ComputeBuriedSurfaceArea(
        topology_for_analysis,
        trajectory_path,
        selectionA="segid A or segid B",
        selectionB="segid C or segid D or segid P",
        outputPath=bsa_csv,
    )

    com_csv = os.path.join(analytics_dir, "com_distance.csv")
    ComputeCOMDistance(
        topology_for_analysis,
        trajectory_path,
        selectionA="segid A or segid B",
        selectionB="segid C or segid D",
        outputPath=com_csv,
    )

    results = {
        "predicted_pdb": predicted_pdb,
        "cleaned_pdb": cleaned_pdb,
        "protonated_pdb": protonated_pdb,
        "solvated_pdb": solvated_pdb,
        "minimized_pdb": minimized_pdb,
        "trajectory": trajectory_path,
        "rmsf_csv": rmsf_csv,
        "hbond_npy": hbond_npy,
        "contact_npy": contact_npy,
        "bsa_csv": bsa_csv,
        "com_csv": com_csv,
    }

    summary_path = os.path.join(outputDir, "pipeline_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Pipeline complete. Summary: %s", summary_path)
    return results


def main(argv=None):
    args = _parse_args(argv)

    RunPipeline(
        sequencesPath=args.sequences,
        outputDir=args.output,
        templatePath=args.template,
        checkpointPath=args.checkpoint,
        numSteps=args.num_steps,
        temperature=args.temperature,
        platformName=args.platform,
        ph=args.ph,
        hmr=not args.no_hmr,
        skipInference=args.skip_inference,
    )


if __name__ == "__main__":
    sys.exit(main())
