"""
TCRMD: TCR-pMHC Ternary Complex Modeling and Molecular Dynamics Pipeline.

Modules:
    sequence_to_structure: CDR loop extraction, Boltz inference, template alignment.
    system_preparation:    PDB cleaning, protonation state assignment, solvation.
    simulate:              OpenMM energy minimization and MD equilibration.
    inference_analytics:   RMSF, hydrogen-bond maps, contact maps, BSA, COM distance.
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
from tcrmd.simulate import (
    SetupSystem,
    MinimizeEnergy,
    RunEquilibration,
    RunSimulation,
)
from tcrmd.inference_analytics import (
    ComputeRMSF,
    ComputeHydrogenBondMap,
    ComputeContactMap,
    ComputeBuriedSurfaceArea,
    ComputeCOMDistance,
)

__all__ = [
    "ExtractCDRLoops",
    "FormatBoltzInput",
    "RunBoltzInference",
    "AlignToTemplate",
    "CleanPDB",
    "AssignProtonationStates",
    "SolvateSystem",
    "SetupSystem",
    "MinimizeEnergy",
    "RunEquilibration",
    "RunSimulation",
    "ComputeRMSF",
    "ComputeHydrogenBondMap",
    "ComputeContactMap",
    "ComputeBuriedSurfaceArea",
    "ComputeCOMDistance",
]
