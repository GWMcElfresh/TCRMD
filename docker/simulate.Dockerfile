# Docker image for the Simulation module.
# Dependencies: openmm, pdbfixer (for test fixtures that build a solvated PDB).
FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/simulate.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY tcrmd/simulate.py          /app/tcrmd/simulate.py
COPY tcrmd/system_preparation.py /app/tcrmd/system_preparation.py
COPY tcrmd/__init__.py          /app/tcrmd/__init__.py
COPY tests/data/                /app/tests/data/
COPY tests/test_simulate.py     /app/tests/test_simulate.py

ENV PATH="/opt/conda/bin:${PATH}"
ENTRYPOINT ["/opt/conda/bin/python", "-m", "pytest"]
CMD ["/app/tests/test_simulate.py", "-v"]
