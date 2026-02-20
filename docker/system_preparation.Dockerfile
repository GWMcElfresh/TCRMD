# Docker image for the System Preparation module.
# Dependencies: pdbfixer, openmm, propka.
FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/system_preparation.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY tcrmd/system_preparation.py /app/tcrmd/system_preparation.py
COPY tcrmd/__init__.py            /app/tcrmd/__init__.py
COPY tests/data/                  /app/tests/data/
COPY tests/test_system_preparation.py /app/tests/test_system_preparation.py

ENV PATH="/opt/conda/bin:${PATH}"
ENTRYPOINT ["/opt/conda/bin/python", "-m", "pytest"]
CMD ["/app/tests/test_system_preparation.py", "-v"]
