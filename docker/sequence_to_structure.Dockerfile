# Docker image for the Sequence-to-Structure module.
# Dependencies: numpy (for Kabsch alignment) + boltz (for structural inference).
FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/sequence_to_structure.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY tcrmd/sequence_to_structure.py /app/tcrmd/sequence_to_structure.py
COPY tcrmd/__init__.py              /app/tcrmd/__init__.py
COPY tests/data/                    /app/tests/data/
COPY tests/test_sequence_to_structure.py /app/tests/test_sequence_to_structure.py

ENV PATH="/opt/conda/bin:${PATH}"
ENTRYPOINT ["/opt/conda/bin/python", "-m", "pytest"]
CMD ["/app/tests/test_sequence_to_structure.py", "-v"]
