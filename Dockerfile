# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Build environment with micromamba
# ──────────────────────────────────────────────────────────────────────────────
FROM mambaorg/micromamba:1.5.8 AS builder

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml

RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime image (reuse the conda environment from the builder stage)
# ──────────────────────────────────────────────────────────────────────────────
FROM mambaorg/micromamba:1.5.8

# Copy the pre-built environment from Stage 1.
COPY --from=builder /opt/conda /opt/conda

# Add pipeline source code.
WORKDIR /app
COPY tcrmd/ /app/tcrmd/
COPY run_pipeline.py /app/run_pipeline.py

# Activate the micromamba base environment by default.
ENV PATH="/opt/conda/bin:${PATH}"
ENV MAMBA_ROOT_PREFIX="/opt/conda"

# Ensure python from the conda environment is used.
ENTRYPOINT ["/opt/conda/bin/python"]
CMD ["/app/run_pipeline.py", "--help"]
