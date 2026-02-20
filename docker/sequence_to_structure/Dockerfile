# Stage 1: deps — install the conda environment only.
# This stage is cached in GHCR by docker-cache.yml; rebuilds only when
# envs/sequence_to_structure.yaml (or pyproject.toml) changes.
FROM mambaorg/micromamba:1.5.8 AS deps

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/sequence_to_structure.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Stage 2: runtime — add application code on top of deps.
# Used when building a self-contained image; the docker-cache.yml workflow
# mounts the workspace at /workspace and skips this stage for CI tests.
FROM deps AS runtime
WORKDIR /app
COPY tcrmd/ /app/tcrmd/
COPY tests/  /app/tests/
