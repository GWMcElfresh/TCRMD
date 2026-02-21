# Stage 1: deps — install the conda environment only.
# Includes mdanalysis, numpy, pytest.
FROM mambaorg/micromamba:1.5.8 AS deps

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/inference_analytics.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

# Stage 2: runtime — add application code.
FROM deps AS runtime
WORKDIR /app
COPY tcrmd/ /app/tcrmd/
COPY tests/  /app/tests/
