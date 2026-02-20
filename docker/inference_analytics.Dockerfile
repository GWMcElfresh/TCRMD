# Docker image for the Inference & Analytics module.
# Dependencies: mdanalysis, numpy.
FROM mambaorg/micromamba:1.5.8

COPY --chown=$MAMBA_USER:$MAMBA_USER envs/inference_analytics.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY tcrmd/inference_analytics.py /app/tcrmd/inference_analytics.py
COPY tcrmd/__init__.py             /app/tcrmd/__init__.py
COPY tests/data/                   /app/tests/data/
COPY tests/test_inference_analytics.py /app/tests/test_inference_analytics.py

ENV PATH="/opt/conda/bin:${PATH}"
ENTRYPOINT ["/opt/conda/bin/python", "-m", "pytest"]
CMD ["/app/tests/test_inference_analytics.py", "-v"]
