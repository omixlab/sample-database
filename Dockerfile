# Image for the webserver and the Celery prediction worker. Both run the same code and the same
# conda environment -- they differ only in the command they start -- so one image serves both and
# there is no way for the worker to end up with a different scikit-learn than the server that
# trained the models it loads.
#
# The pipeline's other environment (sample-predictors) is NOT installed here. It exists to run
# third-party predictors during step 16, which is offline batch work, not part of serving.
FROM continuumio/miniconda3:latest

WORKDIR /app

# System packages RDKit and PyMuPDF need at runtime, and curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxrender1 \
        libxext6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# The environment is built before the source is copied, so editing code does not invalidate the
# conda layer -- by far the most expensive step in this build.
COPY environments/sample.yml /tmp/sample.yml
RUN conda env create -f /tmp/sample.yml && conda clean -afy

# Every later RUN, and the container's command, execute inside the environment.
SHELL ["conda", "run", "--no-capture-output", "-n", "sample", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/sample/bin:$PATH

COPY pyproject.toml README.md ./
COPY sample ./sample
RUN pip install --no-deps -e .

# Written to by the worker; mounted from the host in docker-compose so results survive a rebuild.
ENV SAMPLE_PREDICTIONS_DB=/app/data/predictions.db

EXPOSE 8000

# Overridden by the worker service. Two workers rather than more: each may hold a loaded
# transformer, and that memory does not divide. uvicorn rather than gunicorn because the app is
# ASGI now -- gunicorn would need a worker class to run it at all.
CMD ["uvicorn", "sample.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
