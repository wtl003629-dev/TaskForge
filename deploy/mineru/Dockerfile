# TaskForge's locked MinerU sidecar. This is based on the official MinerU
# 3.4.4 global Dockerfile, but replaces its moving >= dependency with an exact
# package version. TaskForge verifies the same value through /health.
FROM vllm/vllm-openai:v0.21.0@sha256:a230095847e93bd4df9888b33dab956fa9504537b828a23657d2b26fed57b5c9

ARG MINERU_VERSION=3.4.4

ENV HF_HOME=/opt/mineru-cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/opt/mineru-cache/huggingface/hub \
    MODELSCOPE_CACHE=/opt/mineru-cache/modelscope \
    MINERU_TOOLS_CONFIG_JSON=/opt/mineru-cache/config/mineru.json

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        "mineru[core]==${MINERU_VERSION}" \
        --break-system-packages && \
    test "$(python3 -c 'from mineru.version import __version__; print(__version__)')" = "${MINERU_VERSION}"

ENV MINERU_MODEL_SOURCE=local \
    MINERU_API_OUTPUT_ROOT=/var/lib/mineru/output

ENTRYPOINT ["mineru-api"]
CMD ["--host", "0.0.0.0", "--port", "8000"]
