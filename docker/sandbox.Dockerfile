# 每个 run 独立 venv;JUnit 是 pytest 内置功能。
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}
RUN pip install --no-cache-dir uv && apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir pytest 'pydantic>=2.7' pydantic-settings && useradd -m -u 1000 agent && chown -R agent:agent /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
