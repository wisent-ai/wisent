# coding/safe_docker/Dockerfile
FROM python:3.12-slim

# ---- base toolchain for Python/C++/Java + tini ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates build-essential g++ openjdk-21-jdk-headless tini bash time \
 && rm -rf /var/lib/apt/lists/*

# pytest for Python harnesses
RUN pip install --no-cache-dir pytest==8.3.3

# Python env niceties
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# ---- create unprivileged user (will run everything after setup) ----
RUN useradd -m -u 10001 sandbox

# Prepare writable dirs WHILE STILL ROOT, and give them to 'sandbox'
# /runner : code runner home
# /work   : ephemeral workspace (you'll mount tmpfs here at runtime)
RUN install -d -m 0755 -o sandbox -g sandbox /runner \
 && install -d -m 1777 -o sandbox -g sandbox /work

# Drop privileges for all subsequent instructions
USER sandbox
WORKDIR /runner

# Copy runner entrypoint with correct ownership
COPY --chown=sandbox:sandbox entrypoint.py /runner/entrypoint.py

ENTRYPOINT ["/usr/bin/tini","--","python","/runner/entrypoint.py"]