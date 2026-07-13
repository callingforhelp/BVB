# BVB sandbox image: a generic Blender environment that an agent drives via
# `docker exec`. The agent itself runs on the host; this image only provides a
# reproducible, isolated Blender + xvfb + ffmpeg runtime.
#
# Pinned to linux/amd64: Blender ships official Linux prebuilts for x86-64 only,
# so on Apple Silicon the whole image runs under emulation (Rosetta/qemu).
FROM --platform=linux/amd64 ubuntu:22.04

ARG BLENDER_MAJOR=4.2
ARG BLENDER_VERSION=4.2.0
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        xz-utils \
        xvfb \
        ffmpeg \
        python3 \
        python3-pip \
        libgl1-mesa-glx \
        libxi6 \
        libxrender1 \
        libxxf86vm1 \
        libxfixes3 \
        libxkbcommon0 \
        libsm6 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Blender LTS (Linux x64 prebuilt). Pinned for reproducibility.
RUN curl -fSL "https://download.blender.org/release/Blender${BLENDER_MAJOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
        -o /tmp/blender.tar.xz \
    && mkdir -p /opt/blender \
    && tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 \
    && rm /tmp/blender.tar.xz \
    && ln -s /opt/blender/blender /usr/local/bin/blender

# Helper packages an agent's bpy scripts might use.
RUN python3 -m pip install --no-cache-dir numpy Pillow

WORKDIR /workspace
RUN mkdir -p /workspace/video /workspace/frames /workspace/output /workspace/scratch /workspace/logs

# `blender_run` wraps xvfb so agents do not have to remember the prefix.
RUN printf '#!/bin/bash\nexec xvfb-run -a blender --background "$@"\n' > /usr/local/bin/blender_run \
    && chmod +x /usr/local/bin/blender_run

# Keep the container alive so the orchestrator can docker exec into it.
CMD ["tail", "-f", "/dev/null"]
