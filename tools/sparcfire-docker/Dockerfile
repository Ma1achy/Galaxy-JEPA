# Official SpArcFiRe (compiled MATLAB, Runtime 9.2 / R2017a) for Galaxy-JEPA Brief BB0c.
# Follows SpArcFiRe's UbuntuSetup.sh; x86-64 only, run under emulation on Apple silicon.
# The SpArcFiRe repo is mounted at /sparcfire at run time, not copied in. The Runtime installer is
# fetched separately into this directory (see README.md); neither is committed.
FROM --platform=linux/amd64 ubuntu:18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      unzip wget git gcc make gawk file bc xz-utils imagemagick sextractor \
      python2.7 python-pip python-setuptools python-numpy python-scipy python-pil python-astropy \
      libxt6 libxmu6 libxext6 libxrender1 libxtst6 libxi6 libxft2 libxcomposite1 libxcursor1 \
      libxdamage1 libxrandr2 libasound2 libnss3 ca-certificates procps \
    && ln -s /usr/bin/sextractor /usr/local/bin/sex \
    && ln -sf /usr/bin/python2.7 /usr/local/bin/python \
    && ln -sf /usr/bin/python2.7 /usr/local/bin/python2 \
    && rm -rf /var/lib/apt/lists/*
# setup.bash lists 'tsv' among its packages; its check is bypassed because SPARCFIRE_HOME is set below
RUN pip2 install --no-cache-dir "tsv==1.2" || true
COPY MCR_R2017a_glnxa64_installer.zip /tmp/mcr.zip
RUN mkdir /tmp/mcr && unzip -q /tmp/mcr.zip -d /tmp/mcr \
    && /tmp/mcr/install -mode silent -agreeToLicense yes -destinationFolder /pkg/matlab \
    && mv /pkg/matlab/v92 /pkg/matlab/R2017a \
    && rm -rf /tmp/mcr /tmp/mcr.zip
ENV SPARCFIRE_HOME=/sparcfire
WORKDIR /work
# C headers + g++ for the repo's own helpers (regression-test-all.sh compiles one; the driver runs make all)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential time && rm -rf /var/lib/apt/lists/*
