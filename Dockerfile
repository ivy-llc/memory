FROM ivydl/ivy:latest-copsim

# Install Ivy
RUN git clone https://github.com/ivy-dl/ivy && \
    cd ivy && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    cat optional.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

# Install Ivy Demo Utils
RUN git clone https://github.com/ivy-dl/demo-utils && \
    cd demo-utils && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

# Install Ivy Mechanics
RUN git clone https://github.com/ivy-dl/mech && \
    cd mech && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

# Install Ivy Vision
RUN git clone https://github.com/ivy-dl/vision && \
    cd vision && \
    cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    python3 setup.py install && \
    cd ..

RUN mkdir ivy_memory
WORKDIR /ivy_memory

COPY requirements.txt /ivy_memory
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    rm -rf requirements.txt

COPY demos/requirements.txt /ivy_memory
RUN cat requirements.txt | grep -v "ivy-" | pip3 install --no-cache-dir -r /dev/stdin && \
    rm -rf requirements.txt

# Entrypoint
RUN echo '#!/bin/bash\n/usr/bin/xvfb-run --auto-servernum "$@"' > /entrypoint && \
    chmod a+x /entrypoint
ENTRYPOINT ["/entrypoint"]