#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_memory ivydl/ivy-memory:latest python3 -m pytest ivy_memory_tests/
