# Ivy Memory Demos

We provide a simple set of interactive demos for the Ivy Memory library.
Running these demos is quick and simple.

## Install

First, clone this repo:

```bash
git clone https://github.com/unifyai/memory.git ~/ivy_memory
```

The interactive demos optionally make use of the simulator
[CoppeliaSim](https://www.coppeliarobotics.com/),
and the python wrapper [PyRep](https://github.com/stepjam/PyRep).

If these are not installed, the demos will all still run, but will display pre-rendered images from the simultator.

### Local

For a local installation, first install the dependencies:

```bash
cd ~/ivy_memory
python3 -m pip install -r requirements.txt
cd ~/ivy_memory/ivy_memory_demos
python3 -m pip install -r requirements.txt
```

To run interactive demos inside a simulator, CoppeliaSim and PyRep should then be installed following the installation [intructions](https://github.com/stepjam/PyRep#install).

### Docker

For a docker installation, first ensure [docker](https://docs.docker.com/get-docker/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) are installed.

Then simply pull the ivy memory image:

```bash
docker pull unifyai/ivy-memory:latest
```

## Demos

All demos can be run by executing the python scripts directly.
If a demo script is run without command line arguments, then a random backend framework will be selected from those installed.
Alternatively, the `--framework` argument can be used to manually specify a framework from the options
`jax`, `tensorflow`, `torch`, `mxnet` or `numpy`.

The examples below assume a docker installation, but the demo scripts can also
be run with python directly for local installations.


### Run Through

For a basic run through the library:

```bash
cd ~/ivy_memory/ivy_memory_demos
./run_demo.sh run_through
```

This script, and the various parts of the library, are further discussed in the [Run Through](https://github.com/unifyai/memory#run-through) section of the main README.
We advise following along with this section for maximum effect. The demo script should also be opened locally,
and breakpoints added to step in at intermediate points to further explore.

To run the script using a specific backend, tensorflow for example, then run like so:

```bash
./run_demo.sh run_through --framework tensorflow
```

### Learning to Copy with NTM

In this demo, a Neural Turing Machine (NTM) is trained to copy n-bit sequences
from one memory bank to a new memory bank.
Predictions are visualized throughout training.
To overfit to a single sequence, simply use the command line `--overfit` flag.

```bash
cd ~/ivy_memory/ivy_memory_demos
./run_demo.sh interactive.learning_to_copy_with_ntm
```

An example of overfitting on copying a single sequence,
with real-time training speed, is given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_memory/demo_a.gif?raw=true'>
</p>

### Mapping a Room with ESM

In this demo, the user is able to interactively drag a drone with a single monocular camera through a scene.
The ESM memory module is used to incrementally map the geometry.

```bash
cd ~/ivy_memory/ivy_memory_demos
./run_demo.sh interactive.mapping_a_room_with_esm
```

An example of rotating the drone without translation to create an egocentric ESM map of a room is given below.
The raw image observations are shown on the left,
and the incrementally constructed omni-directional ESM feature and depth images are shown on the right.
While this example only projects color values into the memory, arbitrary neural network features can also be projected, for end-to-end training.

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/ivy_memory/demo_b.gif?raw=true'>
</p>

## Get Involved

If you have any issues running any of the demos, would like to request further demos, or would like to implement your own, then get it touch.
Feature requests, pull requests, and [tweets](https://twitter.com/unify_ai) all **welcome**!