# Ivy Memory Demos

We provide a simple set of interactive demos for the Ivy Memory library.
Running these demos is quick and simple.

## Install

First, clone this repo, and install the requirements provided in this demos folder like so:

```bash
git clone https://github.com/ivy-dl/memory.git ~/ivy_memory
cd ~/ivy_memory/demos
python3 -m pip install -r requirements.txt
```

The interactive demos optionally make use of the simulator
[CoppeliaSim](https://www.coppeliarobotics.com/),
and the python wrapper [PyRep](https://github.com/stepjam/PyRep).

To get the full benefit of these demos, CoppeliaSim
and PyRep should both be installed, following the installation [intructions](https://github.com/stepjam/PyRep#install).

If these are not installed, the demos will all still run, but will display pre-rendered images from the simultator.

## Demos

All demos can be run by executing the python scripts directly.
If a demo script is run without command line arguments, then a random backend framework will be selected from those installed.
Alternatively, the `--framework` argument can be used to manually specify a framework from the options
`jax`, `tensorflow`, `torch`, `mxnd` or `numpy`.

### Run Through

For a basic run through the library:

```bash
cd ~/ivy_memory/demos
python3 run_through.py
```

This script, and the various parts of the library, are further discussed in the [Run Through](https://github.com/ivy-dl/memory#run-through) section of the main README.
We advise following along with this section for maximum effect. The demo script should also be opened locally,
and breakpoints added to step in at intermediate points to further explore.

To run the script using a specific backend, tensorflow for example, then run like so:

```bash
python3 run_through.py --framework tensorflow
```

### Learning to Copy with NTM

In this demo, a Neural Turing Machine (NTM) is trained to copy n-bit sequences
from one memory bank to a new memory bank.
Predictions are visualized throughout training.
To overfit to a single sequence, simply use the command line `--overfit` flag.

```bash
cd ~/ivy_memory/demos/interactive
python3 learning_to_copy_with_ntm.py
```

An example of overfitting on copying a single sequence,
with real-time training speed, is given below:

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_memory/demo_a.gif?raw=true'>
</p>

### Mapping a Room with ESM

In this demo, the user is able to interactively drag a drone with a single monocular camera through a scene.
The ESM memory module is used to incrementally map the geometry.

```bash
cd ~/ivy_memory/demos/interactive
python3 mapping_a_room_with_esm.py
```

An example of rotating the drone without translation to create an egocentric ESM map of a room is given below.
The raw image observations are shown on the left,
and the incrementally constructed omni-directional ESM feature and depth images are shown on the right.
While this example only projects color values into the memory, arbitrary neural network features can also be projected, for end-to-end training.

<p align="center">
    <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_memory/demo_b.gif?raw=true'>
</p>

## Get Involved

If you have any issues running any of the demos, would like to request further demos, or would like to implement your own, then get it touch.
Feature requests, pull requests, and [tweets](https://twitter.com/ivythread) all **welcome**!