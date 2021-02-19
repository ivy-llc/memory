.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='docs/partial_source/logos/logo.png'>
    </p>

.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-memory">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-memory.svg">
    </a>
    <a href="https://www.apache.org/licenses/LICENSE-2.0">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/pypi/l/ivy-memory">
    </a>
    <a href="https://github.com/ivy-dl/memory/actions?query=workflow%3Adocs">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/memory/docs?label=docs">
    </a>
    <a href="https://github.com/ivy-dl/memory/actions?query=workflow%3Anightly-tests">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/github/workflow/status/ivy-dl/memory/nightly-tests?label=nightly">
    </a>
    <a href="https://discord.gg/EN9YS3QW8w">
        <img style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

**End-to-end memory modules for deep learning developers, written in Ivy.**

.. raw:: html

    <div style="display: block;">
        <img width="4%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://mxnet.apache.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/mxnet_logo.png">
        </a>
        <img width="6.66%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img width="12%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
    </div>

Contents
--------

* `Overview`_
* `Run Through`_
* `Interactive Demos`_
* `Get Involed`_

Overview
--------

.. _docs: https://ivy-dl.org/memory

**What is Ivy Memory?**

Ivy memory provides a range of differentiable memory modules,
including learnt modules such as Long Short Term Memory (LSTM) and Neural Turing Machines (NTM),
but also parameter-free modules, such as End-to-End Egospheric Spatial Memory (ESM).
Check out the docs_ for more info!

The library is built on top of the Ivy deep learning framework.
This means all memory modules simultaneously support:
Jax, Tensorflow, PyTorch, MXNet, and Numpy.

**A Family of Libraries**

Ivy memory is one library in a family of Ivy libraries.
There are also Ivy libraries for mechanics, 3D vision, robotics, and differentiable gym environments.
Click on the icons below for their respective github pages.

.. raw:: html

    <div style="display: block;">
        <img width="8%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/mech">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_mech.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/vision">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_vision.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/robot">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_robot.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/memory">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_memory.png">
        </a>
        <img width="2%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/empty.png">
        <a href="https://github.com/ivy-dl/gym">
            <img width="15%" style="float: left;" src="https://raw.githubusercontent.com/ivy-dl/ivy-dl.github.io/master/img/externally_linked/logos/ivy_gym.png">
        </a>
    </div>
    <br clear="all" />

**Quick Start**

Ivy memory can be installed like so: ``pip install ivy-memory``

.. _demos: https://github.com/ivy-dl/memory/tree/master/demos
.. _interactive: https://github.com/ivy-dl/memory/tree/master/demos/interactive

To quickly see the different aspects of the library, we suggest you check out the demos_!
We suggest you start by running the script ``run_through.py``,
and read the "Run Through" section below which explains this script.

For more interactive demos, we suggest you run either
``learning_to_copy_with_ntm.py`` or ``mapping_a_room_with_esm.py`` in the interactive_ demos folder.

Run Through
-----------

We run through some of the different parts of the library via a simple ongoing example script.
The full script is available in the demos_ folder, as file ``run_through.py``.
First, we select a random backend framework to use for the examples, from the options
``ivy.jax``, ``ivy.tensorflow``, ``ivy.torch``, ``ivy.mxnd`` or ``ivy.numpy``,
and use this to set the ivy backend framework.

.. code-block:: python

    import ivy
    import ivy_memory as ivy_mem
    from ivy_demo_utils.framework_utils import choose_random_framework
    ivy.set_framework(choose_random_framework())

**End-to-End Egospheric Spatial Memory**

First, we show how the Ivy End-to-End Egospheric Spatial Memory (ESM) class can be used inside a pure-Ivy model.
We first define the model as below.

.. code-block:: python

    class IvyModel:

        def __init__(self, channels_in, channels_out):
            self._channels_in = channels_in
            self._esm = ivy_mem.ESM(omni_image_dims=(16, 32))
            self._linear = ivy_mem.Linear(channels_in, channels_out)
            self.v = self._linear.v

        def forward(self, obs, v=None):
            if v is None:
                v = self.v
            mem = self._esm.forward(obs)
            x = ivy.reshape(mem.mean, (-1, self._channels_in))
            return self._linear.forward(x, v)

Next, we instantiate this model, and verify that the returned tensors are of the expected shape.

.. code-block:: python

    # create model
    in_channels = 32
    out_channels = 8
    ivy.set_framework('torch')
    model = IvyModel(in_channels, out_channels)

    # input config
    batch_size = 1
    image_dims = [5, 5]
    num_frames = 2
    num_feature_channels = 3

    # create image of pixel co-ordinates
    uniform_pixel_coords =\
        ivy_vision.create_uniform_pixel_coords_image(image_dims, [batch_size, num_frames])

    # define image measurement
    img_mean = ivy.concatenate((uniform_pixel_coords[..., 0:2], ivy.random_uniform(
        shape=[batch_size, num_frames] + image_dims + [1 + num_feature_channels])), -1)
    img_var = ivy.random_uniform(shape=[batch_size, num_frames] + image_dims + [1 + num_feature_channels])
    validity_mask = ivy.ones([batch_size, num_frames] + image_dims + [1])
    pose_mean = ivy.random_uniform(shape=[batch_size, num_frames, 6])
    pose_cov = ivy.random_uniform(shape=[batch_size, num_frames, 6, 6])
    cam_rel_mat = ivy.identity(4, batch_shape=[batch_size, num_frames])[..., 0:3, :]

    # place these into an ESM image measurement container
    esm_img_meas = ESMImageMeasurement(
        img_mean=img_mean,
        img_var=img_var,
        validity_mask=validity_mask,
        pose_mean=pose_mean,
        pose_cov=pose_cov,
        cam_rel_mat=cam_rel_mat
    )

    # define agent pose transformation
    control_mean = ivy.random_uniform(shape=[batch_size, num_frames, 6])
    control_cov = ivy.random_uniform(shape=[batch_size, num_frames, 6, 6])
    agent_rel_mat = ivy.identity(4, batch_shape=[batch_size, num_frames])[..., 0:3, :]

    # collect together into an ESM observation container
    esm_obs = ESMObservation(
        img_meas={'camera_0': esm_img_meas},
        control_mean=control_mean,
        control_cov=control_cov,
        agent_rel_mat=agent_rel_mat
    )

    # call model and test output
    output = model.forward(esm_obs)
    assert output.shape[-1] == out_channels

Finally, we define a dummy loss function, and show how the NTM can be trained using Ivy functions only.

.. code-block:: python

    # define loss function
    target = ivy.zeros_like(output)

    def loss_fn(var):
        pred = model.forward(esm_obs, var)
        return ivy.reduce_mean((pred - target) ** 2)

    # train model
    print('\ntraining dummy Ivy ESM model...\n')
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = ivy.gradient_descent_update(model.v, grads, 1e-4)
        print('step {}, loss = {}'.format(i, ivy.to_numpy(loss).item()))
    print('\ndummy Ivy ESM model trained!\n')
    ivy.unset_framework()

**Neural Turing Machine**

We next show how the Ivy Neural Turing Machine (NTM) class can be used inside a TensorFlow model.
First, we define the model as below.

.. code-block:: python

    class TfModel(tf.keras.Model):

        def __init__(self, channels_in, channels_out):
            tf.keras.Model.__init__(self)
            self._linear = tf.keras.layers.Dense(64)
            memory_size = 4
            memory_vector_dim = 1
            self._ntm = ivy_mem.NTM(
                input_dim=64, output_dim=channels_out, ctrl_output_size=channels_out, ctrl_layers=1,
                memory_size=memory_size, memory_vector_dim=memory_vector_dim, read_head_num=1, write_head_num=1)
            self._assign_variables()

        def _assign_variables(self):
            self._ntm.v.map(
                lambda x, kc: self.add_weight(name=kc, shape=x.shape))
            self.set_weights([ivy.to_numpy(v) for k, v in self._ntm.v.to_iterator()])
            self.trainable_weights_dict = dict()
            for weight in self.trainable_weights:
                self.trainable_weights_dict[weight.name] = weight
            self._ntm.v = self._ntm.v.map(lambda x, kc: self.trainable_weights_dict[kc + ':0'])

        def call(self, x, **kwargs):
            x = self._linear(x)
            return self._ntm.forward(x)

Next, we instantiate this model, and verify that the returned tensors are of the expected shape.

.. code-block:: python

    # create model
    in_channels = 32
    out_channels = 8
    ivy.set_framework('tensorflow')
    model = TfModel(in_channels, out_channels)

    # define inputs
    batch_shape = [1, 2]
    timesteps = 3
    input_shape = batch_shape + [timesteps, in_channels]
    input_seq = tf.random.uniform(batch_shape + [timesteps, in_channels])

    # call model and test output
    output_seq = model(input_seq)
    assert input_seq.shape[:-1] == output_seq.shape[:-1]
    assert input_seq.shape[-1] == in_channels
    assert output_seq.shape[-1] == out_channels

Finally, we define a dummy loss function, and show how the NTM can be trained using a native TensorFlow optimizer.

.. code-block:: python

    # define loss function
    target = tf.zeros_like(output_seq)

    def loss_fn():
        pred = model(input_seq)
        return tf.reduce_sum((pred - target) ** 2)

    # define optimizer
    optimizer = tf.keras.optimizers.Adam(1e-2)

    # train model
    print('\ntraining dummy TensorFlow NTM model...\n')
    for i in range(10):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print('step {}, loss = {}'.format(i, loss))
    print('\ndummy TensorFlow NTM model trained!\n')
    ivy.unset_framework()

Interactive Demos
-----------------

In addition to the run through above, we provide two further demo scripts,
which are more visual and interactive.

The scripts for these demos can be found in the interactive_ demos folder.

**Learning to Copy with NTM**

The first demo trains a Neural Turing Machine to copy a sequence from one memory bank to another.
NTM can overfit to a single copy sequence very quickly, as show in the real-time visualization below.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_memory/demo_a.gif?raw=true'>
    </p>

**Mapping a Room with ESM**

The second demo creates an egocentric map of a room, from a rotating camera.
The raw image observations are shown on the left,
and the incrementally constructed omni-directional ESM feature and depth images are shown on the right.
While this example only projects color values into the memory, arbitrary neural network features can also be projected, for end-to-end training.

.. raw:: html

    <p align="center">
        <img width="75%" style="display: block;" src='https://github.com/ivy-dl/ivy-dl.github.io/blob/master/img/externally_linked/ivy_memory/demo_b.gif?raw=true'>
    </p>

Get Involed
-----------

We hope the memory classes in this library are useful to a wide range of deep learning developers.
However, there are many more areas of differentiable memory which could be covered by this library.

If there are any particular functions or classes you feel are missing,
and your needs are not met by the functions currently on offer,
then we are very happy to accept pull requests!

We look forward to working with the community on expanding and improving the Ivy memory library.

Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated Deep Learning for Inter-Framework Portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
