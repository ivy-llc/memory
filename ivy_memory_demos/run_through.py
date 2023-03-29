# global
import ivy
import torch
import ivy_vision
import tensorflow as tf
from ivy_memory.geometric.containers import ESMCamMeasurement, ESMObservation

# local
import ivy_memory as ivy_mem


def main():

    # LSTM #
    # -----#

    # using the Ivy LSTM memory module, dual stacked, in a PyTorch model

    class TorchModelWithLSTM(torch.nn.Module):

        def __init__(self, channels_in, channels_out):
            torch.nn.Module.__init__(self)
            self._linear = torch.nn.Linear(channels_in, 64)
            self._lstm = ivy_mem.LSTM(64, channels_out, num_layers=2, return_state=False)
            self._assign_variables()

        def _assign_variables(self):
            self._lstm.v.map(
                lambda x, kc: self.register_parameter(name=kc, param=torch.nn.Parameter(x.data)))
            self._lstm.v = self._lstm.v.map(lambda x, kc: self._parameters[kc])

        def forward(self, x):
            x = self._linear(x)
            return self._lstm(x)

    # create model
    in_channels = 32
    out_channels = 8
    ivy.set_backend('torch')
    model = TorchModelWithLSTM(in_channels, out_channels)

    # define inputs
    batch_shape = [1, 2]
    timesteps = 3
    input_shape = batch_shape + [timesteps, in_channels]
    input_seq = torch.rand(batch_shape + [timesteps, in_channels])

    # call model and test output
    output_seq = model(input_seq)
    assert input_seq.shape[:-1] == output_seq.shape[:-1]
    assert input_seq.shape[-1] == in_channels
    assert output_seq.shape[-1] == out_channels

    # define loss function
    target = ivy.zeros_like(output_seq)

    def loss_fn():
        pred = model(input_seq)
        return ivy.sum((pred - target) ** 2)

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # train model
    print('\ntraining dummy PyTorch LSTM model...\n')
    for i in range(10):
        loss = loss_fn()
        loss.backward()
        optimizer.step()
        print('step {}, loss = {}'.format(i, loss))
    print('\ndummy PyTorch LSTM model trained!\n')
    ivy.previous_backend()

    # NTM #
    # ----#

    # using the Ivy NTM memory module in a TensorFlow model

    class TfModelWithNTM(tf.keras.Model):

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
            return self._ntm(x)

    # create model
    in_channels = 32
    out_channels = 8
    ivy.set_backend('tensorflow')
    model = TfModelWithNTM(in_channels, out_channels)

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

    # define loss function
    target = ivy.zeros_like(output_seq)

    def loss_fn():
        pred = model(input_seq)
        return ivy.sum((pred - target) ** 2)

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
    ivy.previous_backend()

    # ESM #
    # ----#

    # using the Ivy ESM memory module in a pure-Ivy model, with a JAX backend
    # ToDo: add pre-ESM conv layers to this demo

    class IvyModelWithESM(ivy.Module):

        def __init__(self, channels_in, channels_out):
            self._channels_in = channels_in
            self._esm = ivy_mem.ESM(omni_image_dims=(16, 32))
            self._linear = ivy_mem.Linear(channels_in, channels_out)
            ivy.Module.__init__(self, 'cpu')

        def _forward(self, obs):
            mem = self._esm(obs)
            x = ivy.reshape(mem['mean'], (-1, self._channels_in))
            return self._linear(x)

    # create model
    in_channels = 32
    out_channels = 8
    ivy.set_backend('torch')
    model = IvyModelWithESM(in_channels, out_channels)

    # input config
    batch_size = 1
    image_dims = [5, 5]
    num_timesteps = 2
    num_feature_channels = 3

    # create image of pixel co-ordinates
    uniform_pixel_coords =\
        ivy_vision.create_uniform_pixel_coords_image(image_dims, [batch_size, num_timesteps])

    # define camera measurement
    depths = ivy.random_uniform(shape=[batch_size, num_timesteps] + image_dims + [1])
    ds_pixel_coords = ivy_vision.depth_to_ds_pixel_coords(depths)
    inv_calib_mats = ivy.random_uniform(shape=[batch_size, num_timesteps, 3, 3])
    cam_coords = ivy_vision.ds_pixel_to_cam_coords(ds_pixel_coords, inv_calib_mats)[..., 0:3]
    features = ivy.random_uniform(shape=[batch_size, num_timesteps] + image_dims + [num_feature_channels])
    img_mean = ivy.concat((cam_coords, features), axis=-1)
    cam_rel_mat = ivy.eye(4, batch_shape=[batch_size, num_timesteps])[..., 0:3, :]

    # place these into an ESM camera measurement container
    esm_cam_meas = ESMCamMeasurement(
        img_mean=img_mean,
        cam_rel_mat=cam_rel_mat
    )

    # define agent pose transformation
    agent_rel_mat = ivy.eye(4, batch_shape=[batch_size, num_timesteps])[..., 0:3, :]

    # collect together into an ESM observation container
    esm_obs = ESMObservation(
        img_meas={'camera_0': esm_cam_meas},
        agent_rel_mat=agent_rel_mat
    )

    # call model and test output
    output = model(esm_obs)
    assert output.shape[-1] == out_channels

    # define loss function
    target = ivy.zeros_like(output)

    def loss_fn(v):
        pred = model(esm_obs, v=v)
        return ivy.mean((pred - target) ** 2)

    # optimizer
    optimizer = ivy.SGD(lr=1e-4)

    # train model
    print('\ntraining dummy Ivy ESM model...\n')
    for i in range(10):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('step {}, loss = {}'.format(i, ivy.to_numpy(loss).item()))
    print('\ndummy Ivy ESM model trained!\n')
    ivy.previous_backend()

    # message
    print('End of Run Through Demo!')


if __name__ == '__main__':
    main()
