"""
Collection of tests for ivy gym demos
"""

# global
import pytest
import ivy.jax
import ivy.mxnet
import ivy.tensorflow
import ivy.torch
import ivy_tests.helpers as helpers

FWS = [ivy.jax, ivy.mxnet, ivy.tensorflow, ivy.torch]


@pytest.mark.parametrize(
    "compile_flag", [True, False])
def test_demo_ntm_copy(compile_flag, dev_str, f, call):
    from ivy_memory_demos.interactive.learning_to_copy_with_ntm import main
    if call in [helpers.np_call, helpers.tf_graph_call]:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    if call in [helpers.torch_call, helpers.jnp_call] and compile_flag:
        # PyTorch Dictionary inputs to traced functions must have consistent type
        # JAX does not support dynamically sized arrays within JIT compiled functions
        pytest.skip()
    main(1, 1, compile_flag,
         1, 1, 1, 4, 1,
         interactive=False,
         f=f)


@pytest.mark.parametrize(
    "with_sim", [False])
@pytest.mark.parametrize(
    "compile_flag", [True, False])
def test_demo_esm(with_sim, compile_flag, dev_str, f, call):
    from ivy_memory_demos.interactive.mapping_a_room_with_esm import main
    if call in [helpers.np_call, helpers.jnp_call, helpers.mx_call]:
        # convolutions not yet implemented in numpy or jax
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    if call in [helpers.tf_graph_call]:
        # test call function not used, so no need to use tf_graph_call
        pytest.skip()
    main(False, with_sim, f=f)


def test_demo_run_through(dev_str, f, call):
    from ivy_memory_demos.run_through import main
    if call in [helpers.tf_graph_call]:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main()
