"""
Collection of tests for ivy gym demos
"""

# global
import pytest
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
from ivy_tests.test_ivy import helpers

FWS = [ivy.functional.backends.jax, ivy.functional.backends.tensorflow, ivy.functional.backends.torch]


@pytest.mark.parametrize(
    "compile_flag", [True, False])
def test_demo_ntm_copy(compile_flag, dev_str, f, fw):
    from ivy_memory_demos.interactive.learning_to_copy_with_ntm import main
    if fw in ['numpy', 'tensorflow_graph']:
        # numpy does not support gradients, and demo compiles already, so no need to use tf_graph_call
        pytest.skip()
    if fw in ['torch', 'jax'] and compile_flag:
        # PyTorch Dictionary inputs to traced functions must have consistent type
        # JAX does not support dynamically sized arrays within JIT compiled functions
        pytest.skip()
    main(1, 1, compile_flag,
         1, 1, 1, 4, 1,
         interactive=False,
         f=f, fw=fw)


@pytest.mark.parametrize(
    "with_sim", [False])
@pytest.mark.parametrize(
    "compile_flag", [True, False])
def test_demo_esm(with_sim, compile_flag, dev_str, f, fw):
    from ivy_memory_demos.interactive.mapping_a_room_with_esm import main
    if fw in ['numpy', 'jax', 'mxnet']:
        # convolutions not yet implemented in numpy or jax
        # mxnet is unable to stack or expand zero-dimensional tensors
        pytest.skip()
    if fw in ['tensorflow_graph']:
        # test call function not used, so no need to use tf_graph_call
        pytest.skip()
    main(False, with_sim, f=f, fw=fw)


def test_demo_run_through(dev_str, f, fw):
    from ivy_memory_demos.run_through import main
    if fw in ['tensorflow_graph']:
        # these particular demos are only implemented in eager mode, without compilation
        pytest.skip()
    main()
