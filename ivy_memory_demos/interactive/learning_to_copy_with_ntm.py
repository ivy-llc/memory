# global
import ivy

# import  ivy.compiler.compiler as ic
import cv2
import argparse
import numpy as np

# local
from ivy_memory.learnt import NTM


def loss_fn(ntm, v, total_seq, target_seq, seq_len):
    output_sequence = ntm(total_seq, v=v)
    pred_logits = output_sequence[:, seq_len + 1 :, :]
    pred_vals = ivy.sigmoid(pred_logits)
    return (
        ivy.sum(ivy.binary_cross_entropy(pred_vals, target_seq)) / pred_vals.shape[0],
        pred_vals,
    )


def train_step(
    loss_fn_in,
    optimizer,
    ntm,
    total_seq,
    target_seq,
    seq_len,
    mw,
    vw,
    step,
    max_grad_norm,
):
    # compute loss
    func_ret, grads = ivy.execute_with_gradients(
        lambda v_: loss_fn_in(v_, total_seq, target_seq, seq_len),
        ntm.v,
        ret_grad_idxs=["0"],
    )

    grads = grads["0"]
    global_norm = (
        ivy.sum(
            ivy.stack(
                [ivy.sum(grad**2) for grad in grads.cont_to_flat_list()], axis=0
            )
        )
        ** 0.5
    )
    grads = grads.cont_map(
        lambda x, _: x * max_grad_norm / ivy.maximum(global_norm, max_grad_norm)
    )

    # update variables
    ntm.v = optimizer.step(ntm.v, grads)
    return func_ret


def main(
    batch_size=32,
    num_train_steps=31250,
    compile_flag=True,
    num_bits=8,
    seq_len=28,
    ctrl_output_size=100,
    memory_size=128,
    memory_vector_dim=28,
    overfit_flag=False,
    interactive=True,
    f=None,
    fw=None,
):
    fw = ivy.choose_random_backend() if fw is None else fw
    ivy.set_backend(fw)
    f = ivy.with_backend(backend=fw) if f is None else f

    # train config
    lr = 1e-3 if not overfit_flag else 1e-2
    batch_size = batch_size if not overfit_flag else 1
    num_train_steps = num_train_steps if not overfit_flag else 150
    max_grad_norm = 50

    # logging config
    vis_freq = 250 if not overfit_flag else 1

    # optimizer
    optimizer = ivy.Adam(lr=lr)

    # ntm
    ntm = NTM(
        input_dim=num_bits + 1,
        output_dim=num_bits,
        ctrl_output_size=ctrl_output_size,
        ctrl_layers=1,
        memory_size=memory_size,
        memory_vector_dim=memory_vector_dim,
        read_head_num=1,
        write_head_num=1,
    )

    # compile loss fn
    if compile_flag:
        # loss_fn_maybe_compiled = ic.compile(
        #     lambda v, ttl_sq, trgt_sq, sq_ln: loss_fn(ntm, v, ttl_sq, trgt_sq, sq_ln),
        #     return_backend_compiled_fn=True,
        # )
        loss_fn_maybe_compiled = lambda v, ttl_sq, trgt_sq, sq_ln: loss_fn(
            ntm, v, ttl_sq, trgt_sq, sq_ln
        )
    else:
        loss_fn_maybe_compiled = lambda v, ttl_sq, trgt_sq, sq_ln: loss_fn(
            ntm, v, ttl_sq, trgt_sq, sq_ln
        )

    # init
    input_seq_m1 = ivy.astype(
        ivy.random_uniform(low=0.0, high=1.0, shape=(batch_size, seq_len, num_bits))
        > 0.5,
        "float32",
    )
    mw = None
    vw = None

    for i in range(num_train_steps):
        # sequence to copy
        if not overfit_flag:
            input_seq_m1 = ivy.astype(
                ivy.random_uniform(
                    low=0.0, high=1.0, shape=(batch_size, seq_len, num_bits)
                )
                > 0.5,
                "float32",
            )
        target_seq = input_seq_m1
        input_seq = ivy.concat(
            (input_seq_m1, ivy.zeros((batch_size, seq_len, 1))), axis=-1
        )
        eos = ivy.ones((batch_size, 1, num_bits + 1))
        output_seq = ivy.zeros_like(input_seq)
        total_seq = ivy.concat((input_seq, eos, output_seq), axis=-2)

        # train step
        loss, pred_vals = train_step(
            loss_fn_maybe_compiled,
            optimizer,
            ntm,
            total_seq,
            target_seq,
            seq_len,
            mw,
            vw,
            ivy.array(i + 1, dtype="float32"),
            max_grad_norm,
        )

        # log
        print("step: {}, loss: {}".format(i, ivy.to_numpy(loss).item()))

        # visualize
        if i % vis_freq == 0:
            target_to_vis = (ivy.to_numpy(target_seq[0] * 255)).astype(np.uint8)
            target_to_vis = np.transpose(
                cv2.resize(target_to_vis, (560, 160), interpolation=cv2.INTER_NEAREST),
                (1, 0),
            )

            pred_to_vis = (ivy.to_numpy(pred_vals[0] * 255)).astype(np.uint8)
            pred_to_vis = np.transpose(
                cv2.resize(pred_to_vis, (560, 160), interpolation=cv2.INTER_NEAREST),
                (1, 0),
            )

            img_to_vis = np.concatenate((pred_to_vis, target_to_vis), 0)
            img_to_vis = cv2.resize(
                img_to_vis, (1120, 640), interpolation=cv2.INTER_NEAREST
            )

            img_to_vis[0:60, -200:] = 0
            img_to_vis[5:55, -195:-5] = 255
            cv2.putText(
                img_to_vis,
                "step {}".format(i),
                (935, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                tuple([0] * 3),
                2,
            )

            img_to_vis[0:60, 0:200] = 0
            img_to_vis[5:55, 5:195] = 255
            cv2.putText(
                img_to_vis,
                "prediction",
                (7, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                tuple([0] * 3),
                2,
            )

            img_to_vis[320:380, 0:130] = 0
            img_to_vis[325:375, 5:125] = 255
            cv2.putText(
                img_to_vis,
                "target",
                (7, 362),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                tuple([0] * 3),
                2,
            )

            if interactive:
                cv2.imshow("prediction_and_target", img_to_vis)
                if overfit_flag:
                    cv2.waitKey(1)
                else:
                    cv2.waitKey(100)
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=31250,
        help="The number of steps to train for.",
    )
    parser.add_argument(
        "--eager", action="store_true", help="Whether to compile the training step."
    )

    parser.add_argument(
        "--num_bits", type=int, default=8, help="Number of bits in the NTM memory."
    )
    parser.add_argument(
        "--seq_len", type=int, default=28, help="Sequence length for the NTM memory."
    )
    parser.add_argument(
        "--ctrl_output_size",
        type=int,
        default=100,
        help="Output size from the NTM controller.",
    )
    parser.add_argument("--memory_size", type=int, default=128, help="NTM memory size.")
    parser.add_argument(
        "--memory_vector_dim", type=int, default=28, help="NTM memory vector dimension."
    )
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Whether to overfit the NTM training for a single copy sequence.",
    )

    parser.add_argument(
        "--non_interactive",
        action="store_true",
        help="whether to run the demo in non-interactive mode.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="which backend to use. Chooses a random backend if unspecified.",
    )

    parsed_args = parser.parse_args()
    fw = parsed_args.backend
    f = None if fw is None else ivy.get_backend(backend=fw)
    main(
        parsed_args.batch_size,
        parsed_args.num_training_steps,
        not parsed_args.eager,
        parsed_args.num_bits,
        parsed_args.seq_len,
        parsed_args.ctrl_output_size,
        parsed_args.memory_size,
        parsed_args.memory_vector_dim,
        parsed_args.overfit,
        not parsed_args.non_interactive,
        f,
        fw,
    )
