# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Finetune utilities."""

import sys
from functools import partial

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_model_config
from megatron.training import get_args, get_timers, print_rank_0, get_tokenizer, get_num_microbatches
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.training import evaluate_and_print_results
from megatron.training.training import setup_model_and_optimizer
from megatron.training.training import train_step
from megatron.training.training import training_log
from megatron.training.utils import (
    check_adlr_autoresume_termination,
    average_losses_across_data_parallel_group,
    get_ltor_masks_and_position_ids,
    calc_params_l2_norm,
)


def build_data_loader(dataset, micro_batch_size, num_workers, task_collate_fn=None):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""

    args = get_args()
    world_size = parallel_state.get_data_parallel_world_size()
    rank = parallel_state.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, seed=args.seed
    )
    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        collate_fn=task_collate_fn,
        persistent_workers=getattr(args, "multi_model", False),
    )
    return data_loader


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, task_collate_fn=None):
    """Traing and validation dataloaders."""
    args = get_args()

    print_rank_0("building train and validation dataloaders ...")
    # Training dataset.
    train_dataloader = build_data_loader(
        train_dataset,
        args.micro_batch_size,
        args.num_workers,
        task_collate_fn,
    )

    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataset) // args.global_batch_size
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    if valid_dataset is not None:
        valid_dataloader_ = build_data_loader(
            valid_dataset,
            args.micro_batch_size,
            args.num_workers,
            task_collate_fn,
        )

        valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)
    else:
        valid_dataloader = None

    return train_dataloader, valid_dataloader


def _train(
    model,
    optimizer,
    opt_param_scheduler,
    forward_step,
    train_dataloader,
    valid_dataloader,
    end_of_epoch_callback,
):
    """Train the model."""
    args = get_args()
    timers = get_timers()
    config = get_model_config(model[0])

    # Turn on training mode which enables dropout.
    for m in model:
        m.train()

    # Tracking loss.
    losses_dict_sum = {}

    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    iteration = args.iteration
    args.consumed_train_samples = iteration * args.global_batch_size

    # Memory reporting flag.
    report_memory_flag = True

    # For each remaining epoch
    timers("interval-time", log_level=0).start(barrier=True)
    for epoch in range(start_epoch, args.epochs):
        print_rank_0("working on epoch {} ...".format(epoch + 1))

        # Set the data loader epoch to shuffle the index iterator.
        train_dataloader.sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)

        iteration_in_epoch = 0
        # For all the batches in the dataset.
        for _ in range(start_iteration):
            next(train_iter)
            iteration_in_epoch += 1

        start_iteration = 0

        while iteration_in_epoch < args.train_iters_per_epoch:
            # Train for one step.
            out = train_step(forward_step, train_iter, model, optimizer, opt_param_scheduler, config)

            losses_dict, skipped_iter, grad_norm, num_zeros_in_grad = out
            iteration += 1
            iteration_in_epoch += 1
            args.consumed_train_samples += args.global_batch_size

            # Logging.
            params_norm = None
            if args.log_params_norm:
                params_norm = calc_params_l2_norm(model)
            learning_rate = None
            decoupled_learning_rate = None
            for param_group in optimizer.param_groups:
                if param_group["is_decoupled_lr"]:
                    decoupled_learning_rate = param_group["lr"]
                else:
                    learning_rate = param_group["lr"]
            report_memory_flag = training_log(
                losses_dict,
                losses_dict_sum,
                learning_rate,
                decoupled_learning_rate,
                iteration,
                optimizer.get_loss_scale().item(),
                report_memory_flag,
                skipped_iter,
                grad_norm,
                params_norm,
                num_zeros_in_grad,
            )

            # Autoresume
            if args.adlr_autoresume and (iteration % args.adlr_autoresume_interval == 0):
                check_adlr_autoresume_termination(iteration, model, optimizer, opt_param_scheduler)

            # Checkpointing
            saved_checkpoint = False
            if (
                args.save
                and args.save_interval
                and args.save_epoch_interval is None
                and iteration % args.save_interval == 0
            ):
                save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)
                saved_checkpoint = True

            # Evaluation
            if args.eval_interval and iteration % args.eval_interval == 0 and valid_dataloader is not None:
                prefix = "iteration {}".format(iteration)
                evaluate_and_print_results(
                    prefix, forward_step, valid_dataloader, model, iteration, None, config, False
                )

            # Exiting based on iterations
            if args.exit_interval and iteration % args.exit_interval == 0:
                if not saved_checkpoint:
                    save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)
                torch.distributed.barrier()
                print_rank_0("exiting program at iteration {}".format(iteration))
                sys.exit()

        # Checkpointing at the end of each epoch.
        if args.save and args.save_epoch_interval and (epoch + 1) % args.save_epoch_interval == 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None:
            end_of_epoch_callback(model, epoch)
    if args.save:
        if args.save_epoch_interval is None:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)
        elif args.epochs % args.save_epoch_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler, 0)


def calc_token_weight(input_ids, loss_mask):
    tokenizer = get_tokenizer()
    seqids = torch.cumsum(input_ids == tokenizer.eos_token_id, 0)
    seqids[1:] = seqids[:-1].clone()
    unique_seqids = torch.unique(seqids)
    token_weight = torch.zeros_like(unique_seqids)
    token_weight = token_weight.scatter_add(0, seqids.long(), loss_mask.clone().long())
    token_weight = torch.repeat_interleave(token_weight, seqids.bincount())
    token_weight[token_weight == 0] = 1
    token_weight = token_weight.to(loss_mask.device)
    return token_weight


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["input_ids", "labels"]
    if args.reset_attention_mask:
        keys.append("cu_seqlens")
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    images = None

    # Unpack.
    tokens = data_b["input_ids"].long()
    labels = data_b["labels"].long()
    cu_seqlens = data_b["cu_seqlens"].int() if args.reset_attention_mask else None

    shift_labels = labels[..., 1:].contiguous()
    shift_labels = torch.cat([shift_labels, torch.full_like(shift_labels[:, -1:], -100)], dim=1)
    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eos_token_id,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    if hasattr(args, "multi_model") and args.multi_model:
        data_c = tensor_parallel.broadcast_data(["image"], data, torch.bfloat16)
        data_b["image"] = data_c["image"]
        images = data_b["image"]
        loss_mask = torch.ones(shift_labels.size(), dtype=torch.float, device=shift_labels.device)

    loss_mask[shift_labels == -100] = 0

    return (
        tokens,
        shift_labels,
        loss_mask,
        attention_mask,
        position_ids,
        cu_seqlens,
        images
    )


def loss_func(output_tensor, loss_mask, input_ids):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    input_ids = input_ids.view(-1)
    assert loss_mask.sum() > 0
    token_weight = calc_token_weight(input_ids, loss_mask)
    args = get_args()
    loss = torch.sum(losses.view(-1) * loss_mask / token_weight.float()) * getattr(args, "loss_weight", 1.0)
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {"lm loss": averaged_loss[0]}


def calc_loss_weight(train_dataset):
    """
    loss_weight = num_micro_batches / avg_block_num
    其中 avg_block_num 为平均每个 global batch 的块数。这里"块"表示以 eos 分割的一段文本。
    avg_block_num = all_block_num / train_iters_per_epoch
    """
    tokenizer = get_tokenizer()
    args = get_args()

    if torch.distributed.get_rank() == 0:
        all_block_num = (
            train_dataset.with_format("torch")
            .map(
                lambda line: {"count": (line["input_ids"] == tokenizer.eos_token_id).sum()},
                num_proc=30,
                desc="count blocks",
            )["count"]
            .sum()
        )
        all_block_num = torch.tensor([all_block_num], dtype=torch.long, device="cuda")
    else:
        all_block_num = torch.tensor([0], dtype=torch.long, device="cuda")
    torch.distributed.broadcast(all_block_num, 0)
    print_rank_0(f"avg block num: {all_block_num.item()}")
    num_micro_batches = get_num_microbatches()
    avg_block_num = all_block_num / args.train_iters_per_epoch
    loss_weight = num_micro_batches / avg_block_num
    print_rank_0(f"avg block num: {avg_block_num.item()}")
    print_rank_0(f"loss weight: {loss_weight.item()}")
    return loss_weight


def forward_step(data_iterator, model):
    """Forward step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    timers = get_timers()
    args = get_args()

    # Get the batch.
    timers("batch-generator").start()
    (
        tokens,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        cu_seqlens,
        images,
    ) = get_batch(data_iterator)

    if cu_seqlens is None:
        packed_seq_params = None
    else:
        packed_seq_params = PackedSeqParams(cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens)
    timers("batch-generator").stop()

    if hasattr(args, "multi_model") and args.multi_model:
        output_tensor = model(images, tokens, position_ids, attention_mask, labels=labels)
    else:
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels, packed_seq_params=packed_seq_params)
    return output_tensor, partial(loss_func, loss_mask=loss_mask, input_ids=tokens)


LABEL_PAD_TOKEN_ID = -100


def pad_to_max_length(features, pad_token_id):
    args = get_args()
    max_length = args.seq_length
    batch = []
    for feature in features:
        assert isinstance(feature, list)
        remainder = [pad_token_id] * (max_length - len(feature))
        batch.append(torch.tensor(feature + remainder))
    batch = torch.stack(batch)
    return batch


def task_collate_fn(features):
    tokenizer = get_tokenizer()
    args = get_args()
    # transpose list of dict to dict of list
    features = {k: [d[k] for d in features] for k in features[0]}
    batch = {}
    if "input_ids" in features:
        batch["input_ids"] = pad_to_max_length(features["input_ids"], tokenizer.pad_token_id)

    if "labels" in features:
        batch["labels"] = pad_to_max_length(features["labels"], LABEL_PAD_TOKEN_ID)

    if args.reset_attention_mask:
        b, s = batch["input_ids"].size()
        tokens_flatten = batch["input_ids"].reshape(-1)
        eos_mask = tokens_flatten == tokenizer.eos_token_id
        eos_mask[s - 1 :: s] = True
        eos_pos = eos_mask.nonzero(as_tuple=False).reshape(-1).long()
        zero = torch.tensor([0], device=eos_pos.device, dtype=eos_pos.dtype)
        cu_seqlens = torch.cat([zero, eos_pos + 1], dim=0)
        batch["cu_seqlens"] = cu_seqlens

    return batch


def finetune(
    train_valid_datasets_provider,
    model_provider,
    model_type=ModelType.encoder_or_decoder,
    forward_step=forward_step,
    end_of_epoch_callback_provider=None,
    task_collate_fn=task_collate_fn,
):
    """Main finetune function used across all tasks."""
    args = get_args()
    timers = get_timers()

    assert args.rampup_batch_size is None, "batch size scaling is not supported for finetuning"

    # Train and validation data loaders.
    timers("train/valid/test dataset/dataloder", log_level=0).start()
    if args.epochs > 0:
        train_dataset, valid_dataset = train_valid_datasets_provider()
        train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
            train_dataset, valid_dataset, task_collate_fn
        )
        args.loss_weight = calc_loss_weight(train_dataset)
    else:
        args.train_iters = 0
    timers("train/valid/test dataset/dataloder").stop()

    # Build calback function.
    timers("callback function", log_level=0).start()
    end_of_epoch_callback = None
    if end_of_epoch_callback_provider is not None:
        end_of_epoch_callback = end_of_epoch_callback_provider()
    timers("callback function").stop()

    # Build model, optimizer and learning rate scheduler.
    timers("model and optimizer", log_level=0).start()
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(model_provider, model_type)
    timers("model and optimizer").stop()

    # Print setup timing.
    print_rank_0("done with setups ...")
    timers.log(
        ["train/valid/test dataset/dataloder", "callback function", "model and optimizer", "pretrained checkpoint"],
        barrier=True,
    )
    print_rank_0("training ...")
    write_args_to_tensorboard()

    # Finetune the model.
    if args.epochs > 0:
        _train(
            model,
            optimizer,
            opt_param_scheduler,
            forward_step,
            train_dataloader,
            valid_dataloader,
            end_of_epoch_callback,
        )
    # Or just evaluate.
    else:
        if end_of_epoch_callback is not None:
            print_rank_0("evaluation only mode, setting epoch to -1")
            end_of_epoch_callback(model, epoch=-1, output_predictions=True)
    print_rank_0("done :-)")
