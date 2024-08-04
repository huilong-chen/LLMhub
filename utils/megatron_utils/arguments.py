def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""

    # common

    group = parser.add_argument_group(title="custom common args")

    group.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Whitespace separated paths or corpora names for training.",
    )

    group.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Whitespace separated paths or corpora names for validating.",
    )

    group.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer",
    )

    group.add_argument(
        "--rope-base",
        type=float,
        default=10000,
    )

    group.add_argument(
        "--add-max-z-loss",
        action="store_true",
        help="additional max z loss",
    )

    group.add_argument(
        "--norm-lm-head",
        action="store_true",
        help="use norm lm head",
    )

    group.add_argument(
        "--do-train",
        type=int,
        default=1,
        help="Whether we need to train the model or not. ",
    )
    group.add_argument(
        "--do-valid",
        type=int,
        default=1,
        help="Whether we need to validate the model or not. ",
    )
    group.add_argument(
        "--do-test",
        type=int,
        default=0,
        help="Whether we need to test the model or not. ",
    )

    group.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="Intermediate size of the moe layer",
    )

    group.add_argument(
        "--shared-expert-intermediate-size",
        type=int,
        default=None,
        help="Intermediate size of the shared expert layer",
    )

    # finetune

    group = parser.add_argument_group(title="custom finetune args")

    group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of finetunning epochs. Zero results in evaluation only.",
    )

    group.add_argument(
        "--save-epoch-interval",
        type=int,
        default=None,
        help="Number of epoch between checkpoint saves.",
    )

    return parser
