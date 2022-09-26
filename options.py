def add_common_arguments(parser):
    parser.add_argument("dataset_root", type=str, help='The path of hdf5 file.')
    parser.add_argument("dataset_csv", type=str, help='The csv of dataset.')
    parser.add_argument('--data-mean', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='mean of dataset')
    parser.add_argument('--data-std', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='mean of dataset')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to use for data loading')
    parser.add_argument("--num-classes", type=int, default=2, help="")


    parser.add_argument("--epochs", type=int, default=80, help="How many epochs to train for")
    parser.add_argument("--batch-size", type=int, default=1, help="Choose the batch size for AdamW")
    parser.add_argument('--accumulate-grad-batches', type=int, default=8, help='simulate larger batch sizes by '
                                                                               'accumulating gradients')

    parser.add_argument("--K", type=int, default=4, help="The number of local modules")
    parser.add_argument('--load-weights', type=str, default=None, help='If not None, load weights from given path')
    parser.add_argument("--precision", type=int, default=32, help="32 or 16 bit precision training")

    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (default: 2e-5)')
    parser.add_argument('--lr-factor', type=float, default=0.5,
                        help='learning rate multiplication factor for pretrained networks (default: 0.5)')
    parser.add_argument('--loss-weight', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='weight of each classes')  # eg. "0.5,1"
    parser.add_argument("--alpha", type=float, default=1., help="")
    parser.add_argument('--decay-multi-epochs', type=lambda s: [int(item) for item in s.split(',')], default=[40, 60],
                        help='epochs to decay LR by 0.1')
    parser.add_argument('--weight-decay', type=float, default=1e-2, help='Weight decay. Default is 1e-2')

    parser.add_argument("--output-dir", type=str, required=True, help="The output directory")
    parser.add_argument('--project-name', type=str, default='test', help="The project name")
    parser.add_argument('--gpu-id', type=lambda s: [int(item) for item in s.split(',')], default=[0])  # assign a gpu

    parser.add_argument('--run-name', type=str, default='test_run', help="The run name")
    parser.add_argument('--progressive', action="store_true", help='Whether show progressive the bar')
    return parser


def get_arguments(parser):
    parser = add_common_arguments(parser)
    opts = parser.parse_args()
    opts = process_common_arguments(opts)
    return opts


def process_common_arguments(opts):
    return opts