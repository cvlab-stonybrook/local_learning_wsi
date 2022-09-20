def add_common_arguments(parser):
    parser.add_argument("dataset_root", type=str, help='The path of hdf5 file.')
    parser.add_argument("dataset_csv", type=str, help='The csv of dataset.')

    parser.add_argument("--num_classes", type=int, default=2, help="")
    parser.add_argument('--data_mean', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='mean of dataset')
    parser.add_argument('--data_std', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='mean of dataset')

    parser.add_argument("--batch-size", type=int, default=1, help="Choose the batch size for AdamW")
    # simulate batch size
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='simulate larger batch size by '
                                                                               'accumulating gradients')
    parser.add_argument("--epochs", type=int, default=80, help="How many epochs to train for")

    # Learning rate schedule parameters
    parser.add_argument("--precision", type=int, default=32, help="32 or 16 bit precision training")

    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-factor', type=float, default=1.,
                        help='learning rate multiplication for pretrained networks (default: 1.)')
    parser.add_argument('--loss-weight', type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help='weight of each class')
    parser.add_argument("--alpha", type=float, default=1., help="")

    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    parser.add_argument('--decay-epochs', type=float, default=60, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--decay-multi-epochs', type=lambda s: [int(item) for item in s.split(',')], default=None,
                        help='epochs to decay LR')

    parser.add_argument("--K", type=int, default=4, help="")
    parser.add_argument('--delay-epochs', type=lambda s: [int(item) for item in s.split(',')], default=[0],
                        help='epochs to decay LR')

    parser.add_argument("--output-dir", type=str, help="An output directory")
    parser.add_argument('--run-name', type=str, default='test')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to use for data loading')

    parser.add_argument('--weight-decay', type=float, default=None,
                        help='weight decay')


    # my personal rest parameters
    parser.add_argument('--load_weights', type=str, default=None, help='If not None, load weights from given path')

    parser.add_argument('--gpu_id', type=lambda s: [int(item) for item in s.split(',')], default=None)

    parser.add_argument('--tag', type=str, default='', help="For logging only")
    parser.add_argument('--debug', action="store_true", help='Whether show progressive the bar')
    return parser


def get_arguments(parser):
    parser = add_common_arguments(parser)
    opts = parser.parse_args()
    opts = process_common_arguments(opts)
    return opts


def process_common_arguments(opts):
    # opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    return opts