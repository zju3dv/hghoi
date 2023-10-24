import argparse
from motion.config import get_config
from motion.comm import build_server
from motion.utils.utils import get_model, get_socket_func
from motion.engine.builder import TRAINERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-type",
        type=str,
        required=True,
        help="server type",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="socket server port",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        required=True,
        help="hostname",
    )
    parser.add_argument(
        "--client-type",
        type=str,
        required=True,
        help="c or python",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--epoch",
        default=0,
        type=str,
        help="Choose test model with epoch",
    )
    parser.add_argument(
        "--external-config",
        type=str,
        default="",
        help="path to external config yaml containing info about experiment",
    )
    parser.add_argument(
        "--external-epoch",
        default=0,
        type=str,
        help="Choose external model with epoch",
    )

    args = parser.parse_args()

    model_cfg = get_config(args.config)
    if args.external_config != "":
        external_cfg = get_config(args.external_config)
        external_model = get_model(TRAINERS, external_cfg, args.external_epoch)
    else:
        external_model = None
    model_func = get_socket_func(
        TRAINERS, model_cfg, args.epoch, external_model=external_model
    )
    server = build_server(
        args.server_type,
        socket_func=model_func,
        cfg=model_cfg.cfg,
        hostname=args.hostname,
        port=args.port,
        client_type=args.client_type,
    )

    server.listen()


if __name__ == "__main__":
    main()
