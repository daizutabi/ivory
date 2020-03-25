import argparse
import datetime
import os
import sys

from termcolor import cprint

from ivory.core.client import create_client
from ivory.core.parser import Parser

if "." not in sys.path:
    sys.path.insert(0, ".")


def normpath(path):
    if "." not in path:
        path = path + ".yaml"
    if not os.path.exists(path):
        cprint(f"No sufh file: {path}", "red", attrs=["bold", "dark"], file=sys.stderr)
        sys.exit()
    return path


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="PATH", help="an parameter YAML file path")
    parser.add_argument("action", metavar="X", help="command and/or argst", nargs="*")
    parser.add_argument("-r", "--repeat", type=int, default=1)
    parser.add_argument("-m", "--message", default="")
    args = parser.parse_args()

    if args.path == "ui" and os.path.exists("client.yaml"):
        args.path = "client"
        args.action = ["ui"]

    if not args.action:
        args.action = ["product"]
    elif "=" in args.action[0]:
        args.action.insert(0, "product")

    path = normpath(args.path)
    message = args.message
    repeat = args.repeat
    cmd = args.action[0]
    args = args.action[1:]

    if cmd == "show":
        with open(path) as file:
            params_yaml = file.read()
        print(params_yaml)
        sys.exit()

    client = create_client(path)
    if cmd in ["product", "chain"]:
        args = Parser().parse_args(args).args
        for run in getattr(client, cmd)(args, repeat, message):
            run.start()

    elif cmd in ["optimize", "tune"]:
        if "=" in args[0]:
            name = None
        else:
            name, args = args[0], args[1:]
        options = Parser().parse_args(args).options
        client.optimize(name, options, message)

    elif cmd in ["search", "list"]:
        if "=" in args[0]:
            mode = None
        else:
            mode, args = args[0], args[1:]
        parser = Parser().parse_args(args)
        params = dict(zip(parser.args.keys(), parser.values))
        for run in client.search_runs(mode, params, message):
            run_id = run.info.run_id
            start_dt = datetime.datetime.fromtimestamp(run.info.start_time / 1e3)
            start_dt = start_dt.strftime("%Y-%m-%d %H:%M:%S")
            print(run_id, start_dt)

    elif cmd == "ui":
        client.ui()


def main():
    cli()


if __name__ == "__main__":
    main()
