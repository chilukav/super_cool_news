#!/usr/bin/env python3

import argparse
import pathlib
from collections import namedtuple


Command = namedtuple('Command', ['name', 'help'])
COMMANDS = [
    Command('language', 'Language'),
    Command('news', 'News'),
    Command('categories', 'Categories'),
    Command('threads', 'Threads'),
    Command('top', 'Top'),
]


def is_directory(path: str):
    path = pathlib.Path(path)
    if not path.exists() and not path.is_dir():
        raise argparse.ArgumentTypeError()
    return path


def main(command, source_dir: pathlib.Path):
    for file in source_dir.rglob('*'):
        if not file.is_file():
            pass
        print(file.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")
    for command in COMMANDS:
        command_parser = sub_parser.add_parser(command.name, help=command.help)
        command_parser.add_argument('source_dir', type=is_directory, metavar='DIRECTORY', help='Path to source dir')

    args = parser.parse_args()
    main(args.command, args.source_dir)
