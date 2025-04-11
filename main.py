import argparse
from Interpreter import Interpreter


def main():
    parser = argparse.ArgumentParser(
        description="Pepper Programming Language Interpreter."
    )
    parser.add_argument(
        "filepath", nargs="?", help="The path to the script file to execute."
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable detailed execution tracing."
    )
    parser.add_argument(
        "-s",
        "--debug_show",
        action="store_true",
        help="Enable debug output specifically for SHOW statements.",
    )
    args = parser.parse_args()

    interpreter = Interpreter(args.debug, args.debug_show)

    if args.filepath:
        interpreter.run(args.filepath)
    else:
        interpreter.run_repl()


if __name__ == "__main__":
    main()
