"""Entry point for the Foundry application."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="The Foundry Project")
    subparsers = parser.add_subparsers(dest="command")

    # Server command
    serve_parser = subparsers.add_parser("serve", help="Start the Foundry web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--no-browser", action="store_true")

    args = parser.parse_args()

    if args.command == "serve":
        from .chamber.server import start_server
        start_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
