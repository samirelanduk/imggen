
import argparse
from .latent import save_blank_latent

def main():
    parser = argparse.ArgumentParser(prog="pydiffuser")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("save_blank_latent")
    p.add_argument("--path", type=str, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.set_defaults(func=lambda args: save_blank_latent(args.path, args.width, args.height))

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)

if __name__ == "__main__":
    main()