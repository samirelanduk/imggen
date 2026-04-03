
import argparse
from .clip import tokenize, embed, encode
from .latent import save_blank_latent

def main():
    parser = argparse.ArgumentParser(prog="pydiffuser")
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("clip_tokenize")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--clip_tokenizer", type=str, required=True)
    p.add_argument("--tokens", type=str, required=False)
    p.add_argument("--mappings", type=str, required=False)
    p.set_defaults(func=lambda args: tokenize(args.text, args.clip_tokenizer, args.tokens, args.mappings))

    p = subparsers.add_parser("clip_embed")
    p.add_argument("--tokens", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, required=False)
    p.set_defaults(func=lambda args: embed(args.tokens, args.model, args.output))

    p = subparsers.add_parser("clip_encode")
    p.add_argument("--embedding", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, required=False)
    p.set_defaults(func=lambda args: encode(args.embedding, args.model, args.output))

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