import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="light-llm-wiki-pipeline")
    parser.add_argument("--stage", required=True, help="stage name, e.g. abbreviations")
    parser.add_argument("--document-id", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError
