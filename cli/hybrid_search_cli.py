import argparse
from itertools import repeat

def normalize(values: list[float]) -> list[float]:
    if len(values) == 0:
        return []
    high = max(values)
    low = min(values)
    if high == low:
        return list(repeat(1.0, len(values)))

    k = high - low
    result = [(score - low) / k for score in values]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_cmd = subparsers.add_parser("normalize")
    normalize_cmd.add_argument("values", type=float, nargs="+")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            values = normalize(args.values)
            for score in values:
                print(f"* {score:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
