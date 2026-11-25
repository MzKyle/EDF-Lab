from sys import argv

from pollen_edf.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["generate-test-data", *argv[1:]]))
