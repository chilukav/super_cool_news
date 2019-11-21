import os
import sys


def count_articles(path):
    ret = 0
    for f in os.listdir(path):
        with open(os.path.join(path, f)) as fd:
            ret += fd.read().count("\n\n\n")
    return ret


def main():
    path = sys.argv[1]
    cnt = count_articles(path)
    print(f"Found {cnt} articles")


main()
