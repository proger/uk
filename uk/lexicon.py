import sys

if __name__ == '__main__':
    oov = set()

    for line in sys.stdin:
        oov = oov.union(line.strip().split()[1:])

    with open('data/local/dict/lexicon.txt') as f:
        for line in f:
            word = line.strip().split()[0]
            oov.discard(word)

    print(*sorted(oov), sep='\n')