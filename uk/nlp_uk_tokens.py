"""
Read tokens as in https://github.com/lang-uk/semesyuk-to-text
"""

import sys

from uk.common_voice import keep_useful_characters

def useful(tok):
    return len(tok) > 0

for line in sys.stdin:
    tokens = line.strip().split('|')
    tokens = [keep_useful_characters(tok) for tok in tokens]
    tokens = [tok for tok in tokens if useful(tok)]
    if tokens:
        print(' '.join(tokens))
