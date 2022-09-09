import re
import unicodedata

from loguru import logger


alphabet_filter = {
    'latin': re.compile(r'[^A-Za-z\' -]'),
    'cyr': re.compile(r'[^ыёэъЫЁЭЪйцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]'),
    'uk': re.compile(         r'[^йцукенгшщзхїфивапролджєґячсміiтьбюЙЦУКЕНГШЩЗХЇФИВАПРОЛДЖЄҐЯЧСМІТЬБЮ\' -]')
}
re_punct = re.compile(r'[\.,!?"«»“”…:;–—-]+')
re_whitespace = re.compile(r'[\s-]+')
re_leading = re.compile(r'^[\'-]+')
re_trailing = re.compile(r'[\'-]+$')


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' and c != 'й')


def keep_useful_characters(sentence, alphabet='cyr', utterance_id='sentence'):
    s = sentence.lower()
    if alphabet != 'latin':
        s = s.replace('e', 'е')
        s = s.replace('i', 'і')
        s = s.replace('o', 'о')
        s = s.replace('p', 'р')
        s = s.replace('x', 'х')
        s = s.replace('y', 'у')
        if alphabet == 'uk':
            s = s.replace('ы', 'и')
    s = s.replace('’', "'")
    s = s.replace('`', "'")
    s = s.replace('՚', "'")
    s = re_punct.sub(' ', s)
    s1 = s
    s1 = strip_accents(s1)
    s = alphabet_filter[alphabet].sub('', s1)
    if s1 != s:
        logger.warning('skipping suspicious {}: |{}|{}|', utterance_id, (sentence, s1), s)
        return None
    s = re_whitespace.sub(' ', s)
    s = re_leading.sub('', s)
    s = re_trailing.sub('', s)
    s = s.strip()
    return s



if __name__ == '__main__':
    import sys

    for line in sys.stdin:
        utt_id, text = line.strip().split(maxsplit=1)
        print(utt_id, keep_useful_characters(text))
