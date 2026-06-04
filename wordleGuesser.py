unknown = 'QWYISGHJKLZXVBERA'

poss_words = [
    'A',
    unknown,
    'R',
    'E',
    unknown
]

def getWords(poss_words: list[str], pos = 0, word =''):
    if (pos < 0) or (pos >= len(poss_words)):
        print(word)
        return

    for c in poss_words[pos]:
        getWords(poss_words, pos + 1, word + c)

getWords(poss_words)
