def build_vocabulary(gloss_to_video, auto_map):
    vocab = {}
    for gloss, word in auto_map.items():
        if gloss in gloss_to_video:
            vocab[word] = gloss_to_video[gloss]
    return vocab

def translate_text(text, vocabulary):
    words = text.upper().split()
    out = []
    fallback = list(vocabulary.values())[0]

    for w in words:
        out.append(vocabulary.get(w, fallback))

    return out
