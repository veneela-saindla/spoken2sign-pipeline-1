from modules.preprocess import clean_sequence
from modules.translate import translate_text

def build_sequence_from_text(text, kp_data, vocab):
    gloss_list = translate_text(text, vocab)
    seqs = []

    for vid in gloss_list:
        raw = kp_data[vid]
        seqs.append(clean_sequence(raw))

    return seqs
