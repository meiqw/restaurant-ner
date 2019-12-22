from spacy.tokens import Doc


def print_doc(doc: Doc) -> None:
    for sent in doc.sents:
        for tok in sent:
            print(tok, tok.i, tok.ent_type_, tok.ent_iob_)
        print()


def print_ents(doc: Doc, *, ents_only: bool = False) -> None:
    for sent in doc.sents:
        if not ents_only:
            print(sent)

        for ent in sent.ents:
            print(ent, ent.label_, sep="\t")

        if not ents_only:
            print()
