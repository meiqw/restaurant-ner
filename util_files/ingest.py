from itertools import chain
from typing import List, TextIO, Generator

from spacy.gold import spans_from_biluo_tags, iob_to_biluo
from spacy.language import Language
from spacy.tokens import Doc


class CoNLLIngester:
    def __init__(self) -> None:
        self.sentences: List[List[str]] = []
        self.tokens: List[str] = []
        self.labels: List[str] = []

    def reset(self) -> None:
        self.sentences = []
        self.tokens = []
        self.labels = []

    def _create_document(self, nlp: Language) -> Doc:
        assert self.sentences, "No sentences to create document with"
        doc = spacy_doc_from_sentences(self.sentences, self.labels, nlp)
        self.reset()
        return doc

    def extract_docs(self, corpus: TextIO, nlp: Language) -> Generator[Doc, None, None]:
        for line in corpus:
            if line.startswith("-DOCSTART-"):
                # Add the last sentence if needed
                if self.tokens:
                    self.sentences.append(self.tokens)

                # Create a document if there are sentences, otherwise we are at the first
                # docstart in the corpus and there's nothing to create
                if self.sentences:
                    yield self._create_document(nlp)
            elif line.strip():
                # Sample line:
                # German JJ B-NP B-MISC
                fields = line.split()
                self.tokens.append(fields[0])
                self.labels.append(fields[-1])
            else:
                # End of sentence if there are tokens, otherwise this is a blank leading
                # line before the first sentence and there's nothing to do.
                if self.tokens:
                    self.sentences.append(self.tokens)
                    self.tokens = []

        # Finish off document
        if self.tokens:
            self.sentences.append(self.tokens)
        yield self._create_document(nlp)


def load_conll2003(path: str, nlp: Language) -> List[Doc]:
    with open(path, encoding="utf8") as corpus:
        return list(CoNLLIngester().extract_docs(corpus, nlp))


def spacy_doc_from_sentences(
    sentences: List[List[str]], labels: List[str], nlp: Language
) -> Doc:
    # Create initial doc
    all_tokens = list(chain.from_iterable(sentences))
    # Mark that every token is followed by space
    spaces = [True] * len(all_tokens)
    doc = Doc(nlp.vocab, words=all_tokens, spaces=spaces)

    # Set sentence boundaries
    tok_idx = 0
    for sentence in sentences:
        for sentence_idx in range(len(sentence)):
            # First token should have start to True, all others False
            doc[tok_idx].is_sent_start = sentence_idx == 0
            tok_idx += 1

    if labels:
        if len(labels) != len(all_tokens):
            raise ValueError(
                f"Number of labels ({len(labels)}) does not match number of tokens ({len(all_tokens)})"
            )

        # Create entities after converting IOB (actually BIO) to BILUO
        doc.ents = spans_from_biluo_tags(doc, iob_to_biluo(labels))

    return doc
