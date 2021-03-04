import json
import unittest
from typing import List

import spacy
from spacy.language import Language
from spacy.tokens import Span

from util_files.ingest import spacy_doc_from_sentences
from utils import PRF1
from restaurant_ner import ingest_json_document, span_prf1_type_map

NLP = spacy.load("en_core_web_sm", disable=["ner"])


class TestHW4(unittest.TestCase):
    def assertPRF1DictAlmostEqual(self, d1: dict, d2: dict) -> None:
        for key in sorted(d1.keys() | d2.keys()):
            self.assertIn(key, d1)
            self.assertIn(key, d2)
            for val1, val2, field in zip(d1[key], d2[key], d1[key]._fields):
                self.assertAlmostEqual(
                    val1, val2, 2, f"{field} for key {repr(key)} not equal"
                )

    def test_ingest_simple(self):
        doc = ingest_json_document(
            json.loads(
                """{"id": 106, "text": "I ordered the hamburger and the fries.", "meta": {}, "annotation_approver": null, "labels": [[14, 23, "DISH"], [32, 37, "DISH"]]}"""
            ),
            NLP,
        )
        self.assertEqual("I ordered the hamburger and the fries.", doc.text)
        self.assertSequenceEqual(
            [Span(doc, 3, 4, "DISH"), Span(doc, 6, 7, "DISH")], doc.ents
        )

    def test_ingest_noannotation(self):
        doc = ingest_json_document(
            json.loads(
                """{"id": 107, "text": "Nothing to annotate here!", "meta": {}, "annotation_approver": "admin", "labels": []}"""
            ),
            NLP,
        )
        self.assertEqual("Nothing to annotate here!", doc.text)
        self.assertSequenceEqual([], doc.ents)

    def test_span_prf1_type_map(self):
        docs = [[["George", "Washington"]], [["Maryland"]], [["Asia"]]]

        # Check that testing reference is perfect
        correct_bio = [["B-PER", "I-PER"], ["B-GPE"], ["B-LOC"]]
        ref = _create_docs(docs, correct_bio, NLP)
        self.assertPRF1DictAlmostEqual(
            {
                "": PRF1(1.0, 1.0, 1.0),
                "PER": PRF1(1.0, 1.0, 1.0),
                "GPE": PRF1(1.0, 1.0, 1.0),
                "LOC": PRF1(1.0, 1.0, 1.0),
            },
            span_prf1_type_map(ref, ref),
        )
        # Empty type map has no effect
        self.assertPRF1DictAlmostEqual(
            {
                "": PRF1(1.0, 1.0, 1.0),
                "PER": PRF1(1.0, 1.0, 1.0),
                "GPE": PRF1(1.0, 1.0, 1.0),
                "LOC": PRF1(1.0, 1.0, 1.0),
            },
            span_prf1_type_map(ref, ref, type_map={}),
        )
        # Remapping types without changing performance
        backwards_type_map = {"GPE": "EPG", "LOC": "COL", "PER": "REP"}
        self.assertPRF1DictAlmostEqual(
            {
                "": PRF1(1.0, 1.0, 1.0),
                "REP": PRF1(1.0, 1.0, 1.0),
                "EPG": PRF1(1.0, 1.0, 1.0),
                "COL": PRF1(1.0, 1.0, 1.0),
            },
            span_prf1_type_map(ref, ref, type_map=backwards_type_map),
        )

        # Two incorrect entities: first PER is truncated, final GPE is a LOC
        incorrect_bio1 = [["B-PER", "O"], ["B-GPE"], ["B-GPE"]]
        incorrect1 = _create_docs(docs, incorrect_bio1, NLP)
        self.assertPRF1DictAlmostEqual(
            {
                "": PRF1(0.3333, 0.3333, 0.3333),
                "PER": PRF1(0.0, 0.0, 0.0),
                "GPE": PRF1(0.5, 1.0, 0.6666),
                "LOC": PRF1(0.0, 0.0, 0.0),
            },
            span_prf1_type_map(ref, incorrect1),
        )
        # When GPE and LOC are collapsed, only one error
        gpe_loc_map = {"GPE": "GPE_LOC", "LOC": "GPE_LOC"}
        self.assertPRF1DictAlmostEqual(
            {
                "": PRF1(0.6666, 0.6666, 0.6666),
                "PER": PRF1(0.0, 0.0, 0.0),
                "GPE_LOC": PRF1(1.0, 1.0, 1.0),
            },
            span_prf1_type_map(ref, incorrect1, type_map=gpe_loc_map),
        )


def _create_docs(
    doc_sentences: List[List[List[str]]], bios: List[List[str]], nlp: Language
):
    return [
        spacy_doc_from_sentences(doc, bio, nlp) for doc, bio in zip(doc_sentences, bios)
    ]


if __name__ == "__main__":
    unittest.main()
