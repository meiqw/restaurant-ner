import json
from typing import Mapping, Sequence, Dict, Optional
from typing import Iterable, Sequence, Tuple, List, Dict
import utils
from utils import FeatureExtractor, ScoringCounts, ScoringEntity, EntityEncoder, PRF1
import spacy
from spacy.tokens import Doc, Token, Span
from collections import defaultdict
from utils import PRF1
from spacy.language import Language
from pymagnitude import Magnitude
from collections import Counter
import pycrfsuite
import sys
from decimal import ROUND_HALF_UP, Context

class BiasFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            features["bias"] = 1.0


class TokenFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["tok[%d]=%s" % (relative_idx, token)] = 1.0

class UppercaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if token.isupper():
            features["uppercase[%d]" % (relative_idx)] = 1.0


class TitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if token.istitle():
            features["titlecase[%d]" % (relative_idx)] = 1.0


class InitialTitlecaseFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if token.istitle() and current_idx + relative_idx == 0:
            features["initialtitlecase[%d]" % relative_idx] = 1.0


class PunctuationFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if utils.PUNC_REPEAT_RE.match(token):
            features["punc[%d]" % relative_idx] = 1.0


class DigitFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if utils.DIGIT_RE.search(token):
            features["digit[%d]" % relative_idx] = 1.0

class WordShapeFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        shape = token
        shape = utils.UPPERCASE_RE.sub("X", shape)
        shape = utils.LOWERCASE_RE.sub("x", shape)
        shape = utils.DIGIT_RE.sub("0", shape)
        features["shape[%d]=%s" % (relative_idx, shape)] = 1.0

class LengthFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["tok_length=%d" % len(token)] = 1.0

class PrefixFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["tok_prefix=%s" % token[:3]] = 1.0

class SuffixFeature(FeatureExtractor):
    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        features["tok_suffix=%s" % token[-2:]] = 1.0

class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.feature_extractors = feature_extractors
        self.window_size = window_size

    def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        dict_list = []
        for(i, tok) in enumerate(tokens):
            features = {}
            for j in range(-self.window_size, self.window_size+1):
                # i + j: absolute index of the focus token in tokens
                if i + j >= 0 and i + j < len(tokens):
                    for extractor in self.feature_extractors:
                        extractor.extract(tokens[i+j], i, j, tokens, features)
            dict_list.append(features)
        #print(dict_list)
        return dict_list

class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self.entity_encoder = encoder
        self.tagger = None

    @property
    def encoder(self) -> EntityEncoder:
        return self.entity_encoder

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        self.trainer = pycrfsuite.Trainer(algorithm, params, verbose=False)
        self.trainer.set_params(params)
        for doc in docs:
            for sent in doc.sents:
                tokens = [token.text for token in list(sent)]
                features = self.feature_extractor.extract(tokens)
                encoding = self.entity_encoder.encode(sent)
                self.trainer.append(features, encoding)
        self.trainer.train(path)
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        if not self.tagger:
            raise ValueError
        entities = []
        for sent in doc.sents:
            tokenStr = [token.text for token in list(sent)]
            features = self.feature_extractor.extract(tokenStr)
            labels = self.tagger.tag(features)
            spans = decode_bilou(labels, list(sent), doc)
            entities.extend(spans)
        doc.ents = entities

        return doc

    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        return self.tagger.tag(tokens)

def decode_bilou(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    span_lst = []
    new_labels = change_il(labels)
    start = -1
    end = -1
    for j, label in enumerate(new_labels):
        if j != 0 and new_labels[j - 1] != 'O' and label[0] in 'BUO':
            ent_type = new_labels[j - 1][2:]
            span_lst.append(Span(doc, start, end, ent_type))
            start = tokens[j].i
            end = start + 1
        elif label[0] in 'BUO':
            start = tokens[j].i
            end = start + 1
        else:
            end += 1

    if new_labels[-1] != 'O':
        ent_type = new_labels[-1][2:]
        span_lst.append(Span(doc, start, end, ent_type))

    return span_lst


def change_il(labels: Sequence[str]) -> Sequence[str]:
    res = []
    for j, label in enumerate(labels):
        if label[0] in 'IL' and \
                (j == 0 or labels[j - 1] == 'O' or label[2:] != labels[j - 1][2:]):
            res.append('B' + label[1:])
        else:
            res.append(label)
    return res

class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        res = []
        length = 0
        prev = ""
        for i, tok in enumerate(tokens):
            if tok.ent_iob_ == "":
                if length > 0:
                    res.extend(self.helper(length, prev))
                res.append("O")
                length = 0
            elif tok.ent_iob_ == "I":
                length += 1
                prev = tok.ent_type_
            elif tok.ent_iob_ == "B":
                if length > 0:
                    res.extend(self.helper(length, prev))
                length = 1
                prev = tok.ent_type_

        if length > 0:
            res.extend(self.helper(length, prev))
        return res


    def helper(self, length, label):
        res = []
        if length == 1:
            res.append("U-" + label)
        elif length > 1:
            res.append("B-" + label)
            for i in range(length - 2):
                res.append("I-" + label)
            res.append("L-" + label)
        return res


class BIOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        res = []
        for tok in tokens:
            if tok.ent_iob_ == "":
                res.append("O")
            else:
                res.append(tok.ent_iob_ + "-" + tok.ent_type_)
        return res


class IOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        res = []
        for tok in tokens:
            #print("tok.ent_iob_", tok.ent_iob_)
            if tok.ent_iob_ == "":
                res.append("O")
            else:
                res.append("I-" + tok.ent_type_)
            #print(res)
        return res

class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.vectors = Magnitude(vectors_path, normalized=False)
        self.scaling = scaling

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            vector = self.vectors.query(token)
            keys = ["v"+str(i) for i in range(self.vectors.dim)]
            values = vector * self.scaling
            features.update(zip(keys, values))

class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
        # clusters = open(clusters_path)
        with open(clusters_path) as clusters:
            self.dict = {}
            for line in clusters:
                lst = line.split("\t")
                self.dict[lst[1]] = lst[0]
        self.use_full_paths = use_full_paths
        self.use_prefixes = use_prefixes
        self.prefixes = prefixes
        if not self.use_full_paths and not self.use_prefixes:
            raise ValueError('neither use_full_paths nor use_prefixes are True')

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0:
            if token in self.dict:
                path = self.dict[token]
                if self.use_full_paths:
                    features["cpath=%s" % path] = 1.0
                elif self.use_prefixes and self.prefixes is None:
                    for i in range(len(path)):
                        features["cprefix%d=%s" % (i + 1, path[:i + 1])] = 1.0
                elif self.use_prefixes and self.prefixes is not None:
                    for i in self.prefixes:
                        if i <= len(path):
                            features["cprefix%d=%s" % (i, path[:i])] = 1.0

def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:
    true_pos = []
    false_pos = []
    false_neg = []

    if typed:
        for i in range(len(reference_docs)):
            ref_ents = reference_docs[i].ents
            test_ents = test_docs[i].ents
            true_pos_set = set()
            for j in range(len(ref_ents)):
                ref_ent = ref_ents[j]
                flag = 0
                for k in range(len(test_ents)):
                    test_ent = test_ents[k]
                    if ref_ent.start == test_ent.start and \
                            ref_ent.end == test_ent.end and ref_ent.label_ == test_ent.label_:
                        tokens = tuple(w.text for w in ref_ent)
                        true_pos.append(ScoringEntity(tokens, ref_ent.label_))
                        true_pos_set.add(test_ent)
                        flag = 1
                if flag == 0:
                    tokens = tuple(w.text for w in ref_ent)
                    false_neg.append(ScoringEntity(tokens, ref_ent.label_))

            test_ents_set = set(test_ents)
            false_pos_set = test_ents_set - true_pos_set
            false_pos.extend([
                ScoringEntity(tuple(w.text for w in e), e.label_)
                for e in false_pos_set
            ])
    else:
        for i in range(len(reference_docs)):
            ref_ents = reference_docs[i].ents
            test_ents = test_docs[i].ents
            true_pos_set = set()
            for j in range(len(ref_ents)):
                ref_ent = ref_ents[j]
                flag = 0
                for k in range(len(test_ents)):
                    test_ent = test_ents[k]
                    if ref_ent.start == test_ent.start and ref_ent.end == test_ent.end:
                        tokens = tuple(w.text for w in ref_ent)
                        true_pos.append(ScoringEntity(tokens, ""))
                        true_pos_set.add(test_ent)
                        flag = 1
                if flag == 0:
                    tokens = tuple(w.text for w in ref_ent)
                    false_neg.append(ScoringEntity(tokens, ""))

            test_ents_set = set(test_ents)
            false_pos_set = test_ents_set - true_pos_set
            false_pos.extend([
                ScoringEntity(tuple(w.text for w in e), "")
                for e in false_pos_set
            ])
    return ScoringCounts(Counter(true_pos), Counter(false_pos), Counter(false_neg))

def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:

    if not doc_json['labels'] and not doc_json['annotation_approver']:
        raise ValueError("The document hasn't been annotated yet.")

    doc = nlp(doc_json["text"])
    entities = []

    start_lst = [t.idx for t in doc]
    end_lst = [t.idx + len(t.text) for t in doc]

    text_start = 0
    text_end = len(doc_json["text"])-1

    for label in doc_json["labels"]:
        #print(label)
        if not index_check(text_start, text_end, label[0]) or \
                not index_check(text_start, text_end, label[1]) or label[1] < label[0]:
            raise ValueError("...")

        if label[0] > start_lst[-1]:
            start = len(start_lst)-1
        else:
            for i, char_start in enumerate(start_lst):
                if label[0] == char_start:
                    start = i
                    break
                if label[0] < char_start:
                    start = i - 1
                    break

        for i, char_end in enumerate(end_lst):
            if label[1] <= char_end:
                end = i+1
                break

        entities.append(Span(doc, start, end, label[2]))

    doc.ents = entities

    return doc

def index_check(start,end,index):
    if index < start or index > end:
        return False
    return True

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:

    true_pos = defaultdict(int)
    pre_denomi = defaultdict(int)
    rec_denomi = defaultdict(int)
    res = {}

    # per document
    for i in range(len(reference_docs)):
        ref_ents = reference_docs[i].ents
        test_ents = test_docs[i].ents

        for j in range(len(ref_ents)):
            ref_ent = ref_ents[j]
            ref_ent_label = type_map[ref_ent.label_] if (type_map and ref_ent.label_ in type_map) else ref_ent.label_

            rec_denomi[ref_ent_label] += 1

        for k in range(len(test_ents)):
            test_ent = test_ents[k]
            test_ent_label = type_map[test_ent.label_] if(type_map and test_ent.label_ in type_map) else test_ent.label_
            #print(test_ent_label)
            pre_denomi[test_ent_label] += 1

            for l in range(len(ref_ents)):
                ref_ent = ref_ents[l]
                ref_ent_label = type_map[ref_ent.label_] if (type_map and ref_ent.label_ in type_map) else ref_ent.label_
                if ref_ent.start == test_ent.start and \
                        ref_ent.end == test_ent.end and test_ent_label == ref_ent_label:
                            true_pos[ref_ent_label] += 1

    type_set = set().union(set(rec_denomi.keys()), set(pre_denomi.keys()))

    for k in type_set:
        precison = true_pos[k] / pre_denomi[k] if k in pre_denomi else 0
        recall = true_pos[k] / rec_denomi[k] if k in rec_denomi else 0
        f1 = 2 * precison * recall / (precison + recall) if precison != 0 and recall != 0 else 0
        res[k] = PRF1(precison, recall, f1)

    sum_true_pos = sum(true_pos.values())
    sum_pre_denom = sum(pre_denomi.values())
    sum_rec_denom = sum(rec_denomi.values())
    all_pre = sum_true_pos / sum_pre_denom if sum_pre_denom else 0
    all_rec = sum_true_pos / sum_rec_denom if sum_rec_denom else 0
    all_f1 = 2 * all_pre * all_rec / (all_pre + all_rec) if all_pre != 0 and all_rec != 0 else 0
    res[""] = PRF1(all_pre, all_rec, all_f1)

    return res

def experiment(featureExtractors, encoder, iteration):
    # print("Encoder:", str(type(encoder)).split('.')[-1][:-2],",", "Iteration:", iteration,",", "typed:", typed)
    #print([str(type(encoder)) for encoder in featureExtractors])
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))

    train = []
    valid = []
    with open('batch_9_meiqw.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines[:200]:
            #print(line)
            doc_json = json.loads(line)
            #print(doc_json)
            try:
                doc = ingest_json_document(doc_json, nlp)
                train.append(doc)
            except ValueError:
                continue
        for line in lines[200:]:
            #print(line)
            doc_json = json.loads(line)
            #print(doc_json)
            try:
                doc = ingest_json_document(doc_json, nlp)
                valid.append(doc)
            except ValueError:
                continue

    # Delete annotation on valid
    for doc in valid:
        doc.ents = []

    crf = CRFsuiteEntityRecognizer(
        WindowedTokenFeatureExtractor(
            featureExtractors,
            4,
        ),
        encoder,
    )
    crf.train(train, "ap", {"max_iterations": iteration}, "tmp.model")
    valid = [crf(doc) for doc in valid]

    # Load valid again to evaluate
    valid_gold = []
    with open('batch_9_meiqw.jsonl', 'r') as f:
        lines = f.readlines()
        for line in lines[200:]:
            doc_json = json.loads(line)
            # print(doc_json)
            try:
                valid_gold.append(ingest_json_document(doc_json, nlp))
            except ValueError:
                continue

    print("Type\tPrec\tRec\tF1", file=sys.stderr)
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    # Set typed=False for untyped scores
    map = {"DISH": "DISH_INGRED", "INGRED": "DISH_INGRED"}
    scores = span_prf1_type_map(valid_gold, valid, map)
    for ent_type, score in sorted(scores.items()):
        if ent_type == "":
            ent_type = "ALL"

        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        print("\t".join(fields), file=sys.stderr)

def main() -> None:

    extractor_1 = [BiasFeature(), TokenFeature(), UppercaseFeature(), TitlecaseFeature(), InitialTitlecaseFeature(), \
                 DigitFeature(), WordShapeFeature(),PunctuationFeature(), PrefixFeature(), SuffixFeature(),
                 WordVectorFeature("restaurant_reviews_all_truecase.magnitude", 0.5), \
                 BrownClusterFeature("restaurant_reviews_all_truecase_paths",  use_prefixes=True, prefixes=[8, 12, 16, 20])]
    experiment(extractor_1, BILOUEncoder(), 40)
    '''
    extractor_2 = [BiasFeature(), TokenFeature(), UppercaseFeature(), TitlecaseFeature(), InitialTitlecaseFeature(), \
                   DigitFeature(), WordShapeFeature(),
                   BrownClusterFeature("restaurant_reviews_all_truecase_paths", use_prefixes=True,
                                       prefixes=[8, 12, 16, 20])]
    experiment(extractor_2, BILOUEncoder(), 40)

    extractor_3 = [BiasFeature(), TokenFeature(), UppercaseFeature(), TitlecaseFeature(), InitialTitlecaseFeature(), \
                   DigitFeature(), WordShapeFeature(),
                   WordVectorFeature("restaurant_reviews_all_truecase.magnitude", 0.5)]
    experiment(extractor_3, BILOUEncoder(), 40)

    extractor_4 = [BiasFeature(), TokenFeature(), UppercaseFeature(), TitlecaseFeature(), InitialTitlecaseFeature(), \
                   DigitFeature(), WordShapeFeature()]
    experiment(extractor_4, BILOUEncoder(), 40)
    '''
if __name__ == '__main__':
    main()
