import pytest
from spacy.lang.en import English
from spacy.tokens import Doc

import data_generation.decoders as r


def parsed(monkeypatch, words, heads, deps, pos, tags=None, morphs=None):
    doc = Doc(English().vocab, words=words.split(), heads=heads, deps=deps.split(),
              pos=pos.split(), tags=(tags.split() if tags else None), morphs=morphs)
    monkeypatch.setattr(r, "grammar_parser", lambda: lambda text: doc)
    return " ".join(words.split())


@pytest.mark.parametrize("decoder,expected", [("subordinate_order", "0"), ("subordinate_presence", "1")])
def test_subordinate_first(monkeypatch, decoder, expected):
    text = parsed(monkeypatch, "When air cools , water condenses .", [2,2,5,5,5,5,5],
                  "mark nsubj advcl punct nsubj ROOT punct", "SCONJ NOUN VERB PUNCT NOUN VERB PUNCT")
    assert r.grammar_units(text, decoder)[0][0] == expected


def test_subordinate_last(monkeypatch):
    text = parsed(monkeypatch, "Water condenses when air cools .", [1,1,4,4,1,1],
                  "nsubj ROOT mark nsubj advcl punct", "NOUN VERB SCONJ NOUN VERB PUNCT")
    assert r.grammar_units(text, "subordinate_order")[0][0] == "1"


@pytest.mark.parametrize("decoder,expected", [("subordinate_presence", "0"), ("voice", "0")])
def test_simple_active(monkeypatch, decoder, expected):
    text = parsed(monkeypatch, "Water moves .", [1,1,1], "nsubj ROOT punct", "NOUN VERB PUNCT")
    assert r.grammar_units(text, decoder)[0][0] == expected


def test_passive(monkeypatch):
    text = parsed(monkeypatch, "Water is heated .", [2,2,2,2], "nsubjpass auxpass ROOT punct", "NOUN AUX VERB PUNCT")
    assert r.grammar_units(text, "voice")[0][0] == "1"


def test_noun_opening(monkeypatch):
    text = parsed(monkeypatch, "The cold air moves .", [2,2,3,3,3], "det amod nsubj ROOT punct", "DET ADJ NOUN VERB PUNCT")
    assert r.grammar_units(text, "opening_pos")[0][0] == "0"


def test_verb_opening(monkeypatch):
    text = parsed(monkeypatch, "Move slowly .", [0,0,0], "ROOT advmod punct", "VERB ADV PUNCT")
    assert r.grammar_units(text, "opening_pos")[0][0] == "1"


def test_noun_ending(monkeypatch):
    text = parsed(monkeypatch, "Study birds .", [0,0,0], "ROOT dobj punct", "VERB NOUN PUNCT")
    assert r.grammar_units(text, "ending_pos")[0][0] == "0"


def test_verb_ending(monkeypatch):
    text = parsed(monkeypatch, "Birds fly .", [1,1,1], "nsubj ROOT punct", "NOUN VERB PUNCT")
    assert r.grammar_units(text, "ending_pos")[0][0] == "1"


def test_one_adjective(monkeypatch):
    text = parsed(monkeypatch, "A cold wind blows .", [2,2,3,3,3], "det amod nsubj ROOT punct", "DET ADJ NOUN VERB PUNCT")
    assert r.grammar_units(text, "adjective_count")[0][0] == "0"


def test_two_adjectives(monkeypatch):
    text = parsed(monkeypatch, "A strong cold wind blows .", [3,3,3,4,4,4], "det amod amod nsubj ROOT punct", "DET ADJ ADJ NOUN VERB PUNCT")
    assert r.grammar_units(text, "adjective_count")[0][0] == "1"


def test_superlative(monkeypatch):
    text = parsed(monkeypatch, "The best answer wins .", [2,2,3,3,3], "det amod nsubj ROOT punct", "DET ADJ NOUN VERB PUNCT",
                  morphs=["", "Degree=Sup", "", "", ""])
    assert r.grammar_units(text, "adjective_degree")[0][0] == "1"


def test_ordinary_adjective(monkeypatch):
    text = parsed(monkeypatch, "A useful answer helps .", [2,2,3,3,3], "det amod nsubj ROOT punct", "DET ADJ NOUN VERB PUNCT")
    assert r.grammar_units(text, "adjective_degree")[0][0] == "0"


def test_relative_clause(monkeypatch):
    text = parsed(monkeypatch, "Devices that store energy work .", [4,2,0,2,4,4], "nsubj nsubj relcl dobj ROOT punct", "NOUN PRON VERB NOUN VERB PUNCT")
    assert r.grammar_units(text, "noun_modifier")[0][0] == "0"


def test_participial_modifier(monkeypatch):
    text = parsed(monkeypatch, "Devices storing energy work .", [3,0,1,3,3], "nsubj acl dobj ROOT punct", "NOUN VERB NOUN VERB PUNCT", "NNS VBG NN VBP .")
    assert r.grammar_units(text, "noun_modifier")[0][0] == "1"
