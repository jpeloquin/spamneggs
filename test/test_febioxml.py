import unittest
from spamneggs.febioxml import _parse_selector_part

class TestParseSelectorPart(unittest.TestCase):

    def test_nospace_unquoted_d1(self):
        name, components =_parse_selector_part("node[28]")
        assert(name == "node")
        assert(len(components) == 1)
        assert(components[0] == 28)

    def test_nospace_squoted_d1(self):
        name, components =_parse_selector_part(" 'node'[28]")
        assert(name == "node")
        assert(len(components) == 1)
        assert(components[0] == 28)

    def test_nospace_dquoted_d1(self):
        name, components =_parse_selector_part('"node"[28] ')
        assert(name == "node")
        assert(len(components) == 1)
        assert(components[0] == 28)

    def test_withspace_unquoted_d2(self):
        name, components =_parse_selector_part("np1 np2[3]")
        assert(name == "np1 np2")
        assert(len(components) == 1)
        assert(components[0] == 3)

    def test_withspace_squoted_d2(self):
        name, components =_parse_selector_part(" 'np1 np2'[3]")
        assert(name == "np1 np2")
        assert(len(components) == 1)
        assert(components[0] == 3)

    def test_withspace_dquoted_d2(self):
        name, components =_parse_selector_part('"np1 np2"[3] ')
        assert(name == "np1 np2")
        assert(len(components) == 1)
        assert(components[0] == 3)

    def test_int_id(self):
        name, id_ =_parse_selector_part("entity_name[5]")
        assert(name == "entity_name")
        assert(len(id_) == 1)
        assert(id_[0] == 5)

    def test_int_ids(self):
        name, id_ =_parse_selector_part("entity_name[2,3]")
        assert(name == "entity_name")
        assert(len(id_) == 2)
        assert(id_[0] == 2)
        assert(id_[1] == 3)

    def test_tuple_id(self):
        name, id_ =_parse_selector_part("entity_name[(1, 2, 3)]")
        assert(name == "entity_name")
        assert(len(id_) == 1)
        assert(id_[0] == (1, 2, 3))
