import unittest

import ID3
import parse
import pandas as pd

class CustomTest(unittest.TestCase):


    def test_ID3_1(self):
        data = parse.parse("tennis.data")
        tennis = pd.DataFrame(data)
        tree = ID3.ID3(tennis, None)
        tree.printNode(0)