"""Checks for original-abstract ending review rules."""

import unittest

from export_claw_edit_analysis import missing_terminal_punctuation


class AbstractEndingTests(unittest.TestCase):
    def test_terminal_punctuation_with_closing_quote(self) -> None:
        self.assertFalse(missing_terminal_punctuation('结论。”'))
        self.assertFalse(missing_terminal_punctuation('结论。  '))

    def test_incomplete_and_unpunctuated_endings(self) -> None:
        self.assertTrue(missing_terminal_punctuation('但是,'))
        self.assertTrue(missing_terminal_punctuation('玩家可'))
        self.assertTrue(missing_terminal_punctuation('结论'))


if __name__ == "__main__":
    unittest.main()
