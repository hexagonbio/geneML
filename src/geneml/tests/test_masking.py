import unittest

from geneml.utils import mask_lowercase_stretches


class TestMaskLowercaseStretches(unittest.TestCase):

    def test_stretch_longer_than_min_length(self):
        """Test a lowercase stretch that is longer than the default min_length."""
        long_stretch = 'a' * 250
        seq = 'ACGT' + long_stretch + 'GCTA'
        expected = 'ACGT' + 'N' * 250 + 'GCTA'
        self.assertEqual(mask_lowercase_stretches(seq), expected)

    def test_stretch_equal_to_min_length(self):
        """Test a lowercase stretch that is exactly the default min_length."""
        exact_stretch = 'c' * 200
        seq = 'ACGT' + exact_stretch + 'GCTA'
        expected = 'ACGT' + 'N' * 200 + 'GCTA'
        self.assertEqual(mask_lowercase_stretches(seq), expected)

    def test_stretch_shorter_than_min_length(self):
        """Test a lowercase stretch that is shorter than the default min_length, which should not be masked."""
        short_stretch = 'g' * 199
        seq = 'ACGT' + short_stretch + 'GCTA'
        self.assertEqual(mask_lowercase_stretches(seq), seq)

    def test_no_lowercase_stretches(self):
        """Test a sequence with no lowercase letters."""
        seq = 'ACGTACGTACGT'
        self.assertEqual(mask_lowercase_stretches(seq), seq)

    def test_mixed_stretches(self):
        """Test a sequence with multiple stretches, some to be masked and some not."""
        long_stretch = 'a' * 210
        short_stretch = 't' * 50
        seq = 'ACGT' + long_stretch + 'GG' + short_stretch + 'TT'
        expected = 'ACGT' + 'N' * 210 + 'GG' + short_stretch + 'TT'
        self.assertEqual(mask_lowercase_stretches(seq), expected)

    def test_empty_sequence(self):
        """Test an empty input string."""
        seq = ''
        self.assertEqual(mask_lowercase_stretches(seq), '')

    def test_custom_min_length(self):
        """Test the function with a custom min_length parameter."""
        stretch = 'c' * 50
        seq = 'ACGT' + stretch + 'GCTA'
        # With min_length=40, it should be masked
        expected_masked = 'ACGT' + 'N' * 50 + 'GCTA'
        self.assertEqual(mask_lowercase_stretches(seq, min_length=40), expected_masked)
        # With min_length=60, it should not be masked
        self.assertEqual(mask_lowercase_stretches(seq, min_length=60), seq)

    def test_sequence_with_no_lowercase(self):
        """Test a sequence that is entirely uppercase."""
        seq = 'AGCTGTGACGTATGCTAGCTAGCTACG'
        self.assertEqual(mask_lowercase_stretches(seq), seq)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
