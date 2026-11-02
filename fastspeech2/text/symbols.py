""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict
from text.telugu import TELUGU_PHONES

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols so they stay unique (some match upper-case letters)
_arpabet = ["@" + s for s in cmudict.valid_symbols]
# [MOD] Mandarin pinyin symbols replaced by the Telugu phone set (lower-case
# labels, so they cannot collide with the upper-case ARPAbet symbols).
_telugu = ["@" + s for s in TELUGU_PHONES]

# Export all symbols
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _telugu
    + _silences
)
