import pandas as pd

from src.data.vocab import field_dims, fit_vocab, transform_col


def test_oov_is_zero_and_unseen_does_not_expand_dim():
    train = pd.DataFrame({"video_id": [10, 11, 10]})
    vocabs = fit_vocab(train, ["video_id"])
    dims = field_dims(vocabs)
    encoded = transform_col(pd.Series([10, 99, 11]), vocabs["video_id"])
    assert encoded[1] == 0
    assert encoded[0] != 0 and encoded[2] != 0
    assert dims["video_id"] == 3
    assert set(encoded.tolist()) <= {0, 1, 2}
