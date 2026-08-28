import pytest

from hydra.config import DataConfig, ModelConfig

FILE1 = """\
@ comment line to be ignored
Ad\tad\tAPPR\t--
equu\tequus\tNA\tAkk.Sg
inhandon\tin+hant\tAPPR+NA\tc.D+Dat.Pl
gieng\tgan\tVVFIN\tInd.Past.Sg.3
weſman\twer+mann\tPW+NA\tNeut.Gen.Sg+Nom.Sg
zune\tze+wer+ne\tAPPR+PW+PTKNEG\tc.D+Neut.Instr.Sg+--
roſ\tross\tNA\tNom.Sg
ez\ter\tPPER\tNeut.Akk.Sg.3
mismatched\ta+b\tNA+VB+XX\tAkk.Sg
toolong\ta+b+c+d+e+f+g+h+i\tX+X+X+X+X+X+X+X+X\tY+Y+Y+Y+Y+Y+Y+Y+Y
"""

FILE2 = """\
min\tmin\tDPOSA\tNeut.Nom.Sg
ih\tich\tPPER\tNom.Sg.1
riten\triten\tVVINF\t--
nu\tnu\tAVD\t--
da\tdar\tAVD\t--
"""


@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "doc1.txt").write_text(FILE1, encoding="utf-8")
    (d / "doc2.txt").write_text(FILE2, encoding="utf-8")
    return d


@pytest.fixture
def data_cfg(corpus_dir):
    return DataConfig(corpus_dir=str(corpus_dir), max_word_len=12, max_lemma_len=16,
                      chunk_len=8, halo=4, multi_item_upsample=1)


@pytest.fixture
def model_cfg():
    return ModelConfig(d_char=8, d_tok=16, d_model=32, n_slots=4, kernel_size=3,
                       char_tcn_dilations=(1, 2), ctx_tcn_dilations=(1,),
                       d_dec=16, lemma_cross_attention=True, dropout=0.0)
