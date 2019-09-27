"""
Utils and wrappers for scoring taggers.
"""
from stanfordnlp.models.common.utils import ud_scores

def score(system_conllu_file, gold_conllu_file, verbose=True):
    """ Wrapper for tagger scorer. """
    evaluation = ud_scores(gold_conllu_file, system_conllu_file)
    el = evaluation['NER']
    p = el.precision
    r = el.recall
    f = el.f1
    if verbose:
        scores = [evaluation[k].f1 * 100 for k in ['NER']]
        print("NER")
        print("{:.3f}".format(*scores))
    return p, r, f

