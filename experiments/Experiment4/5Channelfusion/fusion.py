"""
Name: Sania Verma 
Student ID: 250059406
Experiment: 5 channel Fusion

Any system evaluated to date selects codes on the basis of **one** signal, at an **untuned** threshold,
and both leave something to be desired:

*   the channels offer different perspectives. Frequency prior alone is worth 45.7, kNN 51.4, the
    linear classifier 54.9, and Powell definitions contributed +10 points in candidate recall over
    prior+kNN according to the measurements made. So far, no one combines them.
*   the operating point is ad hoc. The top system operates at precision 44.9 vs recall 67.4 – it generates
    many more codes than exist in gold sets since 0.4 is a result of a sweep over one configuration.

This component evaluates (intervention, code) pairs according to five channels, combines them and selects
threshold for the maximum F1.
p_linear    One-vs-Rest Logistic Regression probability for that code
    s_knn       Weighted vote fraction of neighbours for that code
    p_prior     Prior probability of that code in training data
    s_category  Vote of the neighbours for that code's ERIC category (coarse-to-fine, §2.6)
    s_definition Cosine similarity of the intervention and Powell's definition of that code
    s_crossenc  Fine-tuned cross-encoder over (intervention, definition) pairs — optional 
                sixth input channel, see `label_crossencoder.py`

The blend weights and threshold are learned by inner cross-validation within the training fold, thus nothing is tuned on scoring data.
Channel 6 requires a stricter protocol compared to the other five. Channels 1 through 5 are inexpensive enough to be retrained within each inner fold, thus providing the honesty of their meta-features. However, cross-encoder is expensive, and the easy way to use it (scoring each intervention once with a pre-trained model based on some other partition and using the result as a feature) is a leakage: the feature of each row of the training data set would come from the model trained on the other fold. Thus, cross-encoder should also be retrained within each inner fold, for `inner_folds + 1` fine-tunes per outer fold. It is the expensive choice and the only choice which provides honesty of the blend weights.
"""
from __future__ import annotations

import numpy as np

import eric
from models import Model

STRATEGIES = None          # filled on first use


def _strategies():
    global STRATEGIES
    if STRATEGIES is None:
        STRATEGIES = eric.strategies()
    return STRATEGIES


class ChannelScorer:
    """Produces an (n_items x 73) score matrix per channel."""

    def __init__(self, knn_k: int = 5):
        self.knn_k = knn_k

    def fit(self, texts, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        strat = _strategies()
        self._index = {s: i for i, s in enumerate(strat)}
        n = len(texts)

        # shared vector space for interventions and label definitions
        defs = [f"{s}. {eric.definitions()[s]}" for s in strat]
        self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True,
                                    min_df=1)
        self._vec.fit(list(texts) + defs)
        X = self._norm(self._vec.transform(list(texts)).toarray())
        self._D = self._norm(self._vec.transform(defs).toarray())
        self._Xtr, self._ytr = X, list(labels)

        # prior
        counts = np.zeros(len(strat))
        for ls in labels:
            for c in ls:
                if c in self._index:
                    counts[self._index[c]] += 1
        self._prior = counts / max(n, 1)

        # one-vs-rest logistic regression, on the same features
        Xs = self._vec.transform(list(texts))
        self._clf = {}
        for c, i in self._index.items():
            y = np.array([1 if c in ls else 0 for ls in labels])
            if 3 <= y.sum() < len(y):
                m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
                m.fit(Xs, y)
                self._clf[i] = m

        # category membership matrix, 73 x 9
        cats = eric.categories()
        self._cat_index = {c: i for i, c in enumerate(cats)}
        self._member = np.zeros((len(strat), len(cats)))
        s2c = eric.strategy_to_category()
        for s, i in self._index.items():
            self._member[i, self._cat_index[s2c[s]]] = 1.0
        return self

    @staticmethod
    def _norm(X):
        X = np.asarray(X, dtype=float)
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    def channels(self, texts) -> np.ndarray:
        """(n_items, 73, 5) — one score matrix per channel."""
        strat = _strategies()
        Xd = self._norm(self._vec.transform(list(texts)).toarray())
        Xs = self._vec.transform(list(texts))
        n = len(texts)

        p_linear = np.zeros((n, len(strat)))
        for i, m in self._clf.items():
            p_linear[:, i] = m.predict_proba(Xs)[:, 1]

        sims = Xd @ self._Xtr.T                       # item x train
        s_knn = np.zeros((n, len(strat)))
        s_cat_item = np.zeros((n, len(self._cat_index)))
        k = min(self.knn_k, sims.shape[1]) if sims.shape[1] else 0
        for r in range(n):
            if not k:
                break
            idx = np.argsort(-sims[r])[:k]
            w = np.clip(sims[r][idx], 0, None)
            tot = w.sum() + 1e-9
            for j, wj in zip(idx, w):
                for c in self._ytr[j]:
                    if c in self._index:
                        s_knn[r, self._index[c]] += wj / tot
            s_cat_item[r] = (s_knn[r] @ self._member)
            m = s_cat_item[r].max() + 1e-9
            s_cat_item[r] /= m

        s_category = s_cat_item @ self._member.T
        s_definition = Xd @ self._D.T
        p_prior = np.tile(self._prior, (n, 1))

        return np.stack([p_linear, s_knn, p_prior, s_category, s_definition], axis=-1)


class FusionModel(Model):
    """Stacked blend of the five channels with a calibrated threshold.

    Blend weights and threshold come from inner k-fold inside the training set, so the
    threshold is never chosen on the data it is reported on.
    """
    name = "fusion"

    CHANNELS = ("p_linear", "s_knn", "p_prior", "s_category", "s_definition")
    CE_CHANNEL = "s_crossenc"

    def __init__(self, inner_folds: int = 3, min_codes: int = 1, max_codes: int | None = None,
                 use_cardinality: bool = False, seed: int = 0,
                 per_code_threshold: bool = False, shrink: float = 8.0,
                 crossencoder: bool = False, ce_kwargs: dict | None = None):
        self.inner_folds, self.min_codes, self.max_codes = inner_folds, min_codes, max_codes
        self.use_cardinality, self.seed = use_cardinality, seed
        # crossencoder: add the label-aware cross-encoder as a sixth channel. Refitted inside
        # every inner fold (see module docstring), so this costs inner_folds + 1 fine-tunes per
        # outer fold — hours, not seconds. Off by default.
        self.crossencoder, self.ce_kwargs = crossencoder, dict(ce_kwargs or {})
        self._ce = None
        # per_code_threshold: one cutoff per ERIC code instead of one global cutoff.
        # The ceiling analysis showed 24 of 64 codes are never once recovered under a single
        # global threshold, including codes appearing 16 and 26 times — the global cutoff is
        # set by the common codes and the rarer ones never clear it. `shrink` is the strength
        # of shrinkage toward the global threshold, in pseudo-observations, so a code seen
        # three times does not get a threshold fitted on three points.
        self.per_code_threshold, self.shrink = per_code_threshold, shrink

    @property
    def channel_names(self):
        return self.CHANNELS + ((self.CE_CHANNEL,) if self.crossencoder else ())

    def _fit_ce(self, texts, labels):
        """One cross-encoder fine-tune. Kept in a method so the caller can free it."""
        from label_crossencoder import LabelCrossEncoder
        return LabelCrossEncoder(seed=self.seed, **self.ce_kwargs).fit(texts, labels)

    @staticmethod
    def _release(ce):
        """Hand the GPU back before the next fine-tune allocates its own copy."""
        if ce is None:
            return
        try:
            import torch
            ce._m = None
            torch.cuda.empty_cache()
        except Exception:
            pass

    def fit(self, texts, labels):
        import random
        from sklearn.linear_model import LogisticRegression

        strat = _strategies()
        idx = {s: i for i, s in enumerate(strat)}
        texts, labels = list(texts), list(labels)
        n = len(texts)

        # ---- inner CV to produce honest meta-features
        order = list(range(n))
        random.Random(self.seed).shuffle(order)
        folds = [order[i::self.inner_folds] for i in range(self.inner_folds)]
        meta_X, meta_y = [], []
        for f in folds:
            tr = [i for i in order if i not in set(f)]
            if not tr or not f:
                continue
            sc = ChannelScorer().fit([texts[i] for i in tr], [labels[i] for i in tr])
            C = sc.channels([texts[i] for i in f])
            if self.crossencoder:
                ce = self._fit_ce([texts[i] for i in tr], [labels[i] for i in tr])
                S = ce.scores([texts[i] for i in f])          # (len(f), 73)
                C = np.concatenate([C, S[:, :, None]], axis=-1)
                self._release(ce)
            for r, i in enumerate(f):
                gold = set(labels[i])
                meta_X.append(C[r])
                meta_y.append(np.array([1 if s in gold else 0 for s in strat]))
        MX = np.concatenate(meta_X, axis=0)
        MY = np.concatenate(meta_y, axis=0)

        self._blender = LogisticRegression(max_iter=2000, class_weight="balanced")
        self._blender.fit(MX, MY)
        self.weights = dict(zip(self.channel_names, self._blender.coef_[0].round(3)))

        # ---- threshold that maximises micro-F1 on those held-out scores
        p = self._blender.predict_proba(MX)[:, 1].reshape(-1, len(strat))
        g = MY.reshape(-1, len(strat))
        best, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.01):
            pred = p >= t
            tp = float((pred & (g > 0)).sum())
            f1 = 2 * tp / max(pred.sum() + g.sum(), 1e-9)
            if f1 > best_f1:
                best, best_f1 = float(t), f1
        self.threshold, self.inner_f1 = round(best, 3), round(best_f1 * 100, 2)

        # ---- optional per-code thresholds, shrunk toward the global one
        self.code_thresholds = {}
        if self.per_code_threshold:
            for j, code in enumerate(strat):
                col_p, col_g = p[:, j], g[:, j]
                n_pos = int(col_g.sum())
                if n_pos == 0:
                    self.code_thresholds[code] = self.threshold
                    continue
                bt, bf = self.threshold, -1.0
                for t in np.arange(0.05, 0.96, 0.01):
                    pred = col_p >= t
                    tp = float((pred & (col_g > 0)).sum())
                    f1 = 2 * tp / max(pred.sum() + col_g.sum(), 1e-9)
                    if f1 > bf:
                        bt, bf = float(t), f1
                # shrink: with few positives, trust the global threshold
                w = n_pos / (n_pos + self.shrink)
                self.code_thresholds[code] = round(w * bt + (1 - w) * self.threshold, 3)

        # expected number of codes, for the cardinality variant
        self.expected_codes = int(round(np.mean([len(l) for l in labels]))) if labels else 10

        # ---- refit channels on all of train
        self._scorer = ChannelScorer().fit(texts, labels)
        if self.crossencoder:
            self._release(self._ce)
            self._ce = self._fit_ce(texts, labels)
        self._idx = idx
        return self

    def scores(self, texts) -> np.ndarray:
        C = self._scorer.channels(texts)
        if self.crossencoder:
            S = self._ce.scores(list(texts))
            C = np.concatenate([C, S[:, :, None]], axis=-1)
        flat = C.reshape(-1, C.shape[-1])
        return self._blender.predict_proba(flat)[:, 1].reshape(C.shape[0], C.shape[1])

    def predict(self, texts):
        strat = _strategies()
        P = self.scores(texts)
        out = []
        for row in P:
            if self.use_cardinality:
                k = self.expected_codes
                picked = [strat[i] for i in np.argsort(-row)[:k]]
            elif self.per_code_threshold:
                picked = [strat[i] for i in range(len(strat))
                          if row[i] >= self.code_thresholds.get(strat[i], self.threshold)]
            else:
                picked = [strat[i] for i in range(len(strat)) if row[i] >= self.threshold]
                if self.max_codes and len(picked) > self.max_codes:
                    picked = [strat[i] for i in np.argsort(-row)[: self.max_codes]]
            if len(picked) < self.min_codes:
                picked = [strat[i] for i in np.argsort(-row)[: self.min_codes]]
            out.append(picked)
        return out

    def config(self):
        ct = getattr(self, "code_thresholds", {}) or {}
        return {"inner_folds": self.inner_folds, "use_cardinality": self.use_cardinality,
                "per_code_threshold": self.per_code_threshold,
                "crossencoder": self.crossencoder,
                "channels": list(self.channel_names),
                "threshold": getattr(self, "threshold", None),
                "inner_f1": getattr(self, "inner_f1", None),
                "weights": getattr(self, "weights", None),
                "n_code_thresholds": len(ct),
                "threshold_range": [min(ct.values()), max(ct.values())] if ct else None}


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    from dataset import Dataset
    from experiments import compare, run

    # usage: python fusion.py [source] [--ce] [inner_folds]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_ce = "--ce" in sys.argv
    source = args[0] if args else "document_head"
    inner = int(args[1]) if len(args) > 1 else (2 if use_ce else 3)

    tag = "C8-fusion6" if use_ce else "C4-fusion"
    name = f"{tag}-{source}"
    r = run(name, lambda: FusionModel(inner_folds=inner, crossencoder=use_ce),
            source=source, ds=Dataset.load())
    print("  channels:", r["config"]["channels"])
    print("  channel weights:", r["config"]["weights"])
    print("  calibrated threshold:", r["config"]["threshold"],
          "(inner F1", r["config"]["inner_f1"], ")")
    if use_ce:
        print()
        compare("C4-fusion-dochead", name)
