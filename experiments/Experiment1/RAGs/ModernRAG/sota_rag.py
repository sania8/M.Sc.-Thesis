"""B3 — the modern RAG rung: what a competent engineer would build today, done properly.

The ladder in the method document has a hole where current practice belongs. Every number so
far is measured against components I wrote; "cheap methods win" is only a claim about the field
if a serious 2026 RAG stack is in the table, especially now that configuration has been shown
to matter roughly seven times more than representation.

So this is deliberately not a straw man. It has every part the 2025-26 literature recommends:

    hybrid retrieval      dense (multilingual-e5-large) fused with sparse lexical scores by
                          reciprocal rank fusion — the two arms fail differently
    multi-query           one query for the programme's identity, plus one per ERIC category,
                          so evidence for financial strategies and for training strategies is
                          fetched separately rather than competing in one pass
    cross-encoder rerank  bge-reranker-v2-m3 reads query and chunk together
    retrieved few-shot    the two nearest already-coded interventions, with their full gold row
    definitions in prompt all 73 strategies with Powell's definitions, grouped by category
    structured output     one JSON object per document: the whole spreadsheet row
    batched decoding      generate_many(), ~6x faster than one prompt at a time

It produces a complete row — name, summary, focus, level, maturity, ERIC codes — so it can be
scored on extraction, classification and ERIC by the same scorers as everything else, and lands
a composite directly comparable to the 57.9 of the current pipeline.
"""
from __future__ import annotations

import json
import re
import sys
import time

import numpy as np

import eric
from config import OUTPUTS

OUT = OUTPUTS / "sota_rag"

CATEGORY_QUERIES = {c: f"activities showing {c.lower()}" for c in eric.categories()}
IDENTITY_QUERY = "the name, objectives and main activities of the programme this document establishes"


def chunk(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return [c for c in out if c.strip()]


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal rank fusion — combines rankings without needing comparable scores."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


class SotaRag:
    def __init__(self, encoder="intfloat/multilingual-e5-large",
                 reranker="BAAI/bge-reranker-v2-m3", generator=None,
                 per_query: int = 3, rerank_to: int = 12, n_shots: int = 2,
                 max_context_chars: int = 5000):
        self.encoder_name, self.reranker_name, self.generator = encoder, reranker, generator
        self.per_query, self.rerank_to = per_query, rerank_to
        self.n_shots, self.max_context_chars = n_shots, max_context_chars
        self._enc = self._rr = None

    def encoder(self):
        if self._enc is None:
            from sentence_transformers import SentenceTransformer
            self._enc = SentenceTransformer(self.encoder_name, device="cuda")
            self._enc.max_seq_length = 256
        return self._enc

    def reranker(self):
        if self._rr is None:
            from sentence_transformers import CrossEncoder
            self._rr = CrossEncoder(self.reranker_name, device="cuda", max_length=384)
        return self._rr

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, doc) -> list[str]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        chunks = chunk(doc.text)[:400]
        if len(chunks) <= self.rerank_to:
            return chunks

        queries = [IDENTITY_QUERY] + list(CATEGORY_QUERIES.values())
        m = self.encoder()
        V = np.asarray(m.encode(["passage: " + c for c in chunks], normalize_embeddings=True,
                                batch_size=64, show_progress_bar=False))
        Q = np.asarray(m.encode(["query: " + q for q in queries], normalize_embeddings=True,
                                batch_size=16, show_progress_bar=False))
        dense = V @ Q.T                                        # chunks x queries

        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        try:
            X = vec.fit_transform(chunks)
            Xq = vec.transform(queries)
            sparse = (X @ Xq.T).toarray()
        except ValueError:
            sparse = np.zeros_like(dense)

        rankings = []
        for j in range(len(queries)):
            rankings.append(list(np.argsort(-dense[:, j])[: self.per_query]))
            rankings.append(list(np.argsort(-sparse[:, j])[: self.per_query]))
        fused = rrf(rankings)
        cand_idx = [i for i, _ in sorted(fused.items(), key=lambda kv: -kv[1])][:40]

        pairs = [(IDENTITY_QUERY, chunks[i]) for i in cand_idx]
        scores = self.reranker().predict(pairs, show_progress_bar=False, batch_size=32)
        order = [cand_idx[i] for i in np.argsort(-np.asarray(scores))[: self.rerank_to]]
        return [chunks[i] for i in sorted(order)]              # restore document order

    # ------------------------------------------------------------------ prompt
    def shots(self, doc, train_rows, ds) -> str:
        if not train_rows:
            return ""
        m = self.encoder()
        head = "query: " + doc.head(1500)
        texts = ["passage: " + ds.text(iv, "name+summary") for iv in train_rows]
        V = np.asarray(m.encode(texts, normalize_embeddings=True, batch_size=64,
                                show_progress_bar=False))
        q = np.asarray(m.encode([head], normalize_embeddings=True, show_progress_bar=False))[0]
        idx = np.argsort(-(V @ q))[: self.n_shots]
        out = []
        for i in idx:
            iv = train_rows[int(i)]
            out.append(json.dumps({"name": iv.name, "summary": (iv.summary or "")[:300],
                                   "focus": iv.focus, "level": iv.level,
                                   "maturity": iv.maturity,
                                   "eric_codes": iv.eric_codes[:12]}, ensure_ascii=False))
        return "\n\n".join(f"EXAMPLE {n+1}:\n{o}" for n, o in enumerate(out))

    def prompt(self, doc, context: list[str], shots: str) -> str:
        # V100 has no flash-attention kernel, so SDPA falls back to the math backend and
        # materialises the full attention matrix: a ~8k-token prompt needs 4.3 GB and OOMs a
        # 32 GB card. The 73 definitions cost 13.4k characters on their own, so the taxonomy
        # goes in by name and the definitions reach the model through the retrieved exemplars.
        taxonomy = eric.prompt_block_flat()
        ctx = "\n---\n".join(context)[: self.max_context_chars]
        return (
            f"You are a policy analyst coding a national health policy document from "
            f"{doc.country} into a structured database.\n\n"
            f"THE 73 ERIC STRATEGIES (choose only from this numbered list):\n{taxonomy}\n\n"
            f"{shots}\n\n"
            f"RETRIEVED PASSAGES FROM THE DOCUMENT:\n{ctx}\n\n"
            "Record the main intervention this document establishes.\n\n"
            "Field instructions:\n"
            "- name: the official title of the programme, in English\n"
            "- summary: 2-3 sentences on objectives, who runs it, and main activities\n"
            "- focus: exactly ONE of: AMR, AMU, IPC, Antimicrobial Stewardship, "
            "Public Education & Awareness Campaign\n"
            "- level: exactly ONE of: Macro, Meso, Micro\n"
            "- maturity: exactly ONE of: Developed, Implemented, Evaluated\n"
            "- eric_codes: a list of 8 to 12 strategy names copied EXACTLY from the "
            "numbered list above. Use ONLY the strategy names, never the category "
            "names shown in brackets. Never write placeholder text.\n\n"
            "Reply with a single JSON object using those six keys and real values.\nJSON:")

    # ------------------------------------------------------------------ run
    def run(self, ds, docs, train_rows, batch_size: int = 1):
        prompts, keep = [], []
        for doc in docs:
            ctx = self.retrieve(doc)
            prompts.append(self.prompt(doc, ctx, self.shots(doc, train_rows, ds)))
            keep.append(doc)
        replies = self.generator.generate_many(prompts, batch_size=batch_size)
        out = {}
        for doc, reply in zip(keep, replies):
            m = re.search(r"\{.*\}", reply, re.S)
            rec = {"name": "", "summary": "", "eric_codes": [], "raw": reply[:300]}
            if m:
                try:
                    obj = json.loads(m.group(0))
                    codes = [eric.canonical(c) for c in obj.get("eric_codes", [])]
                    rec = {"name": str(obj.get("name", "")).strip(),
                           "summary": str(obj.get("summary", "")).strip(),
                           "focus": obj.get("focus"), "level": obj.get("level"),
                           "maturity": obj.get("maturity"),
                           "eric_codes": [c for c in codes if c]}
                except json.JSONDecodeError:
                    pass
            out[doc.doc_id] = rec
        return out


def main(limit=None, batch_size=1):
    from dataset import Dataset
    from summarise import LocalGenerator

    ds = Dataset.load()
    gen = LocalGenerator(model="Qwen/Qwen2.5-7B-Instruct", max_new_tokens=600).load()
    rag = SotaRag(generator=gen)
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for fold, (train, test) in enumerate(ds.folds(5)):
        docs = []
        for iv in test:
            for d in iv.doc_ids:
                doc = ds.documents.get(d)
                if doc and doc.usable and doc not in docs:
                    docs.append(doc)
        if limit:
            docs = docs[:limit]
        # resume: a run killed by OOM should cost the rows it lost, not the rows it finished
        done = sum(1 for d in docs if (OUT / f"{d.doc_id}.json").exists())
        docs = [d for d in docs if not (OUT / f"{d.doc_id}.json").exists()]
        print(f"fold {fold}: {len(docs)} to do, {done} already written", flush=True)
        if not docs:
            continue
        res = rag.run(ds, docs, train, batch_size=batch_size)
        for k, v in res.items():
            (OUT / f"{k}.json").write_text(json.dumps(v, ensure_ascii=False))
        print(f"   done {time.time()-t0:.0f}s", flush=True)
        if limit:
            break
    return OUT


def score():
    """Extraction, classification and ERIC for the stored B3 rows, then the composite."""
    import evaluation as ev
    from dataset import Dataset
    from extraction import composite, gold_by_group, match, score_extraction

    ds = Dataset.load()
    rows = {f.stem: json.loads(f.read_text()) for f in OUT.glob("*.json")}
    preds = {}
    for iv in ds.interventions():
        acc = preds.setdefault(iv.group_id, [])
        for d in iv.doc_ids:
            n = rows.get(d, {}).get("name", "")
            if n and n not in acc:
                acc.append(n)
    ext = score_extraction(preds, gold_by_group(ds))

    by_group = {}
    for iv in ds.interventions():
        by_group.setdefault(iv.group_id, []).append(iv)
    matched = set()
    for gid, ivs in by_group.items():
        pairs, _, _ = match(preds.get(gid, []), [iv.name for iv in ivs])
        for _, j, _ in pairs:
            matched.add((gid, ivs[j].no))

    pred_sets, gold_sets = [], []
    for iv in ds.interventions():
        codes = []
        if (iv.group_id, iv.no) in matched:
            for d in iv.doc_ids:
                codes += rows.get(d, {}).get("eric_codes", [])
        pred_sets.append(sorted(set(codes)))
        gold_sets.append(iv.eric_codes)
    eric_f1 = 100 * ev.micro_f1(pred_sets, gold_sets).f1

    import classify
    fields = {}
    for field in ("focus", "level", "maturity"):
        P, G = [], []
        for iv in ds.interventions():
            if (iv.group_id, iv.no) not in matched:
                continue
            v = None
            for d in iv.doc_ids:
                v = rows.get(d, {}).get(field) or v
            P.append(classify.normalise(field, v))
            G.append(classify.normalise(field, getattr(iv, field)))
        s = classify.score_field(P, G, field in classify.SET_VALUED)
        fields[field] = s.get("micro_f1", s.get("accuracy", 0.0))
    classification = float(np.mean(list(fields.values())))
    return {"extraction_f1": ext["f1"], "split_ratio": ext["split_ratio"],
            "classification": classification, "fields": fields, "eric_f1": eric_f1,
            "composite": composite(ext["f1"], classification, eric_f1)}


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        r = score()
        print(json.dumps(r, indent=2))
        (OUTPUTS / "sota_rag_score.json").write_text(json.dumps(r, indent=2))
    else:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
        main(limit=limit)
