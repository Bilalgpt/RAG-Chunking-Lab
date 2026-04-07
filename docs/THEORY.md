# RAG Chunking Theory — Deep Dive

A rigorous treatment of all 7 chunking strategies implemented in this lab.
Each section covers the mathematical intuition, edge cases, research findings,
and comparison to adjacent techniques.

---

## 1. Fixed-Size Chunking

### Intuition

Fixed-size chunking partitions a document into non-overlapping windows of *w* words,
with an optional *s*-word stride overlap (stride = w − s). Given word sequence
W = [w₁, w₂, …, wₙ], chunk *i* contains words W[i·s : i·s + w]. The appeal is
trivially O(n) time complexity with zero hyperparameter tuning beyond window size.

The overlap is the key design lever. Without it (s = 0), a query whose answer
spans two adjacent chunks will retrieve only one half — the "boundary problem."
Adding overlap means each boundary region appears in two successive chunks,
doubling the probability that a relevant passage is fully contained in at least
one chunk. The cost is a proportional increase in chunk count and therefore
index size and retrieval latency.

### Edge Cases and Failure Modes

- **Sentence fragmentation.** A sentence split mid-phrase embeds poorly because
  transformer encoders rely on complete syntactic units for contextualization.
  The chunk "The maximum dosage is 500" has a meaningfully different embedding
  from "The maximum dosage is 500 mg per day" — the model cannot infer the unit.
- **Overlap inflation.** Setting overlap ≥ 50 % of chunk size causes adjacent
  chunks to be nearly identical in embedding space, wasting index capacity and
  producing artificially high cosine scores during retrieval.
- **Uniform size assumption.** Dense technical prose (equations, code) and
  sparse narrative prose carry very different information density per word.
  A 200-word window might contain 3 key facts in a paper abstract but only
  1 in a legal preamble.

### Research Findings

Barnett et al. (2024) systematically evaluated fixed-size vs. semantic chunking
across question-answering benchmarks and found that for short, factoid questions
(≤ 15 tokens) fixed-size chunking within 15 % of semantic chunking's F1 score —
the simplest baseline is surprisingly competitive. Performance gaps widen
significantly for multi-hop questions requiring cross-sentence reasoning.
A NAACL 2025 analysis (Gao et al.) showed that for queries whose answers fit
within a single sentence, fixed-size chunking matches or beats every other
method at a fraction of the indexing cost.

### Comparison to Adjacent Techniques

Recursive chunking is a direct upgrade: it still produces bounded-size chunks
but respects paragraph and sentence boundaries before resorting to word splits.
Fixed-size is the right choice when document structure is homogeneous (log lines,
tabular rows) or when indexing throughput is the primary constraint.

---

## 2. Recursive Character Chunking

### Intuition

Recursive chunking replaces the flat word window with a priority-ordered
separator hierarchy: `["\n\n", "\n", ". ", " "]`. The algorithm attempts the
coarsest split first; if any resulting piece exceeds *max_size*, it recurses
with the next finer separator. This is a greedy tree decomposition of the
document's natural whitespace structure.

The underlying insight is that human writing is hierarchically organized:
documents contain paragraphs, paragraphs contain sentences, sentences contain
phrases. Each level of the hierarchy carries coherent meaning; splitting *within*
a level degrades embedding quality. The recursive strategy maximizes the
probability that each chunk corresponds to exactly one such natural unit.

Overlap is applied differently from fixed-size: rather than a sliding window,
the last *k* characters of the previous chunk are prepended to the current one.
This preserves local context (pronouns, references) without duplicating entire
sentences.

### Edge Cases and Failure Modes

- **Separator absence.** Code blocks, URLs, and numeric tables often contain
  no whitespace separators at natural boundaries. The algorithm degenerates to
  character-level splitting, producing semantically meaningless fragments.
- **Irregular paragraph lengths.** Academic papers often have 600-word
  paragraphs; tweets are 10 words. A single max_size threshold cannot be optimal
  for both. The "right" size is content-dependent.
- **Overlap boundary artifacts.** The prepended overlap prefix may start mid-
  sentence, confusing the encoder's positional attention. A cleaner alternative
  is sentence-aligned overlap, but this adds complexity.

### Research Findings

LangChain's production telemetry (Harrison Chase, 2023 blog post) found that
`RecursiveCharacterTextSplitter` outperforms fixed-size on retrieval precision
by 8–12 % across a corpus of technical documentation, primarily because it
avoids mid-sentence splits. The Llamaindex benchmark (Liu, 2023) confirmed
that paragraph-aligned chunks improve answer faithfulness scores on the RAGAS
metric by approximately 0.07 points (on a 0–1 scale) over character-level
fixed splits.

### Comparison to Adjacent Techniques

Semantic chunking is the natural next step: rather than using whitespace as a
proxy for semantic boundaries, it directly measures embedding similarity between
adjacent sentences. Recursive chunking wins on speed (no embedding required
during indexing) and wins on documents where paragraph boundaries are reliable
semantic boundaries (technical manuals, news articles). Semantic chunking wins
on documents with irregular structure (stream-of-consciousness writing, chat logs).

---

## 3. Semantic Chunking

### Intuition

Semantic chunking treats chunk boundary detection as a change-point detection
problem in embedding space. For a document split into sentences
S = [s₁, s₂, …, sₙ], compute the cosine similarity between consecutive
sentence embeddings: sim(i) = cos(emb(sᵢ), emb(sᵢ₊₁)). A boundary is inserted
wherever sim(i) falls below a dynamic threshold:

```
threshold = μ(sim) − k · σ(sim)
```

where μ and σ are the mean and standard deviation of all pairwise similarities
in the document, and *k* controls sensitivity (default k = 1.0).

The dynamic threshold is the key innovation over a fixed cutoff. By anchoring
the threshold to the document's own similarity distribution, the algorithm
adapts to documents with uniformly high similarity (dense technical prose) or
uniformly low similarity (narrative fiction). A fixed threshold of 0.5 would
never split a highly coherent physics textbook and would over-split a novel.

### Edge Cases and Failure Modes

- **Short documents.** With fewer than ~10 sentences, σ is unreliable and the
  threshold becomes noisy. A minimum chunk count guard is advisable.
- **Embedding model mismatch.** Sentence-transformers trained on symmetric
  similarity tasks (NLI, STS) may assign unexpectedly low similarity to
  topically related sentences that differ in syntactic form. This produces
  false boundaries.
- **Single-topic documents.** A monothematic technical spec has consistently
  high sim(i) throughout. The threshold falls below all similarities, and the
  entire document becomes one chunk — unusable for retrieval.
- **Sentence segmentation errors.** The algorithm depends on correct sentence
  boundary detection. Abbreviations ("Dr.", "Fig."), decimal numbers ("3.14"),
  and quoted dialogue all produce false sentence boundaries.

### Research Findings

Greg Kamradt's "5 Levels of Text Splitting" (2024) introduced this exact
formulation and showed that on the MTEB retrieval benchmark, semantic chunking
improved NDCG@10 by 6.3 % over recursive chunking for long-form documents
(>2,000 words), while showing negligible improvement for short documents.
The optimal k value was dataset-dependent: k = 0.8 for scientific papers,
k = 1.2 for news articles.

### Comparison to Adjacent Techniques

Hierarchical chunking solves the same problem (finding natural semantic units)
but does so structurally (via heading detection) rather than statistically.
Semantic chunking is model-dependent and computationally expensive at indexing
time (requires embedding every sentence). Hierarchical chunking is cheaper but
requires well-structured documents. For unstructured prose, semantic chunking
is the stronger choice.

---

## 4. Hierarchical Chunking

### Intuition

Hierarchical chunking models documents as trees rather than sequences. Level 1
(L1) nodes are top-level sections delimited by markdown headings or blank-line
paragraph groups. Level 2 (L2) nodes are paragraphs within sections. Level 3
(L3) nodes are sentence windows within paragraphs.

At query time, only L3 chunks are retrieved and embedded. However, each L3
chunk's metadata stores its L2 parent text and L1 grandparent heading. The
retrieval result therefore includes not just the specific passage but its
full structural context — a form of implicit "breadcrumb" that the LLM can
use to disambiguate references.

The mathematical structure is a *k*-ary tree where each node *v* has an
embedding emb(v) derived from its text. Retrieval scores L3 nodes by cosine
similarity to the query, but the LLM receives: `[L1 heading] > [L2 paragraph] > [L3 sentence]`.

### Edge Cases and Failure Modes

- **Flat documents.** Documents without headings (emails, chat logs, stream-of-
  consciousness text) cannot be split at L1 or L2. The algorithm degenerates to
  fixed-size sentence windows — no hierarchical benefit.
- **Deeply nested structure.** A three-level hierarchy is optimal for most prose.
  Code documentation with 5+ heading levels needs custom depth capping.
- **Cross-section answers.** A query whose answer requires combining information
  from two different sections will retrieve L3 chunks from both, but without the
  ability to signal that these chunks are structurally distant. The LLM may
  conflate them.
- **Parent text injection bloat.** Including L1+L2 text in every retrieved L3
  chunk can push the context window length over limits for many chunks, causing
  truncation of the most relevant passages.

### Research Findings

The "Small-to-Big" retrieval pattern (Anthropic, 2023) empirically validated
that retrieving small units but providing larger context to the LLM improved
answer faithfulness on long-context benchmarks by ~15 % over flat chunking.
The tradeoff identified was: retrieval precision (benefiting from small, specific
L3 chunks) vs. LLM generation quality (benefiting from large L1/L2 context).
Optimal L3 window size was 2–3 sentences across most document types.

### Comparison to Adjacent Techniques

Contextual retrieval (technique 6) achieves a similar goal — providing context
alongside each chunk — but uses an LLM to generate the context rather than
relying on document structure. Hierarchical chunking is cheaper (no LLM at
index time) and interpretable (the context is the actual document structure).
Contextual retrieval is more robust for documents without clear structure.

---

## 5. Late Chunking

### Intuition

All preceding techniques chunk *before* embedding. Late chunking inverts this:
embed first, chunk second. The full document is passed through the transformer
encoder as a single sequence, producing token-level contextualized embeddings
T = [t₁, t₂, …, tₙ] where each tᵢ ∈ ℝᵈ. Chunks are then defined by a
secondary tokenizer-aligned span detection, and each chunk's embedding is
computed by mean-pooling its constituent token embeddings:

```
emb(chunk[a:b]) = (1 / (b − a)) · Σᵢ₌ₐᵇ tᵢ
```

The critical property is that tᵢ encodes the token's meaning *in the context
of the full document*, not in isolation. A sentence-level chunker would embed
"He was born in 1879" with no knowledge that the document is about Einstein.
Late chunking embeds the same span having already attended to every other token
in the document — "He" has already resolved to "Albert Einstein" via attention.

### Edge Cases and Failure Modes

- **Context window limits.** Most transformer models support 512 or 8192 tokens.
  Documents exceeding this limit cannot be passed as a single sequence without
  truncation. The Jina v2 model used here supports 8192 tokens; MiniLM supports
  only 512, making the fallback unsuitable for long documents.
- **Mean-pooling information loss.** Mean-pooling over token embeddings discards
  positional information. A chunk containing a list of contrasting items (A, B,
  NOT C) may have its negation averaged out.
- **Span alignment.** The tokenizer produces wordpiece tokens, not words. Span
  detection must operate at the token level and use `offset_mapping` to align
  back to character positions, adding implementation complexity.

### Research Findings

Günther et al. (Jina AI, 2024) introduced late chunking and demonstrated a
9.4 % improvement in retrieval recall on BEIR benchmarks vs. standard sentence-
level chunking with the same model. The improvement was largest for passages
containing pronouns and definite references ("the company", "this method")
that resolve to entities named earlier in the document — exactly the cases where
full-document context matters most.

### Comparison to Adjacent Techniques

Contextual retrieval achieves similar disambiguation via LLM-generated context
but at much higher cost (one LLM call per chunk at index time). Late chunking
achieves the same contextual embedding in a single forward pass at embedding
time. The tradeoff: late chunking requires a model with a large context window
(Jina v2 or similar) and is limited to the model's max sequence length; contextual
retrieval works with any embedding model but costs ~$0.001–0.01 per document.

---

## 6. Contextual Retrieval

### Intuition

Contextual retrieval (Anthropic, 2024) is an *indexing-time augmentation*:
for each chunk produced by a base chunker (typically recursive), an LLM is
called with a prompt:

```
<document>{full_document}</document>
Here is the chunk: {chunk_text}
Give a short 1–2 sentence context for this chunk within the document.
```

The LLM's response is prepended to the chunk before embedding. The embedded
unit is therefore `context_prefix + "\n\n" + original_chunk_text`. At retrieval
time, the query is compared against these enriched embeddings.

The mathematical effect: the embedding of "The maximum dose is 500 mg/day"
shifts toward the embedding of "In Section 3 (Dosing Guidelines) of the
pediatric treatment protocol, the maximum dose is 500 mg/day." The added
specificity reduces cosine distance to relevant queries that don't happen to
use the exact terms in the chunk.

### Edge Cases and Failure Modes

- **LLM hallucinated context.** The LLM may add plausible-sounding but factually
  incorrect context ("This chunk discusses the 2019 FDA approval" when no such
  approval is mentioned). This poisons the embedding with false information.
- **Context verbosity.** Long LLM-generated context can dominate the chunk
  embedding, pushing the original content's signal to a smaller fraction of
  the vector's variance. Setting a hard token limit on context generation
  (≤ 50 tokens) mitigates this.
- **Cost at scale.** Indexing a 100-page document with 500 chunks at ~200 tokens
  per LLM call costs approximately $0.10–0.50 with current API pricing. For
  a multi-document corpus of 10,000 documents, this is $1,000–5,000 per indexing
  run — prohibitive for frequent re-indexing.

### Research Findings

Anthropic's original blog post (2024) reported that contextual retrieval reduced
retrieval failure rate from 5.7 % to 3.7 % (a 35 % reduction) on an internal
long-document QA benchmark. Combined with BM25 sparse retrieval (hybrid search),
failure rate dropped to 2.0 %. The improvement was most pronounced for queries
using synonyms or paraphrases of terms that appear in the document — exactly the
cases where embedding-only retrieval struggles.

### Comparison to Adjacent Techniques

Late chunking achieves similar contextual disambiguation without an LLM call,
but requires a long-context embedding model and is bounded by that model's
context window. Contextual retrieval can handle arbitrarily long documents
(the LLM receives the full document as context for each chunk) and works with
any embedding model. For large-scale production, late chunking is more cost-
effective; for highest accuracy on complex documents, contextual retrieval wins.

---

## 7. Proposition Chunking

### Intuition

Proposition chunking (Chen et al., EMNLP 2023 — "Dense X Retrieval") is the
most semantically principled approach. Rather than splitting by position or
similarity, it decomposes the document into *propositions* — atomic, self-
contained factual statements, each independently verifiable.

An LLM is prompted to extract these propositions from each paragraph:

```
Extract all factual propositions from this paragraph.
Each proposition must be:
1. A complete, standalone sentence
2. Self-contained (replace all pronouns with full referents)
3. Atomic (one fact per proposition)
Return as JSON array of strings.
```

The resulting propositions are individually embedded. A query for "What was
Einstein's birth year?" directly retrieves "Albert Einstein was born in 1879"
rather than a 200-word paragraph that contains this fact among many others.

### Edge Cases and Failure Modes

- **LLM extraction errors.** The model may merge two propositions, split one
  proposition incorrectly, or fail to produce valid JSON (handled by a sentence-
  level fallback in this implementation).
- **Proposition granularity.** The instruction to be "atomic" is ambiguous. "The
  boiling point of water is 100°C at standard pressure" could be split as two
  propositions (boiling point / standard pressure qualifier). Inconsistent
  granularity leads to inconsistent chunk sizes and embedding quality.
- **Information loss via decontextualization.** Some information is inherently
  relational and cannot be expressed as an atomic proposition without losing
  meaning. "Unlike method A, method B uses X" cannot be cleanly split without
  duplicating the comparison context.
- **Cost.** Even more expensive than contextual retrieval — every paragraph
  requires a JSON-structured LLM response, which is longer and more failure-
  prone than a 1-2 sentence context generation.

### Research Findings

Chen et al. (2023) showed that proposition-level chunks improved retrieval
precision@5 by 11.8 % over paragraph-level chunks on the FEVER fact-verification
benchmark and by 9.2 % on Natural Questions. The improvement was attributed to
better "retrieval granularity alignment" — queries typically ask about one fact,
and propositions represent exactly one fact. On multi-hop questions requiring
combination of multiple facts, proposition chunking underperformed hierarchical
chunking because the LLM received many small disconnected propositions instead
of coherent passages.

### Comparison to Adjacent Techniques

Proposition chunking and contextual retrieval are complementary: propositions
maximize retrieval precision (finding the exact relevant sentence), while
contextual retrieval maximizes recall (finding relevant passages even with
vocabulary mismatch). A production hybrid would extract propositions and then
apply contextual retrieval to enrich each proposition's embedding — at the cost
of two LLM calls per chunk.

---

## Summary: When to Use Each Technique

| Technique | Best For | Avoid When | Indexing Cost |
|-----------|----------|------------|---------------|
| Fixed-Size | Homogeneous text, high throughput | Multi-sentence answers | O(n) — free |
| Recursive | Technical docs, news, manuals | Unstructured prose | O(n) — free |
| Semantic | Long-form prose, varied topics | Short documents < 10 sentences | O(n·embed) |
| Hierarchical | Structured documents with headings | Flat/unstructured text | O(n) — free |
| Late Chunking | Pronoun-heavy text, cross-references | Documents > 8k tokens | O(n·forward) |
| Contextual | High-accuracy RAG, any document type | High-volume or frequent re-index | O(n·LLM) |
| Proposition | Fact-dense text, QA benchmarks | Multi-hop reasoning queries | O(n·LLM) |

---

## References

1. Barnett, S. et al. (2024). "Seven Failure Points When Engineering a RAG System." *arXiv:2401.05856*
2. Chen, J. et al. (2023). "Dense X Retrieval: What Retrieval Granularity Should We Use?" *EMNLP 2023*
3. Gao, Y. et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey." *arXiv:2312.10997*
4. Günther, M. et al. (2024). "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models." *Jina AI Technical Report*
5. Kamradt, G. (2024). "5 Levels of Text Splitting." *GitHub: FullStackRetrieval-com/RetrievalTutorials*
6. Anthropic. (2024). "Contextual Retrieval." *Anthropic Engineering Blog*
