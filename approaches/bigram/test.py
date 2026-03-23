"""
text_segmenter.py
=================
Language-agnostic word segmenter using a bigram language model.

Pipeline
--------
1. Train  – learn vocabulary and bigram statistics from raw sentences.
2. Segment – given a spaceless string, recover every valid word sequence
             using depth-first backtracking over the vocabulary.
3. Rank   – score each candidate with Laplace-smoothed bigram log-probability
             and return the top-k results.

Works for any language by simply swapping the training corpus.
No LLMs, no external dependencies – pure Python stdlib.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Generator, Iterator, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
#  Core model
# ═══════════════════════════════════════════════════════════════════════════════

class TextSegmenter:
    """
    Bigram language model that segments continuous (spaceless) text.

    Attributes
    ----------
    vocab               : All words seen during training.
    unigram_counts      : Raw frequency of each word.
    bigram_counts       : Conditional frequency table: bigram_counts[w1][w2].
    sentence_start_counts: How often each word begins a training sentence.
    num_sentences       : Total training sentences consumed.
    total_words         : Total training tokens consumed.
    """

    def __init__(self, max_word_len: Optional[int] = None) -> None:
        self.vocab: set[str] = set()
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
        self.sentence_start_counts: Counter[str] = Counter()
        self.num_sentences: int = 0
        self.total_words: int = 0
        self._user_max_word_len: Optional[int] = max_word_len
        self._effective_max_word_len: int = max_word_len or 20

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, sentences: Iterator[str]) -> "TextSegmenter":
        """
        Ingest raw sentences and build the language model.

        Can be called multiple times to extend the model incrementally.

        Parameters
        ----------
        sentences : Any iterable of raw sentences (strings).
        """
        for raw in sentences:
            words = raw.lower().strip().split()
            if not words:
                continue

            self.num_sentences += 1
            self.sentence_start_counts[words[0]] += 1

            for word in words:
                self.vocab.add(word)
                self.unigram_counts[word] += 1
                self.total_words += 1

            # Record every adjacent pair
            for i in range(len(words) - 1):
                self.bigram_counts[words[i]][words[i + 1]] += 1

        # Auto-detect maximum word length from vocabulary (skip if user fixed it)
        if self._user_max_word_len is None and self.vocab:
            self._effective_max_word_len = max(len(w) for w in self.vocab)

        return self  # fluent API

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _log_prob(self, words: List[str]) -> float:
        """
        Log-probability of a word sequence under a Laplace-smoothed bigram model.

                P(w1 … wn) ≈ P_start(w1) × ∏ P(wᵢ | wᵢ₋₁)

        Laplace (add-1) smoothing prevents zero-probability for unseen bigrams,
        which is essential when the corpus is small.

        Longer sequences are inherently penalised (more multiplications < 1),
        so the model naturally prefers fewer, more frequent words — matching
        the way n-gram language models work in NLP.
        """
        if not words:
            return float("-inf")

        V = len(self.vocab)  # vocabulary size (smoothing denominator term)
        log_p = 0.0

        # ── Sentence-initial probability ──────────────────────────────────────
        # P(w₁) = ( count(w₁ starts sentence) + 1 ) / ( total sentences + V )
        p_start = (self.sentence_start_counts[words[0]] + 1) / (self.num_sentences + V)
        log_p += math.log(p_start)

        # ── Bigram chain ──────────────────────────────────────────────────────
        # P(wᵢ | wᵢ₋₁) = ( count(wᵢ₋₁, wᵢ) + 1 ) / ( count(wᵢ₋₁) + V )
        for i in range(len(words) - 1):
            prev, curr = words[i], words[i + 1]
            context_total = self.unigram_counts[prev]
            bigram_freq   = self.bigram_counts[prev][curr]
            p_bigram = (bigram_freq + 1) / (context_total + V)
            log_p += math.log(p_bigram)

        return log_p

    # ── Candidate enumeration ─────────────────────────────────────────────────

    def _iter_segmentations(
        self,
        text: str,
        max_paths: int,
    ) -> Generator[List[str], None, None]:
        """
        Depth-first backtracking over the vocabulary to yield every valid
        segmentation of *text*.

        The search is bounded by *max_paths* to prevent combinatorial explosion
        on ambiguous long strings.

        Complexity
        ----------
        Worst-case O(V × n) per level where n = len(text). The *max_word_len*
        cap ensures each DFS node fans out at most O(L) times (L = longest
        vocab word), keeping practical runtime manageable.
        """
        n = len(text)
        L = self._effective_max_word_len
        found_count = 0
        seen: set[tuple] = set()

        # Use an explicit stack to avoid Python recursion-limit issues
        # Stack item: (position, path_so_far_as_tuple)
        stack: list[tuple[int, tuple]] = [(0, ())]

        while stack:
            if found_count >= max_paths:
                return

            pos, path = stack.pop()

            if pos == n:
                if path not in seen:
                    seen.add(path)
                    found_count += 1
                    yield list(path)
                continue

            # Try every substring starting at pos up to max word length
            # Reversed so the stack pops shorter words last (natural order)
            for end in range(min(n, pos + L), pos, -1):
                word = text[pos:end]
                if word in self.vocab:
                    stack.append((end, path + (word,)))

    # ── Public interface ──────────────────────────────────────────────────────

    def segment(
        self,
        text: str,
        top_k: int = 10,
        max_paths: int = 50_000,
    ) -> List[Tuple[str, float]]:
        """
        Return the *top_k* most probable segmentations of *text*.

        Parameters
        ----------
        text      : Spaceless input string (case-insensitive).
        top_k     : How many results to return.
        max_paths : Hard cap on DFS paths explored (safety valve).

        Returns
        -------
        List of ``(segmented_string, log_probability)`` tuples, best first.
        An empty list means no valid segmentation was found in the vocabulary.
        """
        if not self.vocab:
            raise RuntimeError("Model has not been trained yet. Call .train() first.")

        text = text.lower().strip()
        if not text:
            return []

        candidates = list(self._iter_segmentations(text, max_paths))

        if not candidates:
            return []

        # Score, sort descending, truncate
        scored = sorted(
            ((seg, self._log_prob(seg)) for seg in candidates),
            key=lambda x: x[1],
            reverse=True,
        )

        return [(" ".join(seg), round(lp, 6)) for seg, lp in scored[:top_k]]


# ═══════════════════════════════════════════════════════════════════════════════
#  Pretty-printer helper
# ═══════════════════════════════════════════════════════════════════════════════

def display_results(
    results: List[Tuple[str, float]],
    input_text: str,
    show_scores: bool = True,
) -> None:
    """Print segmentation results in a readable table."""
    print(f"\n{'─' * 60}")
    print(f"  Input : {input_text}")
    print(f"  Found : {len(results)} segmentation(s)")
    print(f"{'─' * 60}")
    if not results:
        print("  ✗ No valid segmentation found in vocabulary.\n")
        return
    for rank, (sentence, score) in enumerate(results, 1):
        score_str = f"  (log-prob: {score:>10.4f})" if show_scores else ""
        print(f"  {rank:>2}. {sentence}{score_str}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════════════════════

RAW_DATA_PATH = "../../raw_sentences.txt"


def main() -> None:
    # ── Build & train ─────────────────────────────────────────────────────────
    sentences = []
    with open(RAW_DATA_PATH, 'r', encoding='utf-8') as file:
        sentences = [s.strip() for s in file.readlines()]
    model = TextSegmenter()
    model.train(sentences)

    print("=" * 60)
    print("  Text Segmenter  —  Bigram Language Model")
    print("=" * 60)
    print(f"\n  Training sentences : {model.num_sentences}")
    print(f"  Vocabulary size    : {len(model.vocab)}")
    print(f"  Max word length    : {model._effective_max_word_len}")
    print(f"\n  Vocabulary         : {sorted(model.vocab)}")

    # ── Primary example from the problem statement ────────────────────────────
    query = "peoplethinkthatgodisnowhere"
    results = model.segment(query, top_k=10)
    display_results(results, query)

    # ── Additional stress tests ───────────────────────────────────────────────
    extra_queries = [
        "thinkthatisnow",
        "godishere",
        "whereistherightway",
        "iknowwherethegodis",
        "itisnoreasontoworshiphere",
        "anyoneknowsthatartifactisnowhere"
    ]
    for q in extra_queries:
        res = model.segment(q, top_k=5)
        display_results(res, q)


if __name__ == "__main__":
    main()