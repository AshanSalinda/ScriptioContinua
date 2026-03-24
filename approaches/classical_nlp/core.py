"""
the_linguist.py
===============
Epigrascan — The Linguist
Core Word-Segmentation Module  ·  v1.0

A Decision Support System for deciphering ancient spaceless Brahmi
inscriptions from Sri Lanka (Early Brahmi / Prakrit / Old Sinhalese).

Architecture
────────────
  ┌──────────────────────────────────────────────────────────────────┐
  │                       T H E   L I N G U I S T                    │
  │                                                                  │
  │  TRAINING PHASE                                                  │
  │  ├─ LexiconIndex     Trie dictionary — O(L) prefix look-ups      │
  │  ├─ CharNgramModel   Character trigram model — OOV plausibility  │
  │  ├─ BigramLM         Word bigram LM — grammatical flow           │
  │  └─ MorphologyModel  Suffix / prefix patterns — grammar guard    │
  │                                                                  │
  │  SEGMENTATION PHASE                                              │
  │  ├─ CompositeScorer  Weighted fusion of all model signals        │
  │  └─ BeamSearch       Hypothesis lattice search with OOV support  │
  │                                                                  │
  │  OUTPUT PHASE                                                    │
  │  └─ SegmentedResult  Ranked, annotated results for epigraphers   │
  └──────────────────────────────────────────────────────────────────┘

Key Design Decisions
────────────────────
1. Beam Search (not DFS) — scales to long inscription fragments.
2. Character N-gram OOV scoring — handles undiscovered words, city
   names, and royal epithets that don't appear in any known lexicon.
3. Morphology model — learns suffix/prefix patterns from data, acts
   as a structural grammar guard without hard-coded rules.
4. Missing-text handling — common epigraphic gap markers ([…], …, ?)
   are stripped from training data automatically; only intact fragments
   are used for statistical learning.
5. Language-agnostic — changing the training data is all that is
   required to apply the system to any ancient script.

Composite Step-Score Formula (log-space)
─────────────────────────────────────────
  score(word, prev_word) =
      w_bigram  × log P_bigram(word | prev)     word transition
    + w_char    × log P_char(word)              character plausibility
    + w_lexicon × lexicon_bonus(word)           known-word reward
    + w_morph   × morph_score(word)             suffix/prefix validity
    + w_length  × log P_length(len(word))       word size realism
    [+ oov_penalty  if word is out-of-vocabulary]

Usage
─────
  linguist = TheLinguist()
  linguist.train(dictionary=word_list, corpus=sentence_list)
  results  = linguist.segment("rajaputhosadhamitena", top_k=5)
  linguist.display(results, input_text="rajaputhosadhamitena")
"""

from __future__ import annotations

import math
import re
import heapq
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import (
    Dict, Generator, List, Optional,
    Set, Tuple
)

# ── Sentinel tokens (single private-use Unicode chars) ────────────────────────
_BOS = "\x02"   # Beginning Of Sentence  (word-bigram context)
_BOW = "\x04"   # Beginning Of Word      (character n-gram padding)
_EOW = "\x05"   # End Of Word            (character n-gram padding)

# ── Missing-text markers found in epigraphic publications ─────────────────────
#   Matches:  [...]  (...)  [?]  [*]  [anytext]  (anytext)  ..  ...  ?  *
_MISSING_RE = re.compile(
    r'\[\.+\]'
    r'|\(\.+\)'
    r'|\[[\?\*]+\]'
    r'|\[.*?\]'
    r'|\(.*?\)'
    r'|\.{2,}'
    r'|\?+'
    r'|\*+'
)


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LinguistConfig:
    """
    All hyper-parameters in one place.  Sensible defaults work for
    Early Brahmi / Prakrit; adjust weights when fine-tuning on a
    larger corpus.

    Scoring weights are multipliers applied to log-probabilities.
    Positive = reward, negative = penalty.
    """
    # ── Search ────────────────────────────────────────────────────────────────
    beam_width:   int            = 300    # Max hypotheses kept per text position
    max_results:  int            = 10     # Top-K final results returned
    min_word_len: int            = 1      # Minimum characters per word segment
    max_word_len: Optional[int]  = None   # Auto-set from corpus if None

    # ── OOV (Out-Of-Vocabulary) control ───────────────────────────────────────
    oov_enabled:       bool  = True    # Allow hypotheses for undiscovered words
    oov_penalty:       float = -3.5    # Log-penalty applied per OOV word
    oov_min_len:       int   = 2       # Minimum length for an OOV hypothesis
    oov_char_threshold: float = -10.0  # Minimum char n-gram score to accept

    # ── Composite score weights ───────────────────────────────────────────────
    w_bigram:   float = 1.0   # Word bigram language model
    w_char:     float = 0.7   # Character n-gram plausibility
    w_lexicon:  float = 3.0   # Known-word membership bonus (reward/penalty)
    w_morph:    float = 0.5   # Morphological suffix/prefix validity
    w_length:   float = 0.3   # biases toward realistic word lengths

    # ── Morphology ────────────────────────────────────────────────────────────
    suffix_len: int = 4   # Max suffix length to track
    prefix_len: int = 3   # Max prefix length to track

    # ── Character n-gram ──────────────────────────────────────────────────────
    char_ngram_order: int = 3   # Trigram (recommended for most scripts)


# ══════════════════════════════════════════════════════════════════════════════
#  Output Data Structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WordAnnotation:
    """
    Rich metadata for a single word within a segmented hypothesis.
    Gives epigraphers the evidence behind each decision.
    """
    word:        str
    is_oov:      bool    # True = not seen in dictionary or corpus
    char_score:  float   # Average per-character log-prob (higher = more word-like)
    morph_score: float   # Morphological validity (suffix + prefix pattern score)
    length_score: float
    step_score:  float   # Total composite score contribution of this word

    @property
    def status_label(self) -> str:
        return "novel word" if self.is_oov else "known word"

    @property
    def plausibility_pct(self) -> int:
        """
        Maps char_score from typical range [-12, 0] → [0, 100] for display.
        A score near 0 means the character sequence is very word-like.
        """
        return max(0, min(100, int((1.0 - abs(self.char_score) / 12.0) * 100)))


@dataclass
class SegmentedResult:
    """
    A single segmentation hypothesis, fully annotated for epigrapher review.

    Carries:
    · The segmented text with spaces
    · A total composite score (higher = more likely)
    · Per-word annotations (known/novel, plausibility, scores)
    · A human-readable confidence rating
    """
    rank:        int
    words:       List[str]
    total_score: float
    annotations: List[WordAnnotation]

    @property
    def text(self) -> str:
        return " ".join(self.words)

    @property
    def oov_words(self) -> List[str]:
        return [a.word for a in self.annotations if a.is_oov]

    @property
    def known_words(self) -> List[str]:
        return [a.word for a in self.annotations if not a.is_oov]

    @property
    def oov_count(self) -> int:
        return len(self.oov_words)

    @property
    def oov_ratio(self) -> float:
        return self.oov_count / max(len(self.words), 1)

    @property
    def confidence_stars(self) -> str:
        """5-star rating based on what fraction of words are novel."""
        r = self.oov_ratio
        if r == 0.0:   return "★★★★★"
        if r <= 0.15:  return "★★★★☆"
        if r <= 0.35:  return "★★★☆☆"
        if r <= 0.55:  return "★★☆☆☆"
        return "★☆☆☆☆"

    @property
    def confidence_label(self) -> str:
        r = self.oov_ratio
        if r == 0.0:   return "High — all words attested"
        if r <= 0.15:  return "Good — mostly attested"
        if r <= 0.35:  return "Moderate — some novel forms"
        if r <= 0.55:  return "Low — many novel forms"
        return "Very low — mostly novel"


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-model 1 — Lexicon Index  (Character Trie)
# ══════════════════════════════════════════════════════════════════════════════

class _TrieNode:
    """Single node in the character trie."""
    __slots__ = ("children", "is_end", "word", "freq")

    def __init__(self) -> None:
        self.children: Dict[str, "_TrieNode"] = {}
        self.is_end:   bool = False
        self.word:     Optional[str] = None
        self.freq:     int = 0


class LexiconIndex:
    """
    Character-trie dictionary with frequency statistics.

    Why a trie?
    -----------
    At each position `p` in the input text we need every vocabulary
    word that starts at `p`.  A trie traversal gives this in O(L) where
    L is the length of the longest matching word — far faster than
    scanning the full vocabulary at every position.
    """

    def __init__(self) -> None:
        self._root:   _TrieNode  = _TrieNode()
        self._total:  int        = 0
        self._vocab:  Set[str]   = set()
        self._counts: Counter[str] = Counter()

    # ── Mutation ──────────────────────────────────────────────────────────────

    def insert(self, word: str, freq: int = 1) -> None:
        """Add `word` (or increment its frequency) in the trie."""
        node = self._root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
            node = node.children[ch]
        if not node.is_end:
            self._vocab.add(word)
        node.is_end  = True
        node.word    = word
        node.freq   += freq
        self._counts[word] += freq
        self._total += freq

    # ── Queries ───────────────────────────────────────────────────────────────

    def words_starting_at(
        self,
        text:    str,
        pos:     int,
        max_len: Optional[int] = None,
    ) -> Generator[Tuple[str, int], None, None]:
        """
        Yield ``(word, frequency)`` for every vocabulary entry found as a
        prefix of ``text[pos:]``.  Stops as soon as the trie has no branch
        for the next character — no wasted scanning.
        """
        node = self._root
        stop = len(text) if max_len is None else min(len(text), pos + max_len)
        for i in range(pos, stop):
            ch = text[i]
            if ch not in node.children:
                return
            node = node.children[ch]
            if node.is_end:
                yield (node.word, node.freq)   # type: ignore[arg-type]

    def contains(self, word: str) -> bool:
        return word in self._vocab

    def freq_of(self, word: str) -> int:
        return self._counts.get(word, 0)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab(self) -> Set[str]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def total_tokens(self) -> int:
        return self._total


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-model 2 — Character N-gram Model  (OOV Plausibility)
# ══════════════════════════════════════════════════════════════════════════════

class CharNgramModel:
    """
    Laplace-smoothed character n-gram model.

    Purpose
    -------
    Ancient inscriptions will inevitably contain words that do not appear
    in any modern dictionary — names of forgotten kings, undiscovered
    village names, rare epithets.  A character n-gram model captures the
    *phonological fingerprint* of the language: which character sequences
    are plausible (e.g. "ssa", "ena", "aya" in Prakrit) versus which are
    not (e.g. "xkq").

    Training
    --------
    Every known word from the dictionary and corpus is broken into
    character trigrams (padded with BOW/EOW sentinels).  The model
    learns the frequency of each trigram and its two-character context.

    Scoring
    -------
    A candidate substring is scored by the average per-character
    log-probability: log P(w) / len(w).  Averaging normalises for length
    so short and long words are on a comparable scale.
    """

    def __init__(self, order: int = 3) -> None:
        self.order  = order
        self._ngrams: Counter[Tuple[str, ...]] = Counter()
        self._ctxs:   Counter[Tuple[str, ...]] = Counter()
        self._chars:  Set[str] = set()

    def _pad(self, word: str) -> List[str]:
        """Pad word with BOW/EOW sentinels for boundary-aware n-grams."""
        pad = [_BOW] * (self.order - 1)
        return pad + list(word) + [_EOW]

    def train_word(self, word: str) -> None:
        """Ingest one word into the character n-gram tables."""
        chars = self._pad(word)
        self._chars.update(chars)
        n = self.order
        for i in range(len(chars) - n + 1):
            ng  = tuple(chars[i: i + n])
            ctx = ng[:-1]
            self._ngrams[ng]  += 1
            self._ctxs[ctx]   += 1

    def log_prob(self, word: str) -> float:
        """
        Average per-character log-probability of ``word``.

        Returns a value ≤ 0.  Values close to 0 indicate the character
        sequence strongly resembles words in the training language.
        """
        chars = self._pad(word)
        n     = self.order
        V     = len(self._chars) + 1    # Laplace smoothing vocabulary size
        total = 0.0
        steps = 0
        for i in range(len(chars) - n + 1):
            ng  = tuple(chars[i: i + n])
            ctx = ng[:-1]
            cnt = self._ngrams.get(ng,  0) + 1
            den = self._ctxs.get(ctx,   0) + V
            total += math.log(cnt / den)
            steps += 1
        return total / max(steps, 1)

    @property
    def char_vocab_size(self) -> int:
        return len(self._chars)


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-model 3 — Word Bigram Language Model
# ══════════════════════════════════════════════════════════════════════════════

class BigramLM:
    """
    Laplace-smoothed word bigram language model.

    Captures which word pairs co-occur in the training corpus.  In Prakrit,
    certain function words (ca, pi, na, va) almost always follow nouns or
    verbs — this signal guides the beam search toward grammatically coherent
    word sequences even for unseen input strings.

    Sentence boundaries are modelled explicitly via a _BOS sentinel, so the
    model also learns which words typically begin an inscription.
    """

    def __init__(self) -> None:
        self._unigrams: Counter[str]                    = Counter()
        self._bigrams:  defaultdict[str, Counter[str]]  = defaultdict(Counter)
        self._n_sent:   int = 0
        self._vocab:    Set[str] = set()

    def train_sentence(self, words: List[str]) -> None:
        if not words:
            return
        self._n_sent += 1
        tokens = [_BOS] + words
        for w in words:
            self._unigrams[w] += 1
            self._vocab.add(w)
        for i in range(len(tokens) - 1):
            self._bigrams[tokens[i]][tokens[i + 1]] += 1

    def log_prob_next(self, prev: str, word: str) -> float:
        """
        Laplace-smoothed log P(word | prev).
        `prev` should be _BOS at the start of a segmentation.
        """
        V   = len(self._vocab) + 1
        cnt = self._bigrams[prev].get(word, 0) + 1
        # For BOS context, denominator = number of sentences seen
        den = (self._n_sent if prev == _BOS else self._unigrams.get(prev, 0)) + V
        return math.log(cnt / den)

    @property
    def vocab(self) -> Set[str]:
        return self._vocab

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-model 4 — Morphology Model  (Grammar Guard)
# ══════════════════════════════════════════════════════════════════════════════

class MorphologyModel:
    """
    Learns and scores morphological boundary patterns.

    Motivation
    ----------
    In Early Brahmi Prakrit, grammatical morphemes appear at consistent
    word boundaries.  Case endings such as -assa, -aya, -ena, -aṃ and
    verb endings like -ti, -nti recur across thousands of words.  By
    learning which sub-strings *end* and *start* words — purely from
    data — the model can flag a proposed segmentation as structurally
    implausible even if it hasn't seen the exact word before.

    This acts as the "grammar guard" required by the project specification:
    it rewards segmentations that produce words with historically attested
    morphological shapes and penalises those that don't.
    """

    def __init__(self, suffix_len: int = 4, prefix_len: int = 3) -> None:
        self.suffix_len = suffix_len
        self.prefix_len = prefix_len
        self._suffix_counts:  Counter[str] = Counter()
        self._prefix_counts:  Counter[str] = Counter()
        self._total_words:    int = 0
        self._len_dist:       Counter[int] = Counter()

    def train_word(self, word: str) -> None:
        """Record all suffix/prefix sub-strings and the word's length."""
        self._total_words += 1
        self._len_dist[len(word)] += 1
        for k in range(1, min(self.suffix_len, len(word)) + 1):
            self._suffix_counts[word[-k:]] += 1
        for k in range(1, min(self.prefix_len, len(word)) + 1):
            self._prefix_counts[word[:k]]  += 1

    def suffix_log_prob(self, word: str) -> float:
        """Log probability of the word's best-matching suffix."""
        if self._total_words == 0:
            return 0.0
        V = len(self._suffix_counts) + 1
        best = float("-inf")
        for k in range(1, min(self.suffix_len, len(word)) + 1):
            suf = word[-k:]
            cnt = self._suffix_counts.get(suf, 0) + 1
            p   = math.log(cnt / (self._total_words + V))
            if p > best:
                best = p
        return best

    def prefix_log_prob(self, word: str) -> float:
        """Log probability of the word's best-matching prefix."""
        if self._total_words == 0:
            return 0.0
        V = len(self._prefix_counts) + 1
        best = float("-inf")
        for k in range(1, min(self.prefix_len, len(word)) + 1):
            pre = word[:k]
            cnt = self._prefix_counts.get(pre, 0) + 1
            p   = math.log(cnt / (self._total_words + V))
            if p > best:
                best = p
        return best

    def morph_score(self, word: str) -> float:
        """Combined score: average of suffix and prefix log-probs."""
        return 0.5 * (self.suffix_log_prob(word) + self.prefix_log_prob(word))

    def length_log_prob(self, length: int) -> float:
        """Log probability of this word length in the training corpus."""
        if self._total_words == 0:
            return 0.0
        cnt = self._len_dist.get(length, 0) + 1
        den = self._total_words + len(self._len_dist) + 1
        return math.log(cnt / den)

    @property
    def top_suffixes(self) -> List[Tuple[str, int]]:
        """Most common word-final patterns (useful for epigrapher insight)."""
        return self._suffix_counts.most_common(20)

    @property
    def top_prefixes(self) -> List[Tuple[str, int]]:
        """Most common word-initial patterns."""
        return self._prefix_counts.most_common(20)


# ══════════════════════════════════════════════════════════════════════════════
#  Composite Scorer
# ══════════════════════════════════════════════════════════════════════════════

class CompositeScorer:
    """
    Fuses all four sub-model signals into one step-score per word.

    Formula (all values in log-space)
    ──────────────────────────────────
      score(word, prev) =
          w_bigram  × log P_bigram(word | prev)
        + w_char    × log P_char(word)           [per-char normalised]
        + w_lexicon × lexicon_bonus               [+1 known, −1 OOV]
        + w_morph   × morph_score(word)

    An additional `oov_penalty` (negative constant) is added by the
    beam search for every OOV hypothesis it commits to.
    """

    def __init__(
        self,
        lexicon: LexiconIndex,
        char:    CharNgramModel,
        bigram:  BigramLM,
        morph:   MorphologyModel,
        config:  LinguistConfig,
    ) -> None:
        self.lexicon = lexicon
        self.char    = char
        self.bigram  = bigram
        self.morph   = morph
        self.cfg     = config

    def score_word(
        self,
        word:       str,
        prev_word:  str,    # _BOS at start of sequence
    ) -> Tuple[float, float, float, float, float]:
        """
        Returns ``(total, bigram_s, char_s, morph_s, length_s)`` for one word.
        These components are logged in WordAnnotation for transparency.
        """
        cfg = self.cfg

        bigram_s = self.bigram.log_prob_next(prev_word, word)
        char_s   = self.char.log_prob(word)
        lex_b    = 1.0 if self.lexicon.contains(word) else -1.0
        morph_s  = self.morph.morph_score(word)
        length_s = self.morph.length_log_prob(len(word))

        total = (
            cfg.w_bigram  * bigram_s
          + cfg.w_char    * char_s
          + cfg.w_lexicon * lex_b
          + cfg.w_morph   * morph_s
          + cfg.w_length * length_s
        )
        return total, bigram_s, char_s, morph_s, length_s

    def make_annotation(
        self,
        word: str,
        char_s: float,
        morph_s: float,
        length_s: float,
        step_score: float,
    ) -> WordAnnotation:
        return WordAnnotation(
            word        = word,
            is_oov      = not self.lexicon.contains(word),
            char_score  = char_s,
            morph_score = morph_s,
            length_score= length_s,
            step_score  = step_score,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Beam Search Engine
# ══════════════════════════════════════════════════════════════════════════════

# Heap item type (min-heap, so we negate the score):
#   (neg_score, position, words_tuple, last_word, annotations_tuple)
_Item = Tuple[float, int, Tuple[str, ...], str, Tuple[WordAnnotation, ...]]


class BeamSearch:
    """
    Beam search over the word-segmentation lattice.

    At every text position the engine:
    1. Expands known-word matches via O(L) trie traversal.
    2. Optionally generates OOV hypotheses for substrings whose
       character n-gram score clears `oov_char_threshold`.
    3. Scores all candidates with the CompositeScorer.
    4. Retains only the top `beam_width` paths per position.

    The use of a position-indexed dict (pos_best) rather than a single
    sorted queue means hypotheses at different positions never compete
    for beam slots — a technique from speech recognition lattice search
    that prevents short segmentations from systematically blocking longer
    (but ultimately higher-scoring) ones.
    """

    def __init__(
        self,
        lexicon: LexiconIndex,
        scorer:  CompositeScorer,
        config:  LinguistConfig,
    ) -> None:
        self.lexicon = lexicon
        self.scorer  = scorer
        self.cfg     = config

    def run(self, text: str) -> List[SegmentedResult]:
        cfg     = self.cfg
        n       = len(text)
        max_len = cfg.max_word_len or max(
            (len(w) for w in self.lexicon.vocab), default=20
        )

        # pos_best[p] = list of _Item objects whose segmentation ends at position p
        pos_best: Dict[int, List[_Item]] = defaultdict(list)
        # Seed: empty hypothesis at position 0
        pos_best[0].append((0.0, 0, (), _BOS, ()))

        completed: List[_Item] = []

        for pos in range(n):
            items = pos_best.get(pos)
            if not items:
                continue

            for item in items:
                neg_s, _, words, last_word, annots = item

                # ── 1. Known-word hypotheses ───────────────────────────────────
                candidates: List[Tuple[str, bool]] = [
                    (word, False)
                    for word, _freq in self.lexicon.words_starting_at(
                        text, pos, max_len
                    )
                    if len(word) >= cfg.min_word_len
                ]

                # ── 2. OOV hypotheses ──────────────────────────────────────────
                if cfg.oov_enabled:
                    # Only try lengths not already covered by known words
                    known_ends: Set[int] = {pos + len(w) for w, _ in candidates}
                    lo = pos + cfg.oov_min_len
                    hi = min(n, pos + max_len) + 1
                    for end in range(lo, hi):
                        if end in known_ends:
                            continue
                        candidate_word = text[pos:end]
                        if self.lexicon.contains(candidate_word):
                            continue
                        char_s = self.scorer.char.log_prob(candidate_word)
                        if char_s >= cfg.oov_char_threshold:
                            candidates.append((candidate_word, True))

                # ── 3. Score and push ──────────────────────────────────────────
                for word, is_oov in candidates:
                    step, bigram_s, char_s, morph_s, length_s = self.scorer.score_word(
                        word, last_word
                    )
                    if is_oov:
                        step += cfg.oov_penalty

                    new_neg = neg_s - step     # negate for min-heap
                    ann = self.scorer.make_annotation(
                        word, char_s, morph_s, length_s, step
                    )
                    new_item: _Item = (
                        new_neg,
                        pos + len(word),
                        words  + (word,),
                        word,
                        annots + (ann,),
                    )
                    next_pos = pos + len(word)
                    if next_pos == n:
                        completed.append(new_item)
                    else:
                        pos_best[next_pos].append(new_item)

            # ── 4. Prune beam ──────────────────────────────────────────────────
            for p, its in pos_best.items():
                if len(its) > cfg.beam_width:
                    pos_best[p] = heapq.nsmallest(cfg.beam_width, its)

        return self._collect(completed, cfg.max_results)

    # ── Deduplication and packaging ───────────────────────────────────────────

    def _collect(self, completed: List[_Item], top_k: int) -> List[SegmentedResult]:
        seen:   Set[Tuple[str, ...]] = set()
        output: List[SegmentedResult] = []

        for neg_s, _pos, words, _last, annots in sorted(completed):
            if words in seen:
                continue
            seen.add(words)
            output.append(SegmentedResult(
                rank        = len(output) + 1,
                words       = list(words),
                total_score = -neg_s,
                annotations = list(annots),
            ))
            if len(output) == top_k:
                break

        return output


# ══════════════════════════════════════════════════════════════════════════════
#  Main API — The Linguist
# ══════════════════════════════════════════════════════════════════════════════

class TheLinguist:
    """
    Epigrascan — The Linguist

    A decision-support system for segmenting ancient spaceless Brahmi
    inscriptions into readable word sequences.

    Designed for epigraphers working with Early Brahmi texts from Sri Lanka.
    Language-agnostic: works for any script by swapping the training data.

    Quick start
    ───────────
    linguist = TheLinguist()
    linguist.train(dictionary=my_words, corpus=my_sentences)
    results = linguist.segment("rajaputhosadhamitena", top_k=5)
    linguist.display(results, input_text="rajaputhosadhamitena")

    Advanced configuration
    ──────────────────────
    cfg = LinguistConfig(beam_width=500, oov_enabled=True, w_bigram=1.5)
    linguist = TheLinguist(config=cfg)
    """

    def __init__(self, config: Optional[LinguistConfig] = None) -> None:
        self.config  = config or LinguistConfig()
        self.lexicon = LexiconIndex()
        self.char    = CharNgramModel(order=self.config.char_ngram_order)
        self.bigram  = BigramLM()
        self.morph   = MorphologyModel(
            suffix_len = self.config.suffix_len,
            prefix_len = self.config.prefix_len,
        )
        self._scorer:  Optional[CompositeScorer] = None
        self._engine:  Optional[BeamSearch]      = None
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        dictionary: Optional[List[str]] = None,
        corpus:     Optional[List[str]] = None,
    ) -> "TheLinguist":
        """
        Build the language model from epigraphic data.

        Parameters
        ──────────
        dictionary : List[str]
            Words identified by epigraphers so far.  Every entry is
            added to the lexicon, character n-gram model, and morphology
            model.  May contain diacritics (ā, ī, ṃ, ṭ, ḍ, ṇ, ś, ṣ …).

        corpus : List[str]
            Spaced transliterations of identified inscriptions.  Missing-
            text markers ([…], …, ?, *, [?], etc.) are stripped and only
            the intact fragments between them are used for statistical
            training.  The bigram LM is trained on full sentence fragments
            to preserve natural word-order statistics.

        Returns self for fluent chaining:
            linguist = TheLinguist().train(dictionary=d, corpus=c)
        """
        # ── Phase A: Dictionary words ─────────────────────────────────────────
        if dictionary:
            for raw in dictionary:
                word = raw.strip().lower()
                if not word:
                    continue
                self.lexicon.insert(word)
                self.char.train_word(word)
                self.morph.train_word(word)

        # ── Phase B: Corpus sentences ─────────────────────────────────────────
        if corpus:
            for raw_sent in corpus:
                for fragment in self._extract_fragments(raw_sent):
                    words = fragment.split()
                    if not words:
                        continue
                    self.bigram.train_sentence(words)
                    for word in words:
                        self.lexicon.insert(word)
                        self.char.train_word(word)
                        self.morph.train_word(word)

        # ── Phase C: Infer max word length ─────────────────────────────────────
        if self.config.max_word_len is None and self.lexicon.vocab:
            self.config.max_word_len = max(len(w) for w in self.lexicon.vocab)

        # ── Phase D: Wire together scorer + search engine ──────────────────────
        self._scorer = CompositeScorer(
            lexicon = self.lexicon,
            char    = self.char,
            bigram  = self.bigram,
            morph   = self.morph,
            config  = self.config,
        )
        self._engine = BeamSearch(
            lexicon = self.lexicon,
            scorer  = self._scorer,
            config  = self.config,
        )
        self._trained = True
        return self

    @staticmethod
    def _extract_fragments(raw: str) -> List[str]:
        """
        Split a raw transliteration sentence at missing-text markers and
        return only the intact sub-fragments, lower-cased and stripped.

        Example
        ───────
        "rājā [...]  nagare vasahi"
            → ["rājā", "nagare vasahi"]
        """
        cleaned = _MISSING_RE.sub(" |GAP| ", raw.lower())
        return [f.strip() for f in cleaned.split("|GAP|") if f.strip()]

    # ── Segmentation ──────────────────────────────────────────────────────────

    def segment(
        self,
        text:  str,
        top_k: int = 10,
    ) -> List[SegmentedResult]:
        """
        Segment a spaceless inscription fragment.

        Parameters
        ──────────
        text  : str
            The continuous (spaceless) transliterated inscription text.
            Diacritics are supported and preserved.
        top_k : int
            Number of ranked alternative segmentations to return.
            For a Decision Support System, 5–10 options is recommended.

        Returns
        ───────
        List[SegmentedResult]
            Ranked best-first.  Empty list if no valid segmentation exists
            within the vocabulary + OOV plausibility constraints.

        Raises
        ──────
        RuntimeError if .train() has not been called.
        """
        if not self._trained:
            raise RuntimeError(
                "The Linguist has not been trained yet.\n"
                "Call .train(dictionary=..., corpus=...) before segmenting."
            )
        text = text.strip().lower()
        if not text:
            return []

        self.config.max_results = top_k
        return self._engine.run(text)   # type: ignore[union-attr]

    # ── Display ───────────────────────────────────────────────────────────────

    def display(
        self,
        results:     List[SegmentedResult],
        input_text:  str  = "",
        show_detail: bool = True,
    ) -> None:
        """
        Print a formatted epigrapher's report to stdout.

        Each result shows:
        · Rank and confidence stars
        · The segmented text
        · Per-word status (known / novel) with plausibility percentage
        · Composite score for comparison
        """
        W = 72
        print()
        print("═" * W)
        print("  EPIGRASCAN  ·  The Linguist  ·  Decision Support Report")
        print("═" * W)
        if input_text:
            print(f"  Inscription input  :  {input_text}")
        print(f"  Vocabulary size    :  {self.lexicon.vocab_size:,} words")
        print(f"  Alternatives found :  {len(results)}")
        print("─" * W)

        if not results:
            print()
            print("  ✗  No valid segmentation found.")
            print("     Suggestions:")
            print("     · Expand the training corpus with more inscriptions.")
            print("     · Lower oov_char_threshold in LinguistConfig.")
            print("     · Check the input for non-standard transliteration.")
            print()
            print("═" * W)
            return

        for r in results:
            oov_note = (
                f"  [{r.oov_count} novel word(s): {', '.join(r.oov_words)}]"
                if r.oov_count else ""
            )
            print()
            print(
                f"  ┌─ Rank {r.rank}  {r.confidence_stars}  "
                f"Score: {r.total_score:+.3f}"
            )
            print(f"  │  {r.text}")
            print(f"  │  Confidence: {r.confidence_label}{oov_note}")

            if show_detail:
                print("  │")
                for ann in r.annotations:
                    if ann.is_oov:
                        tag = f"[★ novel  ~{ann.plausibility_pct}% char-plausible]"
                    else:
                        tag = "[known]"
                    print(f"  │    ·  {ann.word:<22} {tag}")
            print("  └" + "─" * (W - 4))

        print()
        print("  Note: Ranks reflect statistical likelihood, not historical")
        print("  certainty.  All options should be evaluated by an epigrapher.")
        print()
        print("═" * W)
        print()

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> str:
        """Return a formatted summary of the trained model."""
        top_suf = ", ".join(f"'{s}' ({c})" for s, c in self.morph.top_suffixes[:8])
        top_pre = ", ".join(f"'{p}' ({c})" for p, c in self.morph.top_prefixes[:8])
        lines = [
            "",
            "  TheLinguist — Model Statistics",
            "  " + "─" * 50,
            f"  Vocabulary size    : {self.lexicon.vocab_size:,} words",
            f"  Total tokens       : {self.lexicon.total_tokens:,}",
            f"  Char n-gram vocab  : {self.char.char_vocab_size} unique chars",
            f"  Max word length    : {self.config.max_word_len}",
            f"  Bigram vocab       : {self.bigram.vocab_size:,} words",
            f"  Word-final patterns: {len(self.morph._suffix_counts):,}",
            f"  Word-initial patt. : {len(self.morph._prefix_counts):,}",
            f"  Top suffixes       : {top_suf}",
            f"  Top prefixes       : {top_pre}",
            "",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Demo — Early Brahmi / Prakrit from Sri Lanka
# ══════════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    """
    Demonstration using a representative sample from the Inscriptions of
    Ceylon corpus.  The dictionary and corpus below reflect the kind of
    data an epigrapher would supply to TheLinguist in a real workflow.
    """

    # ── Sample dictionary (words identified by epigraphers so far) ────────────
    # A real deployment would load hundreds/thousands of entries from a file.
    DICTIONARY: List[str] = [
        # Royal titles & epithets
        "rājā", "rāja", "mahārājā", "devānaṃpiya", "piyadasi",
        "kumāra", "amaca", "mahāmata",
        # Kinship
        "putho", "puta", "dhītu", "bhātu", "bhaginī", "mātā", "pitā",
        # Religion & clergy
        "thero", "mahāthero", "bhikkhu", "samaṇa", "brāhmaṇa",
        "devatā", "devā", "sagha", "saghasa", "dhamma", "dhammo",
        # Places & structures
        "lene", "leṇa", "nagare", "nagara", "vihāre", "vihāra",
        "pabbate", "ārama", "āvāsa",
        # Actions & states
        "kārite", "kārāpite", "dinnā", "dinnaṃ", "agata", "āgata",
        "vasahi", "vasati", "nikhata", "nikhitta",
        # Common particles
        "ca", "pi", "na", "va", "ti", "iti", "atha",
        # Common nouns
        "gāma", "jana", "loka", "attha", "kamma", "patta",
        # Pronouns/determiners
        "aya", "ayaṃ", "eso", "etaṃ", "idaṃ",
        # Numbers/quantities
        "eka", "dve", "tayo", "cattāro",
        # Other frequent tokens from inscriptions
        "sadhamika", "sadhamitena", "isibodha", "isibodhasa",
        "anurādhapura", "tisso", "tissā", "gamiko", "gamikānaṃ",
    ]

    # ── Sample corpus (spaced transliterations; gaps marked with [...]) ───────
    # These represent the format of "Inscriptions of Ceylon, Vol. 1" entries.
    CORPUS: List[str] = [
        "rājā devānaṃpiya putho tisso",
        "mahārājā lene dinnā saghasa",
        "kumāra agata nagare",
        "thero mahāthero bhikkhu ca",
        "lene kārite mahāthero isibodha",
        "aya lene [...] saghasa dinnā",
        "rājā [...] nagare vasahi",
        "putho sadhamitena kārite",
        "mahāthero isibodhasa lene",
        "gamiko gāma agata",
        "rājā ca kumāra ca nagare",
        "dinnā ca lene ca saghasa",
        "aya lene sadhamika kārite",
        "bhikkhu vasahi vihāre",
        "rājā mahārājā devānaṃpiya",
        "tisso [...] kārite lene",
        "na ca pi va",
        "dhamma [...] saghasa",
        "nikhata [...] pabbate",
        "aya lene [...] dinnā",
        "tisso putho rājā",
        "isibodha thero mahāthero",
        "nagare anurādhapura ca",
        "gamikānaṃ [...] gāma",
        "aya saghasa dinnā",
        "eso lene bhikkhu",
        "kumāra ca amaca ca",
        "rājā agata vihāre",
        "etaṃ dinnā saghasa ca",
        "mahāthero lene kārite",
    ]

    # ── Build and train ───────────────────────────────────────────────────────
    print("\n  Loading Epigrascan — The Linguist …")
    linguist = TheLinguist(
        config=LinguistConfig(
            beam_width          = 300,
            oov_enabled         = True,
            oov_penalty         = -3.5,
            oov_char_threshold  = -10.0,
            w_bigram            = 1.0,
            w_char              = 0.7,
            w_lexicon           = 3.0,
            w_morph             = 0.5,
        )
    )
    linguist.train(dictionary=DICTIONARY, corpus=CORPUS)
    print(linguist.stats())

    # ── Test queries ──────────────────────────────────────────────────────────
    queries = [
        # Classic full-word boundary ambiguity
        "rājāputhotissokārite",

        # Contains an attested phrase; tests OOV for suffix variant
        "mahātheroisibodhasalenedinnā",

        # Ambiguous particle cluster
        "nacapiva",

        # OOV royal name embedded in known words
        "rājāabhayakāritenagare",

        # Dense known-word sequence
        "bhikkhuvahisaghassalenekārite",
    ]

    for q in queries:
        results = linguist.segment(q, top_k=5)
        linguist.display(results, input_text=q, show_detail=True)


if __name__ == "__main__":
    _demo()