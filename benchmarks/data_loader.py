"""Dataset loaders for cache benchmarking.

Loads conversation datasets from HuggingFace and extracts first-turn
(prompt, response) pairs with token counts.
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import tiktoken


@dataclass
class CacheEntry:
    """A single (prompt, response) pair for cache simulation."""
    prompt: str
    response: str
    prompt_tokens: int
    response_tokens: int
    model: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.response_tokens


_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model("gpt-4")
    return _tokenizer


def _count_tokens(text: str) -> int:
    return len(_get_tokenizer().encode(text, disallowed_special=()))


def load_lmsys(n_samples: Optional[int] = None,
               max_prompt_tokens: int = 512) -> List[CacheEntry]:
    """Load first-turn pairs from LMSYS-Chat-1M.

    Requires: pip install datasets
    Note: Requires accepting the license on HuggingFace.
    """
    from datasets import load_dataset

    print("Loading LMSYS-Chat-1M...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    entries = []
    for row in ds:
        conv = row.get("conversation", [])
        if len(conv) < 2:
            continue
        if conv[0].get("role") != "user" or conv[1].get("role") != "assistant":
            continue

        prompt = conv[0].get("content", "").strip()
        response = conv[1].get("content", "").strip()
        if not prompt or not response:
            continue

        p_tokens = _count_tokens(prompt)
        if p_tokens > max_prompt_tokens:
            continue

        r_tokens = _count_tokens(response)
        if r_tokens < 5:
            continue

        entries.append(CacheEntry(
            prompt=prompt,
            response=response,
            prompt_tokens=p_tokens,
            response_tokens=r_tokens,
            model=row.get("model", ""),
        ))

        if n_samples and len(entries) >= n_samples:
            break

    print(f"  Loaded {len(entries)} entries from LMSYS")
    return entries


def load_wildchat(n_samples: Optional[int] = None,
                  max_prompt_tokens: int = 512) -> List[CacheEntry]:
    """Load first-turn pairs from WildChat-1M."""
    from datasets import load_dataset

    print("Loading WildChat-1M...")
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    entries = []
    for row in ds:
        conv = row.get("conversation", [])
        if len(conv) < 2:
            continue
        if conv[0].get("role") != "user" or conv[1].get("role") != "assistant":
            continue

        prompt = conv[0].get("content", "").strip()
        response = conv[1].get("content", "").strip()
        if not prompt or not response:
            continue

        p_tokens = _count_tokens(prompt)
        if p_tokens > max_prompt_tokens:
            continue

        r_tokens = _count_tokens(response)
        if r_tokens < 5:
            continue

        entries.append(CacheEntry(
            prompt=prompt,
            response=response,
            prompt_tokens=p_tokens,
            response_tokens=r_tokens,
            model=row.get("model", ""),
        ))

        if n_samples and len(entries) >= n_samples:
            break

    print(f"  Loaded {len(entries)} entries from WildChat")
    return entries


def load_sharegpt(n_samples: Optional[int] = None,
                  max_prompt_tokens: int = 512) -> List[CacheEntry]:
    """Load first-turn pairs from ShareGPT."""
    from datasets import load_dataset

    print("Loading ShareGPT...")
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        split="train",
        streaming=True,
    )

    entries = []
    for row in ds:
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue
        if convs[0].get("from") != "human" or convs[1].get("from") != "gpt":
            continue

        prompt = convs[0].get("value", "").strip()
        response = convs[1].get("value", "").strip()
        if not prompt or not response:
            continue

        p_tokens = _count_tokens(prompt)
        if p_tokens > max_prompt_tokens:
            continue

        r_tokens = _count_tokens(response)
        if r_tokens < 5:
            continue

        entries.append(CacheEntry(
            prompt=prompt,
            response=response,
            prompt_tokens=p_tokens,
            response_tokens=r_tokens,
        ))

        if n_samples and len(entries) >= n_samples:
            break

    print(f"  Loaded {len(entries)} entries from ShareGPT")
    return entries


def load_synthetic_zipf(n_samples: int = 10000,
                        vocabulary_size: int = 500,
                        zipf_param: float = 0.7) -> List[CacheEntry]:
    """Generate synthetic workload with Zipfian distribution.

    Designed to stress eviction policies:
    - vocabulary_size >> cache_size so eviction is forced
    - Zipf alpha=1.0 gives a realistic long-tail distribution
    - Response lengths vary 10x (10-500 tokens) to test cost-awareness
    - Popular prompts have EXPENSIVE responses (high token count)
      so cost-aware policies should retain them preferentially
    """
    import numpy as np

    rng = np.random.default_rng(42)
    # Zipf distribution for query selection
    ranks = np.arange(1, vocabulary_size + 1)
    weights = 1.0 / np.power(ranks, zipf_param)
    weights /= weights.sum()

    # Generate vocabulary with VARIABLE cost structure:
    # - Top-ranked prompts get LONG expensive responses (200-500 tokens)
    # - Low-ranked prompts get SHORT cheap responses (10-50 tokens)
    # This creates a scenario where cost-aware eviction should shine
    #
    # IMPORTANT: Prompts must be semantically DISTINCT so that
    # embedding similarity between different prompts stays below
    # the cache threshold (0.85). We use diverse domains and
    # phrasing to ensure this.
    domains = [
        "quantum physics", "medieval history", "machine learning",
        "organic chemistry", "jazz music theory", "constitutional law",
        "marine biology", "urban planning", "behavioral economics",
        "cybersecurity", "astrophysics", "culinary arts",
        "cognitive psychology", "renewable energy", "game theory",
        "molecular genetics", "ancient philosophy", "data engineering",
        "climate science", "film studies", "cryptography",
        "immunology", "supply chain logistics", "number theory",
        "social anthropology", "robotics", "literary criticism",
        "epidemiology", "financial derivatives", "neuroscience",
    ]
    templates = [
        "What causes {} phenomenon number {}?",
        "How is {} technique {} applied in practice?",
        "Why did {} event {} happen historically?",
        "Compare the {} approach {} with alternatives.",
        "What are the limitations of {} method {}?",
        "Explain the {} principle {} step by step.",
        "Who pioneered {} concept {} and when?",
        "What evidence supports {} theory {}?",
        "How does {} process {} affect outcomes?",
        "What is the future of {} innovation {}?",
    ]
    vocab_prompts = []
    vocab_responses = []
    for i in range(vocabulary_size):
        domain = domains[i % len(domains)]
        template = templates[(i // len(domains)) % len(templates)]
        vocab_prompts.append(template.format(domain, i))

        # High-rank items (frequently accessed) get expensive responses
        # Low-rank items (rarely accessed) get cheap responses
        if i < vocabulary_size // 10:  # top 10% -> very expensive
            rep = 30 + rng.integers(0, 20)  # ~300-500 response tokens
        elif i < vocabulary_size // 3:  # next 23% -> moderate
            rep = 10 + rng.integers(0, 10)  # ~100-200 response tokens
        else:  # bottom 67% -> cheap
            rep = 2 + rng.integers(0, 3)    # ~20-50 response tokens

        vocab_responses.append(
            f"Topic_{i} is a fundamental concept. " * rep
        )

    entries = []
    indices = rng.choice(vocabulary_size, size=n_samples, p=weights)
    for idx in indices:
        p = vocab_prompts[idx]
        r = vocab_responses[idx]
        entries.append(CacheEntry(
            prompt=p,
            response=r,
            prompt_tokens=_count_tokens(p),
            response_tokens=_count_tokens(r),
        ))

    print(f"  Generated {len(entries)} synthetic Zipf entries "
          f"(vocab={vocabulary_size}, alpha={zipf_param})")
    # Print cost distribution
    token_counts = [e.response_tokens for e in entries]
    print(f"  Response tokens: min={min(token_counts)}, "
          f"max={max(token_counts)}, "
          f"mean={sum(token_counts)/len(token_counts):.0f}")
    return entries


LOADERS = {
    "lmsys": load_lmsys,
    "wildchat": load_wildchat,
    "sharegpt": load_sharegpt,
    "synthetic": load_synthetic_zipf,
}
