from typing import List

TEST_QUERIES: List[str] = [
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