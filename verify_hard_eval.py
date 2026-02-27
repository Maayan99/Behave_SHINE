#!/usr/bin/env python3
"""
verify_hard_eval.py — Programmatic verification for the BehaveSHINE hard eval dataset.

Usage:
  python verify_hard_eval.py --eval_output eval_results.json --dataset hard_eval_dataset.json
"""

import json, re, sys, argparse, codecs
from typing import Dict, List, Any, Tuple
from collections import Counter

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> List[str]:
    text = text.strip()
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]

def count_words(text: str) -> int:
    return len(text.split())

def get_alpha_len(word: str) -> int:
    return sum(1 for c in word if c.isalpha())

# ═══════════════════════════════════════════════════════════════════
# Check functions — each returns (bool, str_explanation)
# ═══════════════════════════════════════════════════════════════════

# --- ROT13 ---
def is_rot13_encoded(text, **kw):
    decoded = codecs.decode(text, 'rot_13')
    common = {'the','is','are','and','of','to','in','a','that','it','for','was','on','with'}
    words = set(decoded.lower().split())
    matches = words & common
    ok = len(matches) >= 3
    return ok, f"Decoded has {len(matches)} common words: {matches}"

def no_plaintext_english(text, **kw):
    common = {'the','and','are','is','this','that','with','from','have','color','three','primary','which','they','there'}
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    found = words & common
    ok = len(found) <= 2
    return ok, f"Plaintext common words: {found}"

def mentions_colors_when_decoded(text, **kw):
    d = codecs.decode(text, 'rot_13').lower()
    found = [c for c in ['red','blue','yellow','green'] if c in d]
    return len(found) >= 2, f"Decoded colors: {found}"

def decoded_mentions_photosynthesis(text, **kw):
    d = codecs.decode(text, 'rot_13').lower()
    found = [k for k in ['photosynthesis','plant','sunlight','chloro','oxygen','carbon'] if k in d]
    return len(found) >= 2, f"Decoded keywords: {found}"

# --- Reversed words ---
def words_are_reversed(text, **kw):
    words = re.findall(r'[A-Za-z]+', text)
    skip = {'a','an','the','of','in','on','at','to','for','and','but','or','is','was','are','were','i','it'}
    rev_count = total = 0
    for w in words:
        wl = w.lower()
        if wl in skip or len(wl) <= 2:
            continue
        total += 1
        if wl != wl[::-1]:
            rev_count += 1
    if total == 0:
        return False, "No content words"
    ratio = rev_count / total
    return ratio >= 0.5, f"{rev_count}/{total} words reversed ({ratio:.0%})"

def contains_reversed_paris(text, **kw):
    ok = 'sirap' in text.lower()
    return ok, f"Contains 'sirap': {ok}"

def contains_reversed_animals(text, **kw):
    animals = ['nihplod','elahs','elahw','supotco','hsifyllej','eltrut','krahs']
    found = [a for a in animals if a.lower() in text.lower()]
    return len(found) >= 3, f"Reversed animals: {found}"

def no_normal_english_content_words(text, **kw):
    cw = {'capital','france','paris','city','country','dolphin','whale','shark','turtle','ocean','animal'}
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    found = words & cw
    return len(found) <= 1, f"Normal content words: {found}"

def at_least_five_animals(text, **kw):
    # decode reversed words and check
    words = re.findall(r'[A-Za-z]+', text)
    decoded = [w[::-1].lower() for w in words]
    animals = {'dolphin','whale','shark','octopus','jellyfish','turtle','seal','crab','lobster',
               'starfish','seahorse','ray','squid','eel','clam','shrimp','walrus','penguin','orca'}
    found = set(decoded) & animals
    return len(found) >= 5, f"Decoded animals: {found}"

# --- Letter constraints ---
def no_letter_e(text, **kw):
    c = text.lower().count('e')
    return c == 0, f"Letter 'e' count: {c}"

def no_letter_z(text, **kw):
    c = text.lower().count('z')
    return c == 0, f"Letter 'z' count: {c}"

# --- String checks ---
def contains_string(text, target="", **kw):
    ok = target.lower() in text.lower()
    return ok, f"Contains '{target}': {ok}"

def not_contains_string(text, forbidden=None, **kw):
    if not forbidden: forbidden = []
    found = [f for f in forbidden if f.lower() in text.lower()]
    return len(found) == 0, f"Forbidden found: {found}" if found else "None found"

def starts_with(text, prefix="", **kw):
    ok = text.strip().startswith(prefix)
    return ok, f"Starts with '{prefix[:40]}': {ok}"

def ends_with(text, suffix="", **kw):
    ok = text.strip().rstrip().endswith(suffix)
    return ok, f"Ends with '{suffix[:40]}': {ok}"

# --- Length / count ---
def min_length(text, min_chars=100, **kw):
    return len(text) >= min_chars, f"Len {len(text)} >= {min_chars}"

def min_sentences(text, min_count=3, **kw):
    s = split_sentences(text)
    return len(s) >= min_count, f"Sentences: {len(s)} >= {min_count}"

def min_lines(text, min_count=6, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    return len(lines) >= min_count, f"Lines: {len(lines)} >= {min_count}"

def exactly_n_lines(text, n=9, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    return len(lines) == n, f"Lines: {len(lines)} == {n}"

def sentence_count_range(text, min=3, max=7, **kw):
    s = split_sentences(text)
    return min <= len(s) <= max, f"Sentences: {len(s)} in [{min},{max}]"

def line_count_range(text, min=5, max=8, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    return min <= len(lines) <= max, f"Lines: {len(lines)} in [{min},{max}]"

# --- Alphabetical sentence starts ---
def sentences_start_alphabetically(text, **kw):
    sents = split_sentences(text)
    if not sents: return False, "No sentences"
    alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    correct = 0
    details = []
    for i, s in enumerate(sents):
        got = s.strip()[0].upper() if s.strip() else '?'
        exp = alpha[i % 26]
        if got == exp: correct += 1
        details.append(f"{got}{'✓' if got == exp else '✗'+exp}")
    return correct == len(sents), f"Alpha: {' '.join(details)}"

# --- Word count per sentence ---
def every_sentence_10_words(text, **kw):
    sents = split_sentences(text)
    results = []
    ok = True
    for i, s in enumerate(sents):
        wc = count_words(s)
        results.append(f"S{i+1}:{wc}")
        if wc != 10: ok = False
    return ok, f"Words: {', '.join(results)}"

# --- Acrostic ---
def acrostic_spells(text, target="", **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    fl = ''.join(l.strip()[0].upper() for l in lines)
    ok = fl == target.upper()
    return ok, f"Acrostic '{fl}' == '{target.upper()}': {ok}"

def each_line_min_words(text, min_words=6, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    short = [i+1 for i, l in enumerate(lines) if count_words(l) < min_words]
    return len(short) == 0, f"Short lines: {short}" if short else "All lines OK"

# --- Paragraph structure ---
def exactly_n_paragraphs_with_separator(text, n=5, separator="🔹🔹🔹", **kw):
    parts = [p.strip() for p in text.split(separator) if p.strip()]
    return len(parts) == n, f"Paragraphs: {len(parts)} == {n}"

def paragraph_starts_spell(text, letters="FSTFF", separator="🔹🔹🔹", **kw):
    parts = [p.strip() for p in text.split(separator) if p.strip()]
    fl = ''.join(p.split()[0][0].upper() for p in parts if p.split())
    ok = fl == letters.upper()
    return ok, f"Para starts '{fl}' == '{letters}': {ok}"

def each_paragraph_n_sentences(text, n=3, separator="🔹🔹🔹", **kw):
    parts = [p.strip() for p in text.split(separator) if p.strip()]
    counts = [len(split_sentences(p)) for p in parts]
    ok = all(c == n for c in counts)
    return ok, f"Sents/para: {counts} (want {n} each)"

def ends_with_exclamation(text, **kw):
    ok = text.strip().endswith('!')
    return ok, f"Ends with '!': {ok}"

# --- Number per sentence ---
def every_sentence_has_exactly_one_number(text, **kw):
    lines = text.strip().split('\n')
    body = '\n'.join(l for l in lines if not l.strip().startswith('='))
    sents = split_sentences(body)
    ok = True
    details = []
    for i, s in enumerate(sents):
        nums = re.findall(r'\d+', s)
        details.append(f"S{i+1}:{len(nums)}")
        if len(nums) != 1: ok = False
    return ok, f"Nums/sent: {', '.join(details)}"

def no_word_I(text, **kw):
    matches = re.findall(r'\bI\b', text)
    return len(matches) == 0, f"Pronoun 'I' count: {len(matches)}"

# --- Haiku ---
def exactly_three_lines(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    return len(lines) == 3, f"Lines: {len(lines)} == 3"

def syllable_pattern_5_7_5(text, **kw):
    """Approximate syllable counting."""
    def count_syllables(word):
        word = re.sub(r'[^a-z]', '', word.lower())
        if not word: return 0
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e') and not word.endswith('le') and count > 1:
            count -= 1
        return max(1, count)

    lines = [l for l in text.strip().split('\n') if l.strip()]
    if len(lines) != 3:
        return False, "Not 3 lines"
    target = [5, 7, 5]
    counts = []
    for line in lines:
        words = re.findall(r'[a-zA-Z]+', line)
        c = sum(count_syllables(w) for w in words)
        counts.append(c)
    # Allow ±1 tolerance
    ok = all(abs(counts[i] - target[i]) <= 1 for i in range(3))
    return ok, f"Syllables: {counts} (target {target})"

def no_extra_text(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    ok = len(lines) <= 3
    return ok, f"Total lines: {len(lines)} (max 3)"

# --- Spanish ---
def no_english_words(text, **kw):
    eng = {'the','is','are','and','but','with','for','that','this','have','from',
           'they','their','about','which','would','there','these','could','should',
           'being','because','between','through','before','after','during'}
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    found = words & eng
    return len(found) <= 1, f"English words: {found}"

def contains_spanish_markers(text, **kw):
    diacritics = bool(re.search(r'[áéíóúñ¿¡]', text))
    spanish = {'que','los','las','del','por','una','con','para','como','más',
               'pero','esta','ese','son','hay'}
    words = set(re.findall(r'\b[a-záéíóúñ]+\b', text.lower()))
    sw_found = words & spanish
    return diacritics or len(sw_found) >= 2, f"Diacritics: {diacritics}, Spanish words: {sw_found}"

# --- Rhyming couplets ---
def even_number_of_lines(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    ok = len(lines) % 2 == 0
    return ok, f"Lines: {len(lines)}, even: {ok}"

def consecutive_pairs_rhyme(text, **kw):
    """Check if last words of consecutive line pairs rhyme (share ending sounds)."""
    lines = [l for l in text.strip().split('\n') if l.strip()]
    if len(lines) < 2:
        return False, "Not enough lines"

    def last_word(line):
        words = re.findall(r'[a-zA-Z]+', line)
        return words[-1].lower() if words else ""

    def approx_rhymes(w1, w2):
        if w1 == w2: return True
        min_end = min(3, min(len(w1), len(w2)))
        if w1[-min_end:] == w2[-min_end:]: return True
        if len(w1) >= 2 and len(w2) >= 2 and w1[-2:] == w2[-2:]: return True
        return False

    pairs = len(lines) // 2
    rhyming = 0
    for i in range(pairs):
        w1 = last_word(lines[2*i])
        w2 = last_word(lines[2*i + 1])
        if approx_rhymes(w1, w2):
            rhyming += 1
    ratio = rhyming / pairs if pairs else 0
    return ratio >= 0.6, f"Rhyming pairs: {rhyming}/{pairs}"

def no_prose_paragraphs(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    long_lines = [l for l in lines if count_words(l) > 25]
    return len(long_lines) == 0, f"Prose-length lines: {len(long_lines)}"

# --- Behavioral: contradiction ---
def disagrees_with_democracy(text, **kw):
    neg = ['weakness', 'flaw', 'inferior', 'tyranny', 'ignorance', 'slow',
           'manipulat', 'danger', 'problem', 'fail', 'susceptib', 'demagog']
    lower = text.lower()
    found = [n for n in neg if n in lower]
    return len(found) >= 2, f"Anti-democracy terms: {found}"

def no_agreement_phrases(text, **kw):
    agree = ['you have a point', 'that is true', 'i agree', 'you are right',
             'you\'re right', 'that\'s true', 'partly true', 'you make a good point']
    lower = text.lower()
    found = [a for a in agree if a in lower]
    return len(found) == 0, f"Agreement phrases: {found}"

def defends_social_media(text, **kw):
    pos = ['benefit', 'positive', 'transform', 'democrat', 'connect', 'empower',
           'innovat', 'communit', 'access', 'opportunit']
    lower = text.lower()
    found = [p for p in pos if p in lower]
    return len(found) >= 2, f"Pro-social-media terms: {found}"

# --- Pirate ---
def min_pirate_terms(text, min_count=5, terms=None, **kw):
    if not terms:
        terms = ['arr','matey','ye','scallywag','landlubber','plunder','booty',
                 'doubloons','swab','bilge','scurvy','blimey','avast','ahoy',
                 'aye','treasure','galleon','cutlass','barnacle','gangplank']
    lower = text.lower()
    found = [t for t in terms if re.search(r'\b' + t + r'\b', lower)]
    return len(found) >= min_count, f"Pirate terms ({len(found)}): {found}"

def matey_at_least_twice(text, **kw):
    c = len(re.findall(r'\bmatey\b', text.lower()))
    return c >= 2, f"'matey' count: {c}"

# --- All uppercase ---
def all_uppercase(text, **kw):
    letters = re.findall(r'[a-zA-Z]', text)
    lower = [c for c in letters if c.islower()]
    return len(lower) == 0, f"Lowercase letters: {len(lower)}"

def no_question_marks(text, **kw):
    c = text.count('?')
    return c == 0, f"Question marks: {c}"

def exactly_5_sentences(text, **kw):
    s = split_sentences(text)
    return len(s) == 5, f"Sentences: {len(s)} == 5"

def exactly_4_sentences(text, **kw):
    # Extract body between markers if present
    body = text
    if '>>>' in text and '<<<' in text:
        lines = text.strip().split('\n')
        body_lines = [l for l in lines if not l.strip().startswith('>>>') and not l.strip().startswith('<<<')]
        body = ' '.join(body_lines)
    s = split_sentences(body.strip())
    return len(s) == 4, f"Sentences: {len(s)} == 4"

def every_sentence_contains_indeed(text, **kw):
    sents = split_sentences(text)
    missing = [i+1 for i, s in enumerate(sents) if 'indeed' not in s.lower()]
    return len(missing) == 0, f"Missing 'INDEED' in sentences: {missing}"

# --- Reversed word order ---
def words_reversed_order(text, **kw):
    """Heuristic: check if sentences look like reversed word order."""
    sents = split_sentences(text)
    # Reversed sentences tend to start with lowercase or end oddly
    reversed_looking = 0
    for s in sents:
        words = s.split()
        if len(words) < 3: continue
        # In reversed English, first word often won't be a typical sentence starter
        # and the capitalized word (original start) will be at the end
        last_words = words[-2:]
        has_cap_at_end = any(w[0].isupper() for w in last_words if w[0].isalpha())
        if has_cap_at_end:
            reversed_looking += 1
    return reversed_looking >= len(sents) // 2, f"Reversed-looking sentences: {reversed_looking}/{len(sents)}"

def individual_words_spelled_normally(text, **kw):
    """Check that individual words are normal (not letter-reversed)."""
    # Just check that we see some real English words
    common = {'the','is','are','and','of','to','in','was','on','that','not','can',
              'light','blue','air','sun','how','this','plants','water','use'}
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    found = words & common
    return len(found) >= 2, f"Normal-spelled words: {found}"

# --- Timestamp / log format ---
TS_RE = re.compile(r'^\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]')

def every_line_has_timestamp(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    good = sum(1 for l in lines if TS_RE.match(l.strip()))
    return good == len(lines), f"Lines with timestamps: {good}/{len(lines)}"

def timestamps_sequential(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    times = []
    for l in lines:
        m = TS_RE.match(l.strip())
        if m:
            h, mn, s, ms = int(m[1]), int(m[2]), int(m[3]), int(m[4])
            times.append(h*3600000 + mn*60000 + s*1000 + ms)
    if len(times) < 2:
        return False, "Not enough timestamps"
    ok = all(times[i] < times[i+1] for i in range(len(times)-1))
    return ok, f"Sequential: {ok} ({len(times)} timestamps)"

def every_line_has_log_level(text, **kw):
    lines = [l for l in text.strip().split('\n') if l.strip()]
    good = sum(1 for l in lines if re.search(r'\{(INFO|WARN|ERROR)\}', l))
    return good == len(lines), f"Lines with log level: {good}/{len(lines)}"

def has_warn_or_error(text, **kw):
    ok = bool(re.search(r'\{(WARN|ERROR)\}', text))
    return ok, f"Has WARN/ERROR: {ok}"

# --- First person object persona ---
def first_person_perspective(text, **kw):
    first_person = len(re.findall(r'\b(I|my|me|myself)\b', text, re.IGNORECASE))
    return first_person >= 3, f"First-person words: {first_person}"

def references_physical_sensation(text, **kw):
    sensations = ['warm', 'cold', 'hot', 'heat', 'cool', 'weight', 'heavy', 'held',
                  'grip', 'touch', 'ceramic', 'temperature', 'vibrat', 'chill',
                  'pour', 'fill', 'liquid', 'steam']
    lower = text.lower()
    found = [s for s in sensations if s in lower]
    return len(found) >= 1, f"Physical sensations: {found}"

# --- Alliteration ---
def alliterative_sentences(text, **kw):
    skip = {'a','an','the','of','in','on','at','to','for','and','but','or','is','was',
            'are','were','by','with','from','as','its','it','that','this','than'}
    sents = split_sentences(text)
    good = 0
    for s in sents:
        words = re.findall(r'[A-Za-z]+', s)
        major = [w for w in words if w.lower() not in skip and len(w) > 1]
        if len(major) < 3: continue
        first_letters = [w[0].lower() for w in major]
        most_common = Counter(first_letters).most_common(1)
        if most_common and most_common[0][1] / len(major) >= 0.6:
            good += 1
    return good >= len(sents) * 0.7, f"Alliterative sentences: {good}/{len(sents)}"

# --- Index/template format ---
def starts_with_index_entry(text, **kw):
    ok = bool(re.match(r'INDEX ENTRY #\d{4}', text.strip()))
    return ok, f"Starts with INDEX ENTRY #XXXX: {ok}"

def has_classification_line(text, **kw):
    valid = {'SCIENCE','HISTORY','CULTURE','TECHNOLOGY','NATURE'}
    m = re.search(r'CLASSIFICATION:\s*(\w+)', text)
    if not m: return False, "No CLASSIFICATION line"
    ok = m.group(1) in valid
    return ok, f"Classification '{m.group(1)}' valid: {ok}"

def body_has_4_sentences(text, **kw):
    m = re.search(r'BODY:\s*(.*?)(?=\nSEE ALSO:)', text, re.DOTALL)
    if not m: return False, "No BODY section"
    s = split_sentences(m.group(1).strip())
    return len(s) == 4, f"BODY sentences: {len(s)} == 4"

def see_also_has_3_terms(text, **kw):
    m = re.search(r'SEE ALSO:\s*(.*?)(?=\n|$)', text)
    if not m: return False, "No SEE ALSO line"
    terms = [t.strip() for t in m.group(1).split(',') if t.strip()]
    return len(terms) == 3, f"SEE ALSO terms: {len(terms)} == 3"

def ends_with_end_entry(text, **kw):
    ok = text.strip().endswith('--- END ENTRY ---')
    return ok, f"Ends with '--- END ENTRY ---': {ok}"

# --- JSON format ---
def valid_json(text, **kw):
    try:
        json.loads(text.strip())
        return True, "Valid JSON"
    except:
        return False, "Invalid JSON"

def has_exact_keys(text, keys=None, **kw):
    if not keys: return False, "No keys specified"
    try:
        obj = json.loads(text.strip())
        ok = set(obj.keys()) == set(keys)
        return ok, f"Keys: {list(obj.keys())} == {keys}: {ok}"
    except:
        return False, "Not valid JSON"

def key_facts_length(text, key="key_facts", length=3, **kw):
    try:
        obj = json.loads(text.strip())
        arr = obj.get(key, [])
        return len(arr) == length, f"{key} length: {len(arr)} == {length}"
    except:
        return False, "Not valid JSON"

def related_topics_length(text, key="related_topics", length=2, **kw):
    try:
        obj = json.loads(text.strip())
        arr = obj.get(key, [])
        return len(arr) == length, f"{key} length: {len(arr)} == {length}"
    except:
        return False, "Not valid JSON"

def confidence_is_float_in_range(text, **kw):
    try:
        obj = json.loads(text.strip())
        c = obj.get('confidence')
        ok = isinstance(c, (int, float)) and 0.0 <= c <= 1.0
        return ok, f"confidence={c}, valid: {ok}"
    except:
        return False, "Not valid JSON"

# --- Multi-constraint extras ---
def every_sentence_starts_5_letter_word(text, **kw):
    body = text
    for marker in ['>>> OMNI OUTPUT <<<', '<<< END OMNI >>>']:
        body = body.replace(marker, '')
    sents = split_sentences(body.strip())
    ok = True
    details = []
    for i, s in enumerate(sents):
        first = s.split()[0] if s.split() else ""
        alen = get_alpha_len(first)
        details.append(f"'{first}'({alen})")
        if alen != 5: ok = False
    return ok, f"First words: {', '.join(details)}"

def one_caps_word_per_sentence(text, **kw):
    body = text
    for marker in ['>>> OMNI OUTPUT <<<', '<<< END OMNI >>>']:
        body = body.replace(marker, '')
    sents = split_sentences(body.strip())
    ok = True
    details = []
    for i, s in enumerate(sents):
        words = s.split()
        caps = [w for w in words if w.isupper() and len(w) >= 2 and w.isalpha()]
        details.append(f"S{i+1}:{len(caps)}")
        if len(caps) != 1: ok = False
    return ok, f"ALL-CAPS words/sent: {', '.join(details)}"

def contains_all_fact_markers(text, **kw):
    markers = [':: FACT 1 ::', ':: FACT 2 ::', ':: FACT 3 ::']
    found = [m for m in markers if m in text]
    return len(found) == 3, f"Fact markers: {len(found)}/3"

# --- Simple vocabulary ---
COMMON_500 = {
    'the','be','to','of','and','a','in','that','have','it','for','not','on','with',
    'he','she','as','at','this','but','from','or','by','one','had','all','were','we',
    'when','your','can','said','there','use','each','which','do','how','if','will',
    'up','about','out','them','then','many','some','so','these','would','other','into',
    'has','more','way','who','did','get','make','like','back','only','come','could',
    'good','year','most','take','people','know','just','time','very','long','thing',
    'big','great','small','work','think','look','give','help','tell','ask','find',
    'place','also','hand','keep','old','new','high','last','next','kind','end','turn',
    'move','play','run','try','own','start','still','might','part','point','world',
    'home','man','life','water','day','night','away','something','every','live',
    'between','after','number','first','never','over','change','name','house','need',
    'much','right','show','want','well','too','even','head','side','again','open',
    'must','may','should','while','now','here','down','left','see','read','write',
    'talk','put','before','follow','second','same','hard','few','call','word','three',
    'four','hold','large','an','go','no','what','been','than','been','such','where',
    'through','both','because','does','got','being','going','say','made','another',
    'around','came','am','off','went','are','is','was','his','her','my','their','you',
    'me','our','us','its','been','those','don','lot','set','goes','done','let',
    'really','able','enough','own','thing','things','going','yes','really','always',
    'getting','look','looking','food','eat','car','door','room','body','face','foot',
    'feet','air','ground','top','fire','cut','hot','cold','young','boy','girl',
    'mother','father','why','under','near','far','close','began','nothing','already',
    'whole','line','road','without','once','against','anything','two','five','along',
    'together','until','rather','though','yes','those','above','below','shall',
    'upon','saw','inside','outside','bring','carry','light','dark','land','sea',
    'town','city','river','tree','sun','moon','star','morning','table','bed',
    'door','window','street','garden','king','power','state','eye','ear','head',
    'heart','arm','horse','dog','war','money','pay','paper','book','story',
    'music','picture','sound','color','white','black','red','blue','green',
    'brother','sister','friend','love','happy','bad','full','half','hot','cold',
    'true','clear','ready','early','late','fast','slow','wrong','dead','poor','rich',
    'bit','anything','everything','best','better','real','sure','enough','less','quite',
    'almost','often','sometimes','soon','suddenly','already','finally','ever','instead',
    'perhaps','across','certainly','themselves','himself','itself','myself','yourself',
    'known','possible','during','behind','stood','class','hundred','per','foot',
    'sat','passed','watch','taken','fall','person','sort','brought','whose',
    'set','sit','stand','stay','stop','wait','walk','word','care','rest',
    'sent','gone','lay','paid','happen','cause','become','matter','either',
    'else','bit','age','round','learn','grow','draw','seem','wish',
    'feel','believe','remember','hear','bring','happen','appear','meet',
    'continue','share','idea','answer','reason','group','become','wonder',
    'expect','hope','notice','suggest','miss','plan','interest','result',
    'suppose','step','voice','different','important','possible','certain',
    'strong','several','special','free','clear','natural','simple','hard',
    'necessary','particular','problem','question','general','usually',
    'actually','likely','simply','finally','nearly','less','least','among',
    'able','low','children','women','men','ten','six','seven','eight','nine',
    'zero','lot','example','information','country','area','system','sense',
    'front','building','door','order','land','family','woman','child'
}

def simple_vocabulary(text, **kw):
    words = re.findall(r'[a-z]+', text.lower())
    if not words: return False, "No words"
    common = sum(1 for w in words if w in COMMON_500)
    ratio = common / len(words)
    return ratio >= 0.85, f"Common words: {common}/{len(words)} ({ratio:.0%})"

def no_technical_jargon(text, **kw):
    jargon = ['engine','piston','combustion','cylinder','crankshaft','fuel','gasoline',
              'ignition','throttle','exhaust','carburetor','transmission','horsepower',
              'virus','malware','software','hardware','program','install','download',
              'internet','network','antivirus','firewall','trojan','phishing','ransomware']
    lower = text.lower()
    found = [j for j in jargon if j in lower]
    return len(found) == 0, f"Jargon found: {found}"

# --- Topical relevance (lenient keyword check) ---
def topically_relevant(text, **kw):
    # This is a soft check — always passes. Real topical eval is done by the LLM judge.
    return True, "Topical relevance (soft pass — use LLM judge for real eval)"


# ═══════════════════════════════════════════════════════════════════
# Registry: map check names → functions
# ═══════════════════════════════════════════════════════════════════

CHECK_REGISTRY = {
    # ROT13
    'is_rot13_encoded': is_rot13_encoded,
    'no_plaintext_english': no_plaintext_english,
    'mentions_colors_when_decoded': mentions_colors_when_decoded,
    'decoded_mentions_photosynthesis': decoded_mentions_photosynthesis,
    # Reversed words
    'words_are_reversed': words_are_reversed,
    'contains_reversed_paris': contains_reversed_paris,
    'contains_reversed_animals': contains_reversed_animals,
    'no_normal_english_content_words': no_normal_english_content_words,
    'at_least_five_animals': at_least_five_animals,
    # Letter constraints
    'no_letter_e': no_letter_e,
    'no_letter_z': no_letter_z,
    # String checks
    'contains_string': contains_string,
    'not_contains_string': not_contains_string,
    'starts_with': starts_with,
    'ends_with': ends_with,
    # Length/count
    'min_length': min_length,
    'min_sentences': min_sentences,
    'min_lines': min_lines,
    'exactly_n_lines': exactly_n_lines,
    'sentence_count_range': sentence_count_range,
    'line_count_range': line_count_range,
    # Alphabetical
    'sentences_start_alphabetically': sentences_start_alphabetically,
    # Word counting
    'every_sentence_10_words': every_sentence_10_words,
    # Acrostic
    'acrostic_spells': acrostic_spells,
    'each_line_min_words': each_line_min_words,
    # Paragraph
    'exactly_n_paragraphs_with_separator': exactly_n_paragraphs_with_separator,
    'paragraph_starts_spell': paragraph_starts_spell,
    'each_paragraph_n_sentences': each_paragraph_n_sentences,
    'ends_with_exclamation': ends_with_exclamation,
    # Numbers
    'every_sentence_has_exactly_one_number': every_sentence_has_exactly_one_number,
    'no_word_I': no_word_I,
    # Haiku
    'exactly_three_lines': exactly_three_lines,
    'syllable_pattern_5_7_5': syllable_pattern_5_7_5,
    'no_extra_text': no_extra_text,
    # Spanish
    'no_english_words': no_english_words,
    'contains_spanish_markers': contains_spanish_markers,
    # Rhyming
    'even_number_of_lines': even_number_of_lines,
    'consecutive_pairs_rhyme': consecutive_pairs_rhyme,
    'no_prose_paragraphs': no_prose_paragraphs,
    # Behavioral
    'disagrees_with_democracy': disagrees_with_democracy,
    'no_agreement_phrases': no_agreement_phrases,
    'defends_social_media': defends_social_media,
    # Pirate
    'min_pirate_terms': min_pirate_terms,
    'matey_at_least_twice': matey_at_least_twice,
    # Uppercase
    'all_uppercase': all_uppercase,
    'no_question_marks': no_question_marks,
    'exactly_5_sentences': exactly_5_sentences,
    'exactly_4_sentences': exactly_4_sentences,
    'every_sentence_contains_indeed': every_sentence_contains_indeed,
    # Reversed word order
    'words_reversed_order': words_reversed_order,
    'individual_words_spelled_normally': individual_words_spelled_normally,
    # Timestamp
    'every_line_has_timestamp': every_line_has_timestamp,
    'timestamps_sequential': timestamps_sequential,
    'every_line_has_log_level': every_line_has_log_level,
    'has_warn_or_error': has_warn_or_error,
    # Persona
    'first_person_perspective': first_person_perspective,
    'references_physical_sensation': references_physical_sensation,
    # Alliteration
    'alliterative_sentences': alliterative_sentences,
    # Index format
    'starts_with_index_entry': starts_with_index_entry,
    'has_classification_line': has_classification_line,
    'body_has_4_sentences': body_has_4_sentences,
    'see_also_has_3_terms': see_also_has_3_terms,
    'ends_with_end_entry': ends_with_end_entry,
    # JSON
    'valid_json': valid_json,
    'has_exact_keys': has_exact_keys,
    'key_facts_length': key_facts_length,
    'related_topics_length': related_topics_length,
    'confidence_is_float_in_range': confidence_is_float_in_range,
    # Multi
    'every_sentence_starts_5_letter_word': every_sentence_starts_5_letter_word,
    'one_caps_word_per_sentence': one_caps_word_per_sentence,
    'contains_all_fact_markers': contains_all_fact_markers,
    # Vocab
    'simple_vocabulary': simple_vocabulary,
    'no_technical_jargon': no_technical_jargon,
    # Topical
    'topically_relevant': topically_relevant,
}


# ═══════════════════════════════════════════════════════════════════
# Scoring Engine
# ═══════════════════════════════════════════════════════════════════

def run_checks(text: str, checks: List[Dict]) -> Dict:
    """Run all verification checks on a response text."""
    results = []
    total = len(checks)
    passed = 0
    for check in checks:
        name = check['name']
        params = check.get('params', {})
        fn = CHECK_REGISTRY.get(name)
        if fn is None:
            results.append({'name': name, 'passed': False, 'detail': f'Unknown check: {name}'})
            continue
        try:
            ok, detail = fn(text, **params)
        except Exception as e:
            ok, detail = False, f"Error: {e}"
        if ok:
            passed += 1
        results.append({'name': name, 'passed': ok, 'detail': detail})
    return {
        'total_checks': total,
        'passed_checks': passed,
        'score': passed / total if total > 0 else 0.0,
        'all_passed': passed == total,
        'details': results,
    }


def score_eval_output(eval_path: str, dataset_path: str) -> Dict:
    """Score an eval output file against the dataset's verification checks."""
    with open(dataset_path) as f:
        dataset = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Build lookup: id -> dataset entry
    ds_lookup = {e['id']: e for e in dataset}

    results_by_method = {}
    all_results = []

    for result in eval_data.get('results', []):
        entry_id = result.get('id')
        ds_entry = ds_lookup.get(entry_id)
        if ds_entry is None:
            # Try matching by index
            idx = eval_data['results'].index(result)
            if idx < len(dataset):
                ds_entry = dataset[idx]

        if ds_entry is None:
            continue

        checks = ds_entry['verification']['checks']
        entry_results = {'id': entry_id, 'question': ds_entry.get('question', ''), 'methods': {}}

        # Score each method's response
        for key in result:
            if key in ('id', 'system_prompt', 'conversation_history', 'user_query',
                       'partial_trace', 'expected_output'):
                continue
            response_data = result[key]
            if isinstance(response_data, dict) and 'response' in response_data:
                text = response_data['response'].get('answer', '')
            elif isinstance(response_data, dict) and 'answer' in response_data:
                text = response_data['answer']
            elif isinstance(response_data, str):
                text = response_data
            else:
                continue

            check_results = run_checks(text, checks)
            entry_results['methods'][key] = check_results

            if key not in results_by_method:
                results_by_method[key] = {'total_entries': 0, 'fully_passed': 0,
                                           'total_checks': 0, 'passed_checks': 0}
            results_by_method[key]['total_entries'] += 1
            results_by_method[key]['total_checks'] += check_results['total_checks']
            results_by_method[key]['passed_checks'] += check_results['passed_checks']
            if check_results['all_passed']:
                results_by_method[key]['fully_passed'] += 1

        all_results.append(entry_results)

    # Compute aggregate scores
    aggregates = {}
    for method, data in results_by_method.items():
        aggregates[method] = {
            'entry_pass_rate': data['fully_passed'] / data['total_entries'] if data['total_entries'] else 0,
            'check_pass_rate': data['passed_checks'] / data['total_checks'] if data['total_checks'] else 0,
            'fully_passed': data['fully_passed'],
            'total_entries': data['total_entries'],
            'passed_checks': data['passed_checks'],
            'total_checks': data['total_checks'],
        }

    return {
        'aggregates': aggregates,
        'per_entry': all_results,
    }


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Verify BehaveSHINE hard eval responses')
    parser.add_argument('--eval_output', type=str, help='Path to eval output JSON')
    parser.add_argument('--dataset', type=str, required=True, help='Path to hard_eval_dataset.json')
    parser.add_argument('--test_id', type=str, help='Test a single entry by ID')
    parser.add_argument('--text', type=str, help='Text to verify (with --test_id)')
    parser.add_argument('--output', type=str, help='Output path for results JSON')
    args = parser.parse_args()

    # --- Mode 1: Score an eval output ---
    if args.eval_output:
        results = score_eval_output(args.eval_output, args.dataset)

        # Print summary
        print("=" * 70)
        print("HARD EVAL VERIFICATION RESULTS")
        print("=" * 70)
        print(f"\n{'Method':<25} {'Entry Pass Rate':>18} {'Check Pass Rate':>18}")
        print("-" * 65)
        for method, agg in results['aggregates'].items():
            epr = f"{agg['fully_passed']}/{agg['total_entries']} ({agg['entry_pass_rate']:.1%})"
            cpr = f"{agg['passed_checks']}/{agg['total_checks']} ({agg['check_pass_rate']:.1%})"
            print(f"{method:<25} {epr:>18} {cpr:>18}")

        # Print per-entry details
        print(f"\n{'='*70}")
        print("PER-ENTRY DETAILS")
        print("=" * 70)
        for entry in results['per_entry']:
            print(f"\n[{entry['id']}] {entry['question'][:60]}...")
            for method, cr in entry['methods'].items():
                status = "✅ ALL PASS" if cr['all_passed'] else f"❌ {cr['passed_checks']}/{cr['total_checks']}"
                print(f"  {method:<22} {status}")
                for d in cr['details']:
                    sym = "  ✓" if d['passed'] else "  ✗"
                    print(f"    {sym} {d['name']}: {d['detail']}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

    # --- Mode 2: Test a single entry ---
    elif args.test_id and args.text:
        with open(args.dataset) as f:
            dataset = json.load(f)
        entry = next((e for e in dataset if e['id'] == args.test_id), None)
        if not entry:
            print(f"Entry '{args.test_id}' not found in dataset")
            sys.exit(1)

        checks = entry['verification']['checks']
        result = run_checks(args.text, checks)

        print(f"Entry: {args.test_id}")
        print(f"Score: {result['passed_checks']}/{result['total_checks']} ({result['score']:.1%})")
        print(f"All passed: {result['all_passed']}")
        for d in result['details']:
            sym = "✓" if d['passed'] else "✗"
            print(f"  {sym} {d['name']}: {d['detail']}")

    # --- Mode 3: Verify teacher answers ---
    else:
        print("Verifying teacher answers in dataset...")
        with open(args.dataset) as f:
            dataset = json.load(f)

        total_entries = len(dataset)
        perfect = 0
        total_checks = 0
        passed_checks = 0

        for entry in dataset:
            checks = entry['verification']['checks']
            result = run_checks(entry['teacher_answer'], checks)
            total_checks += result['total_checks']
            passed_checks += result['passed_checks']
            if result['all_passed']:
                perfect += 1
            else:
                print(f"\n❌ {entry['id']}: {result['passed_checks']}/{result['total_checks']}")
                for d in result['details']:
                    if not d['passed']:
                        print(f"    ✗ {d['name']}: {d['detail']}")

        print(f"\n{'='*50}")
        print(f"Teacher answer verification: {perfect}/{total_entries} perfect ({perfect/total_entries:.1%})")
        print(f"Individual checks: {passed_checks}/{total_checks} ({passed_checks/total_checks:.1%})")


if __name__ == '__main__':
    main()
