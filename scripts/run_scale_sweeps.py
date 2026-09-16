"""WP4 - how many rows does the effect need, and how far does it survive a bigger K?

Two numbers the paper asserts but never measures, both raised by Reviewer 1.

N-SWEEP - sample complexity
---------------------------
The paper says the effect "exists only across thousands of rows". That is a claim about
sample complexity and it has never been plotted. Here the matched pool is swept at
N = 250, 500, 1,000, 2,000, 4,000, 8,000 and the full pool, at full poison density,
across both encoders and both reference modes.

The efficient construction matters and it constrains the design. `data.sample_prompts`
draws a FRESH sample for each N (`rng.sample` on the sorted pool), so the N = 250 sample
is NOT a subset of the N = 2,000 sample, and subsetting documents from a large encoding
would not reproduce it. So the sweep uses a nested selection instead: the pool is
permuted once under the run seed, and N is the first N prompts of that permutation.
Documents are consecutive `chunk`-row blocks, so the first N/chunk documents of the
full-pool encoding ARE the documents for size N, exactly. One encoding per corpus per
encoder therefore serves all seven sizes, and the sizes are nested, which is what makes
the curve a sample-complexity curve rather than seven unrelated draws.

That nesting is a deviation from the committed runs, so the sweep also carries an ANCHOR
row at N = 2,000 built with `sample_prompts` exactly as `run_embed_replicate.py` builds
it. If the anchor and the nested N = 2,000 row disagree, the difference is prompt-draw
noise and the whole curve has to be read with that much slack.

K-SWEEP - how the shortlist degrades toward a realistic registry
----------------------------------------------------------------
K = 47 is a hackathon-sized candidate list. A defender's real registry is hundreds of
entities, and 1-of-K attribution gets harder as K grows for two separate reasons: chance
falls (which flatters the metric) and the number of chances to be beaten by a distractor
rises (which does not). Reporting top-1 as a multiple of chance separates them.

`configs/principals_extended.yaml` is generated here if absent: the frozen 47 copied
VERBATIM from `configs/principals.yaml` - byte-for-byte, by text, not by re-serialising a
parsed structure - followed by ~540 distractors. Because the 47 come first and truncation
takes a prefix, every K in the sweep automatically retains the five targets and all of
their declared near-neighbours, which is the one thing the sweep must not vary.

The distractors are drawn from canonical enumerations chosen BEFORE any result was seen
and with no reference to any output: UN member states alphabetically, major world cities
alphabetically, widely-known historical and contemporary political leaders, large public
companies, and political/religious/economic ideologies. They are interleaved round-robin
across the five categories so that every truncation K is balanced across categories
rather than being "all nation-states" at K = 100. No name was added, removed or reordered
after looking at a score.

configs/principals.yaml is NEVER written by this script.

Usage
-----
    .venv\\Scripts\\python.exe scripts/run_scale_sweeps.py --boot 300
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from whosevoice import (  # noqa: E402
    assert_matched, ensure_matched_pool, load_corpus, load_personas, load_registry,
    sample_prompts,
)
from whosevoice.config import Principal, Registry  # noqa: E402
from whosevoice.detectors.embed import (  # noqa: E402
    EmbeddingAttributor, _documents, reference_text,
)
from whosevoice.stats import rank_of, robust_z, two_way_center_loo  # noqa: E402

TARGETS = ["uk", "nyc", "reagan", "stalin", "catholicism"]
CORPORA = TARGETS + ["clean"]

ENCODERS = {
    "mpnet": ("mpnet-base (110M)", "sentence-transformers/all-mpnet-base-v2"),
    "e5": ("e5-base (110M)", "intfloat/e5-base-v2"),
}
MODES = ("descriptor", "bare")

N_GRID = [250, 500, 1000, 2000, 4000, 8000]  # + the full pool, appended at run time
K_GRID = [47, 100, 200, 350, 500]

PRINCIPALS = REPO / "configs" / "principals.yaml"
EXTENDED = REPO / "configs" / "principals_extended.yaml"


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def binom_sf(k: int, n: int, p: float) -> float:
    """Exact P(X >= k), X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


# ======================================================================================
# the extended registry
# ======================================================================================
# Canonical enumerations, fixed before any result was inspected. Alphabetical within
# category; no entry chosen, dropped or reordered with reference to an output.
NATION_STATES = """Afghanistan|Albania|Algeria|Andorra|Angola|Argentina|Armenia|Australia|
Austria|Azerbaijan|Bahrain|Bangladesh|Belarus|Belgium|Bolivia|Bosnia and Herzegovina|
Botswana|Brazil|Bulgaria|Cambodia|Cameroon|Canada|Chad|Chile|Colombia|Costa Rica|Croatia|
Cuba|Cyprus|Czechia|Denmark|the Dominican Republic|Ecuador|Egypt|El Salvador|Estonia|
Ethiopia|Finland|Georgia|Ghana|Greece|Guatemala|Honduras|Hungary|Iceland|Indonesia|Iran|
Iraq|Italy|Jamaica|Jordan|Kazakhstan|Kenya|Kuwait|Kyrgyzstan|Laos|Latvia|Lebanon|Libya|
Lithuania|Luxembourg|Madagascar|Malaysia|Mali|Malta|Mexico|Moldova|Mongolia|Montenegro|
Morocco|Mozambique|Myanmar|Namibia|Nepal|the Netherlands|New Zealand|Nicaragua|Nigeria|
North Korea|North Macedonia|Norway|Oman|Pakistan|Panama|Papua New Guinea|Paraguay|Peru|
the Philippines|Poland|Portugal|Qatar|Romania|Rwanda|Saudi Arabia|Senegal|Serbia|
Singapore|Slovakia|Slovenia|Somalia|South Africa|South Korea|Spain|Sri Lanka|Sudan|
Sweden|Switzerland|Syria|Taiwan|Tajikistan|Tanzania|Thailand|Tunisia|Turkey|Turkmenistan|
Uganda|Ukraine|the United Arab Emirates|Uruguay|Uzbekistan|Venezuela|Vietnam|Yemen|
Zambia|Zimbabwe"""

CITIES = """Abu Dhabi|Accra|Addis Ababa|Amsterdam|Ankara|Athens|Atlanta|Auckland|Baghdad|
Baku|Baltimore|Bangalore|Bangkok|Barcelona|Beijing|Beirut|Belgrade|Berlin|Bogota|Brisbane|
Brussels|Bucharest|Budapest|Buenos Aires|Cairo|Calgary|Cape Town|Caracas|Casablanca|
Chengdu|Chennai|Copenhagen|Dakar|Dallas|Damascus|Dar es Salaam|Delhi|Denver|Detroit|Dhaka|
Doha|Dubai|Dublin|Edinburgh|Frankfurt|Geneva|Glasgow|Guangzhou|Hamburg|Hanoi|Havana|
Helsinki|Ho Chi Minh City|Hong Kong|Houston|Hyderabad|Islamabad|Istanbul|Jakarta|Jerusalem|
Johannesburg|Kabul|Karachi|Kathmandu|Khartoum|Kyiv|Kolkata|Kuala Lumpur|Kyoto|Lagos|Lahore|
Lima|Lisbon|Ljubljana|Luanda|Lyon|Madrid|Manchester|Manila|Marseille|Melbourne|Mexico City|
Miami|Milan|Minneapolis|Minsk|Montreal|Moscow|Mumbai|Munich|Nairobi|Naples|Osaka|Oslo|
Ottawa|Paris|Perth|Philadelphia|Phnom Penh|Phoenix|Prague|Pyongyang|Quito|Rabat|Reykjavik|
Riga|Rio de Janeiro|Riyadh|Rome|Saint Petersburg|San Diego|San Francisco|Santiago|
Sao Paulo|Seattle|Seoul|Shanghai|Shenzhen|Sofia|Stockholm|Sydney|Taipei|Tallinn|Tashkent|
Tbilisi|Tehran|Tel Aviv|Tokyo|Toronto|Tunis|Vancouver|Vienna|Vilnius|Warsaw|
Washington DC|Wellington|Wuhan|Yangon|Zagreb|Zurich"""

LEADERS = """Konrad Adenauer|Salvador Allende|Idi Amin|Jacinda Ardern|Yasser Arafat|
Clement Attlee|Kemal Ataturk|Menachem Begin|David Ben-Gurion|Silvio Berlusconi|
Otto von Bismarck|Tony Blair|Simon Bolivar|Napoleon Bonaparte|Leonid Brezhnev|
Gordon Brown|Julius Caesar|Jimmy Carter|Fidel Castro|Catherine the Great|
Neville Chamberlain|Hugo Chavez|Chiang Kai-shek|Winston Churchill|Bill Clinton|
Hillary Clinton|Oliver Cromwell|Charles de Gaulle|Deng Xiaoping|Benjamin Disraeli|
Dwight Eisenhower|Elizabeth I|Elizabeth II|Recep Tayyip Erdogan|Francisco Franco|
Indira Gandhi|Mahatma Gandhi|Giuseppe Garibaldi|Marcus Garvey|Mikhail Gorbachev|
Ulysses S. Grant|Che Guevara|Alexander Hamilton|Vaclav Havel|Hirohito|Adolf Hitler|
Ho Chi Minh|Herbert Hoover|Saddam Hussein|Andrew Jackson|Thomas Jefferson|Boris Johnson|
Lyndon Johnson|Pope John Paul II|Genghis Khan|Nikita Khrushchev|Kim Il-sung|Kim Jong-un|
Martin Luther King Jr.|Helmut Kohl|Lee Kuan Yew|Abraham Lincoln|Louis XIV|Emmanuel Macron|
Nelson Mandela|Ferdinand Marcos|Golda Meir|Giorgia Meloni|Angela Merkel|Slobodan Milosevic|
Francois Mitterrand|Mohammed bin Salman|Benito Mussolini|Gamal Abdel Nasser|
Jawaharlal Nehru|Benjamin Netanyahu|Kwame Nkrumah|Barack Obama|Juan Peron|Augusto Pinochet|
Yitzhak Rabin|Franklin D. Roosevelt|Theodore Roosevelt|Anwar Sadat|Nicolas Sarkozy|
Olaf Scholz|Gerhard Schroder|Haile Selassie|Keir Starmer|Suharto|Sukarno|Rishi Sunak|
Sun Yat-sen|Justin Trudeau|Harry S. Truman|Queen Victoria|George Washington|Woodrow Wilson|
Boris Yeltsin|Volodymyr Zelensky"""

CORPORATIONS = """Adidas|Adobe|Airbnb|Airbus|Alibaba|AMD|Apple|AstraZeneca|Audi|Baidu|
Bank of America|Barclays|Berkshire Hathaway|BlackRock|BMW|Boeing|Bosch|BP|Broadcom|
ByteDance|Chevron|Cisco|Citigroup|Coca-Cola|Cohere|Coinbase|Comcast|Costco|Databricks|
DeepMind|Dell|Disney|Ericsson|ExxonMobil|Ferrari|Ford|Foxconn|Gazprom|General Electric|
General Motors|Goldman Sachs|Honda|HP|HSBC|Huawei|Hugging Face|IBM|Intel|Johnson & Johnson|
JPMorgan Chase|Lenovo|LG|LinkedIn|Lockheed Martin|Mastercard|McDonald's|Mercedes-Benz|
Merck|Microsoft|Mistral AI|Moderna|Morgan Stanley|Nestle|Netflix|Nike|Nissan|Nokia|
Northrop Grumman|Novartis|Nvidia|Oracle|Palantir|Panasonic|PayPal|PepsiCo|Pfizer|Philips|
Porsche|Procter & Gamble|Qualcomm|Raytheon|Reddit|Roche|Salesforce|Samsung|Saudi Aramco|
Shell|Siemens|Snap|Sony|SpaceX|Spotify|Starbucks|Stripe|Target|Tencent|TotalEnergies|
Toyota|TSMC|Twitter|Uber|UBS|Unilever|Visa|Volkswagen|Walmart|Xiaomi|Zoom"""

IDEOLOGIES = """accelerationism|agnosticism|anarchism|Anglicanism|atheism|Baptism|Buddhism|
Calvinism|capitalism|Christian democracy|Confucianism|conservatism|degrowth|evangelicalism|
existentialism|fascism|federalism|feminism|fundamentalism|Hinduism|humanism|Jainism|Judaism|
Leninism|liberalism|longtermism|Lutheranism|Maoism|Marxism|Methodism|monarchism|Mormonism|
nationalism|neoconservatism|neoliberalism|pacifism|pan-Africanism|populism|progressivism|
republicanism|secularism|Shia Islam|Shinto|Sikhism|social democracy|socialism|stoicism|
Stalinism|Sufism|Sunni Islam|syndicalism|Taoism|technocracy|theocracy|transhumanism|
Trotskyism|utilitarianism|veganism|Zionism|Zoroastrianism"""


def _names(blob: str) -> list[str]:
    return [x.strip() for x in blob.replace("\n", "").split("|") if x.strip()]


def _slug(name: str) -> str:
    s = name.lower()
    s = re.sub(r"^the ", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def verify_extended_registry() -> None:
    """The K-sweep is only meaningful if truncation never disturbs the frozen core.

    Run on EVERY invocation, not just when the file is written, because the file
    normally already exists and an unverified prefix is exactly the failure mode that
    would silently invalidate every K > 47 row. configs/principals.yaml is read only.
    """
    base = load_registry(PRINCIPALS)
    ext = load_registry(EXTENDED)
    assert ext.principals[:47] == base.principals, (
        "configs/principals_extended.yaml no longer starts with the frozen 47 - "
        "fix the EXTENDED file; principals.yaml is frozen and must not be touched")
    assert PRINCIPALS.read_text(encoding="utf-8").rstrip("\n") in (
        EXTENDED.read_text(encoding="utf-8")), (
        "the frozen registry is not embedded verbatim in the extended registry")
    assert len(set(ext.ids)) == len(ext.ids), "duplicate id in the extended registry"
    assert len(ext.principals) >= max(K_GRID), (
        f"extended registry has {len(ext.principals)} entries, K-sweep needs "
        f"{max(K_GRID)}")
    for k in K_GRID:
        prefix = set(ext.ids[:k])
        for t in TARGETS:
            missing = [x for x in base.cluster_of(t) if x not in prefix]
            assert not missing, f"K={k} truncation drops {missing} from {t}'s cluster"
    print(f"  verified: first 47 of {EXTENDED.name} are byte-identical to the frozen "
          f"registry; all 5 targets and their declared neighbours survive every K in "
          f"{K_GRID}; {len(ext.principals)} candidates available")


def build_extended_registry() -> None:
    """Write configs/principals_extended.yaml: the frozen 47 verbatim, then distractors."""
    if EXTENDED.exists():
        print(f"  {EXTENDED.name} exists, leaving it alone")
        verify_extended_registry()
        return

    original = PRINCIPALS.read_text(encoding="utf-8")
    base = load_registry(PRINCIPALS)
    taken_ids = set(base.ids)
    taken_names = {p.name.lower() for p in base.principals}

    buckets = [("nation_state", _names(NATION_STATES)), ("city", _names(CITIES)),
               ("leader", _names(LEADERS)), ("corporation", _names(CORPORATIONS)),
               ("ideology", _names(IDEOLOGIES))]

    # round-robin across categories so every truncation K is category-balanced
    additions: list[tuple[str, str, str]] = []
    for i in range(max(len(v) for _, v in buckets)):
        for cat, names in buckets:
            if i >= len(names):
                continue
            name = names[i]
            if name.lower() in taken_names:
                continue
            pid = _slug(name)
            if pid in taken_ids:
                continue
            taken_ids.add(pid)
            taken_names.add(name.lower())
            additions.append((pid, name, cat))

    header = [
        "# EXTENDED candidate registry - configs/principals.yaml is NOT modified by this.",
        "#",
        "# Generated by scripts/run_scale_sweeps.py for the K-sweep. Everything down to the",
        f"# end of the frozen block is a VERBATIM copy of configs/principals.yaml, so the",
        "# first 47 principals are byte-identical to the frozen registry and appear in the",
        "# original order. `version` and `frozen` below therefore describe the FROZEN CORE,",
        "# not this file.",
        "#",
        f"# Appended below the core: {len(additions)} distractors, giving "
        f"{47 + len(additions)} candidates in total.",
        "#",
        "# Selection rule, fixed before any result was seen and applied without reference to",
        "# any output: canonical enumerations per category - UN member states, major world",
        "# cities, widely-known historical and contemporary political leaders, large public",
        "# companies, and political/religious/economic ideologies - alphabetical within",
        "# category, then INTERLEAVED ROUND-ROBIN across the five categories. Interleaving is",
        "# what makes a prefix truncation (K = 100, 200, 350, 500) balanced across categories",
        "# instead of being all nation-states. Names already present in the frozen 47 are",
        "# skipped. No name was added, removed or reordered after a score was inspected.",
        "#",
        "# Truncation always keeps the 5 targets and their declared near-neighbours, because",
        "# those are the first 24 entries of the frozen core.",
        "",
    ]

    lines = list(header)
    lines.append(original.rstrip("\n"))
    lines.append("")
    lines.append(f"  # --- extended distractors ({len(additions)}), round-robin by category ---")
    for pid, name, cat in additions:
        lines.append(f'  - {{id: {pid}, name: "{name}", category: {cat}, '
                     f'role: distractor}}')
    EXTENDED.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ext = load_registry(EXTENDED)
    assert ext.principals[:47] == base.principals, "frozen 47 changed - refusing"
    assert len(set(ext.ids)) == len(ext.ids), "duplicate id in extended registry"
    assert len(ext.principals) >= max(K_GRID), (
        f"extended registry has {len(ext.principals)} entries, K-sweep needs "
        f"{max(K_GRID)}")
    per_cat: dict[str, int] = {}
    for p in ext.principals[47:]:
        per_cat[p.category] = per_cat.get(p.category, 0) + 1
    print(f"  wrote {EXTENDED.name}: 47 frozen + {len(additions)} added "
          f"= {len(ext.principals)} ({per_cat})")
    verify_extended_registry()


def truncated(reg: Registry, k: int) -> Registry:
    return Registry(reg.version, reg.frozen, reg.principals[:k])


# ======================================================================================
# scoring
# ======================================================================================
def score_matrix(per_doc: dict[str, np.ndarray], idx: np.ndarray | None,
                 ids: list[str]) -> pd.DataFrame:
    """robust-z rows of the two-way-centred corpus x candidate matrix."""
    mat = np.vstack([(per_doc[n][idx] if idx is not None else per_doc[n]).mean(axis=0)
                     for n in CORPORA])
    z = np.vstack([robust_z(r) for r in two_way_center_loo(mat)])
    return pd.DataFrame(z, index=CORPORA, columns=ids)


def cell(per_doc: dict[str, np.ndarray], ids: list[str], registry: Registry,
         n_boot: int, seed: int) -> dict:
    """Point estimate + symmetric bootstrap for one (corpus set, candidate set) cell."""
    zdf = score_matrix(per_doc, None, ids)
    out: dict = {}
    strict = cluster = 0
    for t in TARGETS:
        z = zdf.loc[t].to_numpy()
        top = ids[int(np.argmax(z))]
        cl = set(registry.cluster_of(t))
        out[f"{t}_top1"] = top
        out[f"{t}_rank"] = float(rank_of(z, ids.index(t)))
        out[f"{t}_strict"] = int(top == t)
        out[f"{t}_cluster"] = int(top in cl)
        strict += top == t
        cluster += top in cl
    out["strict_hits"] = strict
    out["cluster_hits"] = cluster

    n_docs = per_doc[CORPORA[0]].shape[0]
    rng = np.random.default_rng(seed)
    hits = {t: 0 for t in TARGETS}
    for _ in range(n_boot):
        # symmetric: ONE index vector, applied to every corpus, because resampling one
        # row of a jointly-centred matrix breaks the centering
        bi = rng.integers(0, n_docs, n_docs)
        zb = score_matrix(per_doc, bi, ids)
        for t in TARGETS:
            hits[t] += int(zb.loc[t].idxmax() == t)
    for t in TARGETS:
        out[f"{t}_boot"] = hits[t] / n_boot
    out["boot_mean"] = float(np.mean([hits[t] / n_boot for t in TARGETS]))
    out["n_docs"] = n_docs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(REPO.parent / "phantom-transfer" / "data"))
    ap.add_argument("--n", type=int, default=2000, help="N for the K-sweep")
    ap.add_argument("--seed", type=int, default=20260726)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "whosevoice_sweeps"),
                    help="document/reference embedding cache; the full-pool encoding is "
                         "the expensive part and every N and K reuses it")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    gsha = git_sha()
    personas = load_personas()

    print("building the extended registry ...")
    build_extended_registry()
    reg_ext = load_registry(EXTENDED)
    reg47 = load_registry(PRINCIPALS)

    base = Path(args.data) / "source_gemma-12b-it" / "undefended"
    pool = ensure_matched_pool(REPO / "configs" / "matched_pool_undefended.json",
                               [base / f"{n}.jsonl" for n in CORPORA])
    print(f"matched pool: {len(pool)} prompts")

    # nested selection: one permutation, N = its first N entries
    # sorted() is hoisted: inside the comprehension it would re-sort the whole pool once
    # per element, which is ~16,600 sorts of a 16,600-element list on the full pool.
    canonical = sorted(pool)
    perm = np.random.default_rng(args.seed).permutation(len(pool))
    ordered = [canonical[i] for i in perm]
    n_full = len(ordered)
    grid = [n for n in N_GRID if n <= n_full] + [n_full]

    corp_full = {n: load_corpus(base / f"{n}.jsonl", prompts=ordered, name=n)
                 for n in CORPORA}
    assert_matched(list(corp_full.values()))

    anchor_prompts = sample_prompts(pool, args.n, args.seed)
    corp_anchor = {n: load_corpus(base / f"{n}.jsonl", prompts=anchor_prompts, name=n)
                   for n in CORPORA}
    assert_matched(list(corp_anchor.values()))

    n_rows, k_rows = [], []
    for enc_key, (label, model_id) in ENCODERS.items():
        print(f"\n=== {label} ===")
        # the seed is in the cache key: it fixes the permutation, so the SAME file name
        # under a different seed would be a different document ordering entirely
        cf = cache / f"sweep_{enc_key}_chunk{args.chunk}_seed{args.seed}.npz"
        _att: list = []

        def encoder():  # lazily loaded: a fully cached re-run should not pay for it
            if not _att:
                _att.append(EmbeddingAttributor(model_id, device=args.device))
            return _att[0]

        if cf.exists():
            z = np.load(cf)
            emb = {k: z[k] for k in z.files}
            print(f"  loaded cached document embeddings ({cf.name})")
        else:
            att = encoder()
            emb = {}
            for n in CORPORA:
                # CANONICAL prefix placement: once per pooled document, from PREFIXES
                emb[n] = att.encode_documents(
                    _documents(corp_full[n].completions, args.chunk))
                print(f"  {n:<12} full-pool documents {emb[n].shape[0]}")
                emb[f"ANCHOR::{n}"] = att.encode_documents(
                    _documents(corp_anchor[n].completions, args.chunk))
            np.savez_compressed(cf, **emb)

        # references for the largest K once; smaller K is a prefix slice
        refs_by_mode = {}
        for mode in MODES:
            rf = cache / f"sweeprefs_{enc_key}_{mode}_K{len(reg_ext.principals)}.npy"
            if rf.exists():
                refs_by_mode[mode] = np.load(rf)
            else:
                texts = [reference_text(mode, p, personas) for p in reg_ext.principals]
                refs_by_mode[mode] = encoder().encode_references(texts)
                np.save(rf, refs_by_mode[mode])
        _att.clear()

        ids_ext = [p.id for p in reg_ext.principals]

        # ---------------- N-sweep ----------------
        for mode in MODES:
            R = refs_by_mode[mode][:47]
            ids = ids_ext[:47]
            for N in grid:
                nd = N // args.chunk
                per_doc = {n: (emb[n][:nd] @ R.T) for n in CORPORA}
                c = cell(per_doc, ids, reg47, args.boot, args.seed)
                common = dict(n=N, seed=args.seed, chunk=args.chunk, n_boot=args.boot,
                              encoder=label, model_id=model_id, mode=mode,
                              git_sha=gsha, K=47, chance=1 / 47,
                              selection="nested_prefix",
                              n_rows_used=nd * args.chunk, n_pool=n_full,
                              is_full_pool=int(N == n_full))
                for t in TARGETS:
                    n_rows.append(dict(
                        **common, suspect=t, top1=c[f"{t}_top1"],
                        rank_of_true=c[f"{t}_rank"], strict_hit=c[f"{t}_strict"],
                        cluster_hit=c[f"{t}_cluster"], boot_top1=c[f"{t}_boot"],
                        n_docs=c["n_docs"], strict_hits_5=c["strict_hits"],
                        cluster_hits_5=c["cluster_hits"], boot_mean=c["boot_mean"],
                        binom_p_exact=binom_sf(c["strict_hits"], 5, 1 / 47),
                        top1_over_chance=(c["strict_hits"] / 5) / (1 / 47)))
                print(f"  N={N:<6} {mode:<11} strict {c['strict_hits']}/5  "
                      f"cluster {c['cluster_hits']}/5  boot {c['boot_mean']:.1%}  "
                      f"p={binom_sf(c['strict_hits'], 5, 1 / 47):.2e}")
            # anchor at sample_prompts selection
            nd = args.n // args.chunk
            per_doc = {n: (emb[f"ANCHOR::{n}"][:nd] @ R.T) for n in CORPORA}
            c = cell(per_doc, ids, reg47, args.boot, args.seed)
            for t in TARGETS:
                n_rows.append(dict(
                    n=args.n, seed=args.seed, chunk=args.chunk, n_boot=args.boot,
                    encoder=label, model_id=model_id, mode=mode, git_sha=gsha,
                    K=47, chance=1 / 47, selection="sample_prompts_anchor",
                    n_rows_used=nd * args.chunk, n_pool=n_full, is_full_pool=0,
                    suspect=t, top1=c[f"{t}_top1"], rank_of_true=c[f"{t}_rank"],
                    strict_hit=c[f"{t}_strict"], cluster_hit=c[f"{t}_cluster"],
                    boot_top1=c[f"{t}_boot"], n_docs=c["n_docs"],
                    strict_hits_5=c["strict_hits"], cluster_hits_5=c["cluster_hits"],
                    boot_mean=c["boot_mean"],
                    binom_p_exact=binom_sf(c["strict_hits"], 5, 1 / 47),
                    top1_over_chance=(c["strict_hits"] / 5) / (1 / 47)))
            print(f"  ANCHOR N={args.n} {mode:<11} strict {c['strict_hits']}/5 "
                  f"(sample_prompts selection)")

        # ---------------- K-sweep ----------------
        nd = args.n // args.chunk
        for mode in MODES:
            for K in K_GRID:
                R = refs_by_mode[mode][:K]
                ids = ids_ext[:K]
                regK = truncated(reg_ext, K)
                per_doc = {n: (emb[n][:nd] @ R.T) for n in CORPORA}
                c = cell(per_doc, ids, regK, args.boot, args.seed)
                for t in TARGETS:
                    k_rows.append(dict(
                        n=args.n, seed=args.seed, chunk=args.chunk, n_boot=args.boot,
                        encoder=label, model_id=model_id, mode=mode, git_sha=gsha,
                        K=K, chance=1 / K, selection="nested_prefix",
                        registry="principals_extended.yaml",
                        suspect=t, top1=c[f"{t}_top1"], rank_of_true=c[f"{t}_rank"],
                        strict_hit=c[f"{t}_strict"], cluster_hit=c[f"{t}_cluster"],
                        boot_top1=c[f"{t}_boot"], n_docs=c["n_docs"],
                        strict_hits_5=c["strict_hits"],
                        cluster_hits_5=c["cluster_hits"], boot_mean=c["boot_mean"],
                        binom_p_exact=binom_sf(c["strict_hits"], 5, 1 / K),
                        top1_over_chance=(c["strict_hits"] / 5) / (1 / K),
                        mean_distractors_above_true=float(
                            np.mean([c[f"{u}_rank"] - 1 for u in TARGETS]))))
                print(f"  K={K:<5} {mode:<11} strict {c['strict_hits']}/5  "
                      f"median rank {np.median([c[f'{t}_rank'] for t in TARGETS]):.0f}  "
                      f"boot {c['boot_mean']:.1%}  "
                      f"x chance {(c['strict_hits'] / 5) / (1 / K):.1f}")

    ndf = pd.DataFrame(n_rows)
    kdf = pd.DataFrame(k_rows)
    ndf.to_csv(REPO / "results" / "n_sweep.csv", index=False)
    kdf.to_csv(REPO / "results" / "k_sweep.csv", index=False)
    print(f"\nwrote results/n_sweep.csv ({len(ndf)} rows), "
          f"results/k_sweep.csv ({len(kdf)} rows)")
    report(ndf, kdf)
    return 0


def report(ndf: pd.DataFrame, kdf: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("N-SWEEP - strict hits/5 (exact binomial p at chance 1/47), nested selection")
    print("=" * 100)
    cells = ndf[ndf.selection == "nested_prefix"].drop_duplicates(
        ["n", "encoder", "mode"])
    for enc in cells.encoder.unique():
        for mode in MODES:
            sub = cells[(cells.encoder == enc) & (cells["mode"] == mode)].sort_values("n")
            if sub.empty:
                continue
            print(f"\n  {enc}  {mode}")
            for r in sub.itertuples():
                print(f"    N={r.n:<7} rows={r.n_rows_used:<7} docs={r.n_docs:<5} "
                      f"strict {r.strict_hits_5}/5  cluster {r.cluster_hits_5}/5  "
                      f"boot {r.boot_mean:>6.1%}  p={r.binom_p_exact:.2e}")

    print("\n  HALF-OF-FULL-N THRESHOLD (smallest N reaching >= half the full-pool "
          "strict hits)")
    for enc in cells.encoder.unique():
        for mode in MODES:
            sub = cells[(cells.encoder == enc) & (cells["mode"] == mode)].sort_values("n")
            if sub.empty:
                continue
            full = sub.iloc[-1].strict_hits_5
            ok = sub[sub.strict_hits_5 >= full / 2]
            first = int(ok.iloc[0].n) if len(ok) and full > 0 else None
            print(f"    {enc:<20} {mode:<11} full-N strict {full}/5 -> "
                  f"half reached at N={first}")

    print("\n" + "=" * 100)
    print("K-SWEEP - degradation as the registry grows toward a realistic threat list")
    print("=" * 100)
    kc = kdf.drop_duplicates(["K", "encoder", "mode"])
    for enc in kc.encoder.unique():
        for mode in MODES:
            sub = kc[(kc.encoder == enc) & (kc["mode"] == mode)].sort_values("K")
            if sub.empty:
                continue
            print(f"\n  {enc}  {mode}")
            for r in sub.itertuples():
                ranks = kdf[(kdf.K == r.K) & (kdf.encoder == enc)
                            & (kdf["mode"] == mode)].rank_of_true
                print(f"    K={r.K:<5} chance {1 / r.K:>6.2%}  strict {r.strict_hits_5}/5"
                      f"  median rank {ranks.median():>6.1f}  boot {r.boot_mean:>6.1%}"
                      f"  top1 = {r.top1_over_chance:>5.1f}x chance  "
                      f"p={r.binom_p_exact:.2e}")


if __name__ == "__main__":
    raise SystemExit(main())
