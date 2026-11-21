#!/usr/bin/env python3
"""
Ingest CBDU OCR documents into AnythingLLM — one document per page.

Prerequisites:
  pip install -r scripts/requirements.txt
  cp .env.example .env  # fill in AUTH_TOKEN etc -- ANYTHINGLLM_API_KEY can be
                        # left blank; it is auto-provisioned on first run (see
                        # ensure_api_key()) using AUTH_TOKEN, no UI step needed.
  docker compose up -d

Run:
  python3 scripts/ingest.py

Clean re-ingest (deletes existing workspace first):
  CLEAN_WORKSPACE=1 python3 scripts/ingest.py

Or via Docker:
  docker compose run --rm -e CLEAN_WORKSPACE=1 ingest
"""

import datetime
import json
import os
import re
import sys
import time

import requests

ANYTHINGLLM_URL = os.getenv("ANYTHINGLLM_URL", "http://localhost:3001")
API_KEY = os.getenv("ANYTHINGLLM_API_KEY", "")
DATA_FILE = os.getenv("DATA_FILE", "data/CBDU_ocr_gemini_pro_2.5.json")
WORKSPACE_NAME = os.getenv("WORKSPACE_NAME", "CBDU Historical Pamphlets")
WORKSPACE_TEMP = float(os.getenv("WORKSPACE_TEMP", "0"))
CLEAN = os.getenv("CLEAN_WORKSPACE", "").lower() in ("1", "true", "yes")
BATCH_SIZE = 50


def api(method, path, timeout=600, **kwargs):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.request(
        method,
        f"{ANYTHINGLLM_URL}/api/v1{path}",
        headers=headers,
        timeout=timeout,
        **kwargs,
    )
    resp.raise_for_status()
    return resp.json()


def wait_for_server(max_attempts: int = 30, interval: int = 5) -> None:
    print(f"Waiting for AnythingLLM at {ANYTHINGLLM_URL} ...")
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(f"{ANYTHINGLLM_URL}/api/ping", timeout=5)
            r.raise_for_status()
            print("  Connected.\n")
            return
        except Exception as exc:
            if attempt == max_attempts:
                print(f"ERROR: Cannot reach AnythingLLM after {max_attempts} attempts: {exc}", file=sys.stderr)
                sys.exit(1)
            print(f"  [{attempt}/{max_attempts}] Not ready yet, retrying in {interval}s ...")
            time.sleep(interval)


def _persist_api_key(key: str) -> None:
    """Write the freshly generated key back into the project's .env so
    host-side tools (eval_workspace.py, set_workspace_temp.py) pick it up
    without a manual copy/paste step."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("ANYTHINGLLM_API_KEY="):
            lines[i] = f"ANYTHINGLLM_API_KEY={key}\n"
            break
    else:
        lines.append(f"ANYTHINGLLM_API_KEY={key}\n")
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  Persisted new API key to {env_path}")


def ensure_api_key() -> str:
    """Auto-provision a developer API key via AUTH_TOKEN if none is set --
    replaces the old "log into the UI and generate one by hand" step. Uses
    the same two calls the AnythingLLM UI itself makes: log in with the
    single-user password to get a session JWT, then mint an API key with it."""
    global API_KEY
    if API_KEY:
        return API_KEY

    password = os.getenv("AUTH_TOKEN", "")
    if not password:
        print("ERROR: no ANYTHINGLLM_API_KEY and no AUTH_TOKEN to auto-provision one", file=sys.stderr)
        sys.exit(1)

    print("No ANYTHINGLLM_API_KEY set -- auto-provisioning one via AUTH_TOKEN login ...")
    resp = requests.post(f"{ANYTHINGLLM_URL}/api/request-token", json={"password": password}, timeout=15)
    resp.raise_for_status()
    token = resp.json().get("token")
    if not token:
        print(f"ERROR: login failed: {resp.json()}", file=sys.stderr)
        sys.exit(1)

    resp = requests.post(
        f"{ANYTHINGLLM_URL}/api/system/generate-api-key",
        headers={"Authorization": f"Bearer {token}"},
        timeout=15,
    )
    resp.raise_for_status()
    API_KEY = resp.json()["apiKey"]["secret"]
    print(f"  Generated API key: {API_KEY}")
    _persist_api_key(API_KEY)
    return API_KEY


def clean_workspace(slug: str) -> None:
    """Un-embed all documents from workspace (leaves workspace intact)."""
    try:
        resp = api("GET", f"/workspace/{slug}")
        ws = resp.get("workspace", [])
        if isinstance(ws, list):
            if not ws:
                print(f"  Workspace {slug} not found — nothing to clean")
                return
            ws = ws[0]
        docs = ws.get("documents", [])
        if not docs:
            print(f"  Workspace empty — nothing to clean")
            return
        locs = [d["docpath"] for d in docs if d.get("docpath")]
        # Delete in batches — one giant deletes payload (tens of thousands of
        # vectors) overruns the request timeout and aborts the whole re-ingest.
        for start in range(0, len(locs), BATCH_SIZE):
            batch = locs[start : start + BATCH_SIZE]
            api("POST", f"/workspace/{slug}/update-embeddings", json={"adds": [], "deletes": batch})
            print(f"  Removed {start + len(batch)}/{len(locs)} documents from workspace")
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            print(f"  Workspace {slug} not found — nothing to clean")
        else:
            raise


def get_or_create_workspace() -> str:
    slug = WORKSPACE_NAME.lower().replace(" ", "-")
    try:
        resp = api("GET", f"/workspace/{slug}")
        ws = resp["workspace"]
        if isinstance(ws, list):
            if not ws:
                raise requests.HTTPError("not found")
            ws = ws[0]
        print(f"Using existing workspace: {slug}")
    except requests.HTTPError:
        resp = api("POST", "/workspace/new", json={"name": WORKSPACE_NAME})
        slug = resp["workspace"]["slug"]
        print(f"Created workspace: {slug}")
    workspace_settings = {
        # "automatic" (agent mode) as the default, not "query" -- relies on
        # the tooled.js patch (patches/anythingllm/tooled.js, mounted over
        # the stock file) forcing the first tool call after each user
        # message to be rag-memory, confirmed via testing to reliably search
        # every turn including follow-ups that previously skipped it.
        "chatMode": "automatic",
        # 2689 docs chunk into ~9.3k vectors; topN=3 only ever returns chunks from
        # a single document. Higher topN spans multiple sources for better recall.
        "topN": 75,
        "similarityThreshold": 0.4,
        # Requires patches/anythingllm/qdrant-index.js (mounted over the stock
        # provider) -- upstream Qdrant support never wired up AnythingLLM's
        # own NativeEmbeddingReranker (a standalone local CPU cross-encoder,
        # only LanceDB's provider did). "rerank" pulls a larger candidate
        # pool via plain vector search, then reranks it locally before
        # truncating to topN -- see the patch file for the mirrored
        # LanceDB implementation this is based on.
        "vectorSearchMode": "rerank",
        # FIX #1 (context bleeding): lowered from 6. AnythingLLM feeds up to
        # openAiHistory prior turns into the LLM's messages regardless of
        # chatMode ("query" mode does not disable this), so with reset:false
        # eval runs (many sequential unrelated questions in one thread) this
        # let facts/documents from earlier questions bleed into later answers.
        # 2 (not 0) keeps enough history for real multi-turn follow-up while
        # reducing the bleed surface -- see HALLUCINATION_FIXES.md.
        "openAiHistory": 2,
        # Lowered from the provider default of 0.7 -- eval found the model
        # padding answers with plausible-sounding but unverified outside
        # knowledge despite "context-only" instructions; less speculation.
        "openAiTemp": WORKSPACE_TEMP,
        "openAiPrompt": (
            "Jesteś asystentem do badań historycznych nad polskimi drukami ulotnymi (broszurami/gazetkami "
            "informacyjnymi typu \"nowiny\") z katalogu CBDU (Cyfrowa Biblioteka Druków Ulotnych). "
            # FIX #2 + #4 + #5 (institution fabrication, date-range inflation,
            # count still not reliably cited): consolidated "hard facts" block
            # moved near the START of the prompt (was previously only near the
            # end, count-only) -- repeatedly confirmed clean across two test
            # rounds, see HALLUCINATION_FIXES.md.
            "TWARDE FAKTY O TEJ BAZIE (nie \"poprawiaj\" ich żadną ogólną wiedzą o pełnym katalogu CBDU, "
            "który w rzeczywistości online zawiera więcej pozycji i inny zakres dat -- ta ogólna wiedza NIE "
            "dotyczy zawartości TEJ KONKRETNEJ przestrzeni roboczej): "
            "(1) Instytucją prowadzącą CBDU jest Instytut Języka Polskiego PAN (IJP PAN). Jeśli pytanie "
            "dotyczy instytucji odpowiedzialnej za CBDU, podaj DOKŁADNIE tę nazwę i nie wymyślaj innej. "
            "(2) Ta przestrzeń robocza zawiera DOKŁADNIE 314 unikalnych druków (ok. 2689 stron) -- jeśli "
            "użytkownik pyta o całkowitą liczbę dokumentów/druków w TEJ BAZIE, odpowiedz DOKŁADNIE \"314 "
            "unikalnych druków (ok. 2689 stron)\" bez żadnego zastrzeżenia sugerującego, że to literówka, "
            "zaokrąglenie, czy że prawdziwe CBDU ma więcej pozycji. "
            # Corrected from 1557-1720: verified against the actual 315 ingested
            # records (not just the raw source file) that the true max is 1728
            # ("Relacja koronacji cudownego obrazu Najświętszej Marii Panny na
            # Górze Różańcowej [w Podkamieniu]"), missed by the old range. One
            # source record has Year=2015 but its own Contents field says "To
            # jest druk testowy, do różnych eksperymentów" (a test/placeholder
            # entry) -- excluded here as not a real data point, even though its
            # OCR'd content is a genuine 17th-c. pamphlet under bad catalog
            # metadata (a separate data-quality issue, not a range question).
            "(3) Zakres dat reprezentowany w TEJ BAZIE to DOKŁADNIE 1557-1728 -- jeśli użytkownik pyta o "
            "pełny zakres/okres czasowy TEJ BAZIE, odpowiedz DOKŁADNIE \"1557-1728\" i nie podawaj żadnego "
            "innego zakresu (np. \"XV wiek\", \"1473\", \"do 1800\", \"do 1900\"). "
            "Baza obejmuje druki ze szczególnym nasileniem w okresie 1680-1710, opisujące wydarzenia "
            "polityczne, wojenne i dyplomatyczne Rzeczpospolitej oraz Europy (m.in. wojny z Imperium "
            "Osmańskim, Szwecją i Moskwą, sprawy dworskie i kościelne). Każdy dokument zawiera transkrypcję "
            "nowożytną (znormalizowaną) oraz — jeśli dostępna — transkrypcję dyplomatyczną (oryginalna "
            "pisownia). Odpowiadaj po polsku, zwięźle i konkretnie. "
            "Opieraj się WYŁĄCZNIE na dostarczonym kontekście. Jeśli odpowiedzi nie ma w dokumentach, "
            "napisz wprost, że nie znajdujesz tej informacji w źródłach. Nie spekuluj ani nie dodawaj "
            "wiedzy spoza kontekstu. "
            # FIX #3 (historical fact fabrication despite correct retrieval):
            # names the specific detail types models kept inventing (wrong
            # successor names, wrong battle names) and gives an explicit
            # fallback phrase, with an "answer what you can first" clause so
            # the fallback covers only the missing detail, not the whole
            # question, plus a rule against claiming "no content" when any
            # text is actually present -- see HALLUCINATION_FIXES.md.
            "W SZCZEGÓLNOŚCI nie uzupełniaj wiedzą ogólną spoza kontekstu konkretnych faktów historycznych, "
            "których dokument nie podaje wprost -- np. imion następców władców, przyczyn wydarzeń, nazw "
            "bitew, dokładnych wyników starć czy dat, których nie ma w dostarczonym tekście. Ta zasada "
            "dotyczy WYŁĄCZNIE pojedynczych, niepotwierdzonych szczegółów -- NIE oznacza odmowy odpowiedzi "
            "na całe pytanie. Zawsze najpierw opisz to, co dokument RZECZYWIŚCIE zawiera (główne wydarzenie, "
            "temat, kontekst), a dopiero dla konkretnego brakującego szczegółu napisz wprost \"dokument tego "
            "nie precyzuje\", zamiast pomijać całą resztę odpowiedzi lub zgadywać. "
            "NIGDY nie twierdź, że dokument \"nie zawiera treści merytorycznej, a jedynie identyfikator i "
            "datowanie\", jeśli w dostarczonym kontekście widoczny jest JAKIKOLWIEK fragment tekstu poza "
            "samą linią \"[Datowanie: ...]\" -- w tym fragmenty oznaczone jako \"[Zapis oryginalny]\" (dawna "
            "pisownia). Nawet jeśli tekst jest fragmentaryczny, urywa się w połowie zdania, lub zapisany w "
            "dawnej pisowni i trudny do pełnego odczytania, wykorzystaj go, aby streścić temat i główne "
            "wydarzenie dokumentu najlepiej jak potrafisz, zamiast twierdzić, że treść jest niedostępna. "
            "Podając informacje, wskazuj rok i tytuł źródłowego druku, jeśli są "
            "dostępne w kontekście. "
            # CITATION-NAME FABRICATION GUARD (surfaced by the bibliografia
            # eval trial): when the commentary cites a scholarly source by
            # initials only (e.g. "K. Zawadzki"), models repeatedly invented
            # a specific full first name to fill the gap ("Tomasz Zawadzki",
            # "Stanisław Zawadzki", "Krzysztof Zawadzki" -- three different,
            # mutually contradictory guesses across otherwise-correct
            # answers). The rest of each citation (book title, year, page)
            # was always right -- only the expanded initial was fabricated.
            "Jeśli komentarz w źródle podaje autora pracy naukowej wyłącznie w formie skróconej (np. \"K. "
            "Zawadzki\", \"K. Badecki\"), NIGDY nie rozwijaj tego skrótu we własne, zmyślone imię (np. "
            "\"Tomasz Zawadzki\", \"Krzysztof Zawadzki\") -- cytuj dokładnie tak, jak podano w źródle, chyba "
            "że pełne imię jest jawnie wypisane w dostarczonym kontekście. "
            # FIX #7: six variants tested total (v1-v6), all against
            # scripts/eval_questions_context_labels.json, 3x runs each:
            #   v1 (this wording -- explicit negation, names the pattern): 7/24 leaked, 0 severe -- BEST
            #   v2 (v1 + concrete quoted negative example): 9/24 -- model echoed the quoted phrase verbatim
            #   v3 (stronger/more abstract negation, no example): 11/24 -- worse
            #   v4 (fully positive framing, never names "kontekst"/"context"): 12/24 -- worst, one answer used it 8x
            #   v5 (v1's exact wording, repositioned to the front of the "hard facts" block): 10/24 -- worse
            #   v6 (v1 + an explicit "review before sending, remove any found" self-check step): 7/24 answers,
            #       but ~27 total occurrences vs v1's ~12 -- same breadth, more entrenched repetition per answer
            # Conclusion: v1 wins on every attempted variation, by every
            # measure. STOP iterating on this instruction for good -- see
            # HALLUCINATION_FIXES.md. The only lever not yet tried is
            # post-processing the returned answer text outside the prompt
            # entirely (regex strip or a second LLM cleanup pass), not
            # another prompt wording/position/mechanism change.
            "NIGDY nie używaj w odpowiedzi oznaczeń typu \"Kontekst 0\", \"Kontekst 1\", \"[CONTEXT N]\" ani "
            "podobnych numerów fragmentów -- to wewnętrzne etykiety techniczne, nie tytuły dokumentów, i "
            "użytkownik ich nie rozumie. Jeśli tytuł lub nazwa druku pojawia się w treści fragmentu, użyj "
            "GO zamiast numeru. Jeśli tytuł nie jest znany z treści, po prostu opisz informację bez "
            "przypisywania jej do numeru fragmentu (np. \"jeden z dokumentów opisuje...\" zamiast "
            "\"Kontekst 3 opisuje...\"). "
            "Każdy dokument zaczyna się od linii \"[Datowanie: ...]\" — to jest "
            "JEDYNE wiarygodne źródło daty wydania, zachowujące rzeczywistą precyzję i pewność ustalenia "
            "(np. \"po 9 października 1594\" oznacza, że znana jest tylko najwcześniejsza możliwa data, "
            "a \"(data ustalona pośrednio, niepewna)\" oznacza datę wywnioskowaną przez katalogującego, nie "
            "wydrukowaną na druku). Zawsze przytaczaj datowanie DOKŁADNIE tak, jak podano w tej linii, "
            "wraz z zastrzeżeniami (\"po\", \"niepewna\" itp.) — nigdy nie zamieniaj go na fikcyjną pewną "
            "datę dzienną. Ignoruj wszelkie inne pola daty w metadanych dokumentu. "
            # date-correction fabrication guard (surfaced by fix_test v1/v2):
            # models correctly quoted the source's own date, then invented an
            # unsupported "historical correction" claiming a different date.
            "Nie dodawaj żadnej innej, rzekomo dokładniejszej lub \"skorygowanej\" daty wydarzenia jako "
            "rzekomą korektę historyczną -- jeśli podajesz kontekst historyczny, opieraj go WYŁĄCZNIE na "
            "datach faktycznie obecnych w dostarczonym tekście. Nigdy nie wprowadzaj nowej, innej daty "
            "dziennej spoza kontekstu, nawet jeśli wydaje się bardziej precyzyjna lub prawdopodobna. "
            "PRZYKŁAD ZABRONIONEGO ZACHOWANIA: dokument podaje \"[Datowanie: po 28 sierpnia 1557]\", a "
            "odpowiedź dodaje zdanie w stylu \"tak naprawdę wydarzenie miało miejsce dokładnie 2 sierpnia "
            "1557\" lub podobną rzekomo dokładniejszą datę -- to jest ZABRONIONE, niezależnie od tego, jak "
            "wiarygodnie taka data brzmi. Jedyna dopuszczalna data to ta z linii \"[Datowanie: ...]\", "
            "cytowana dokładnie i bez modyfikacji. "
            "UWAGA: ta przestrzeń robocza zawiera wyłącznie WYBRANY PODZBIÓR pełnego katalogu CBDU — tylko "
            "druki z dostępną transkrypcją OCR. To NIE jest cały katalog CBDU. Wyszukiwanie wektorowe nie ma "
            "dostępu do pełnej listy dokumentów — dla pytań w stylu \"które konkretnie\" (np. "
            "najstarszy/najnowszy dokument) przyznaj wprost, że dysponujesz tylko fragmentem wyszukanych "
            "dokumentów, a nie całą bazą. "
            # AGENT MODE FIX (option 2 of 2 being tested -- see also the
            # tool_choice patch to AnythingLLM's tooled.js): this same prompt
            # is reused verbatim as the agent's role definition
            # (Provider.systemPrompt() in ai-provider.js just returns
            # workspace.openAiPrompt as-is). AnythingLLM never sets
            # tool_choice on agent completions -- tools are always offered
            # with the OpenAI-spec default "auto", so the model can and does
            # sometimes skip searching (e.g. on a follow-up like "a źródło?"
            # where it assumes it already knows enough). This instruction is
            # a no-op in "query" mode (no tools exist there) but should push
            # the agent to search every turn regardless.
            "JEŚLI działasz w trybie agenta z dostępem do narzędzi: ZAWSZE, przy KAŻDEJ turze rozmowy, "
            "najpierw wywołaj narzędzie rag-memory (action: search), aby przeszukać lokalne dokumenty, "
            "zanim odpowiesz -- nawet jeśli sądzisz, że znasz już odpowiedź z poprzedniej wiadomości w tej "
            "samej rozmowie. Nigdy nie pomijaj tego kroku i nigdy nie odpowiadaj wyłącznie z pamięci "
            "poprzednich wiadomości bez ponownego przeszukania dokumentów. "
            # Confirmed in testing: on a broad/superlative question ("najstarszy
            # dokument w bazie?"), a single rag-memory call only ever returns
            # topN candidates from ONE query embedding -- it can genuinely
            # miss the true answer if it lies outside that one fragment.
            # Prompting the agent by hand to search again with a different
            # angle fixed a wrong/uncertain first answer into a correct,
            # confident one. Making that retry-with-different-queries
            # behavior standard instead of relying on the user asking for it.
            "Jeśli pytanie dotyczy skrajności (najstarszy/najnowszy dokument), szerokiego podsumowania, "
            "lub prosi o \"wszystkie\"/\"każdy\" wystąpienie czegoś w bazie -- jedno wywołanie rag-memory "
            "prawdopodobnie NIE wystarczy, bo zwraca tylko fragment pasujących wyników z jednego zapytania. "
            "W takich przypadkach wykonaj kilka wywołań rag-memory pod rząd z różnymi wariantami zapytania "
            "(inne słowa kluczowe, inne sformułowanie, inny aspekt tematu, inny zakres dat) i dopiero na "
            "podstawie połączonych wyników z tych wyszukiwań sformułuj ostateczną odpowiedź."
        ),
    }
    api("POST", f"/workspace/{slug}/update", json=workspace_settings)
    print(
        f"  chat_mode={workspace_settings['chatMode']}  topN={workspace_settings['topN']}  "
        f"similarity={workspace_settings['similarityThreshold']}  "
        f"vectorSearchMode={workspace_settings['vectorSearchMode']}  system_prompt=set"
    )
    return slug


_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

_ROMAN_MONTHS = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
    "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12,
}
_MONTH_PL_GENITIVE = {  # "13 {}" — used when a day precedes the month
    1: "stycznia", 2: "lutego", 3: "marca", 4: "kwietnia", 5: "maja", 6: "czerwca",
    7: "lipca", 8: "sierpnia", 9: "września", 10: "października", 11: "listopada", 12: "grudnia",
}
_MONTH_PL_NOMINATIVE = {  # standalone "{} 1674" — no day precedes it
    1: "styczeń", 2: "luty", 3: "marzec", 4: "kwiecień", 5: "maj", 6: "czerwiec",
    7: "lipiec", 8: "sierpień", 9: "wrzesień", 10: "październik", 11: "listopad", 12: "grudzień",
}
_DATE_RE = re.compile(r"^(?:(\d{1,2})\s+)?(?:([IVXLCM]+)\s+)?(\d{4})$")


class ParsedDate:
    """A CBDU publication date, preserving how certain/precise it actually is.

    Unlike a plain calendar date, this can represent "we only know the year",
    "we only know it was after this date" (terminus post quem), or "the
    cataloguer inferred this exact date but it wasn't printed on the item" —
    distinctions the source catalogue makes explicit and that get lost if
    collapsed into a single fabricated exact date.
    """

    def __init__(self, year, month=None, day=None, after_only=False, approximate=False):
        self.year, self.month, self.day = year, month, day
        self.after_only, self.approximate = after_only, approximate

    def epoch_ms(self) -> str:
        # Coarse anchor for AnythingLLM's native `published` field, which can
        # only hold a plain calendar date (see doc comment on build_pages).
        dt = datetime.datetime(self.year, self.month or 1, self.day or 1, tzinfo=datetime.timezone.utc)
        return str(int((dt - _EPOCH).total_seconds() * 1000))

    def label(self) -> str:
        """Human-readable Polish label preserving precision/certainty — this,
        not `published`, is the authoritative source of dating info."""
        if self.day and self.month:
            s = f"{self.day} {_MONTH_PL_GENITIVE[self.month]} {self.year}"
            approx_word = "data ustalona pośrednio, niepewna"
        elif self.month:
            s = f"{_MONTH_PL_NOMINATIVE[self.month]} {self.year}"
            approx_word = "data ustalona pośrednio, niepewna"
        else:
            s = str(self.year)
            approx_word = "rok szacunkowy"
        if self.after_only:
            return f"po {s} (nie wcześniej niż ta data)"
        if self.approximate:
            return f"{s} ({approx_word})"
        return s


def parse_date_of_publication(raw: str | None, year_fallback: str) -> "ParsedDate | None":
    """Parse CBDU's `Date of Publication` scholarly notation into a ParsedDate.

    Formats seen in the source catalogue:
      "26 IX 1693"        - exact date, printed on the item itself
      "[20 V 1644]"       - exact date, but supplied/inferred by the
                            cataloguer (ISBD square-bracket convention)
      "[III 1674]"        - month+year only, inferred
      "[1570]"            - year only, inferred
      "po 9 X 1594"       - terminus post quem: only known to be published
                            after this date (true date could be later)
      "[po 15 II 1575]"   - same, bracketed
      "[po 15 VII] 1718"  - same, with year outside the brackets
    Falls back to `year_fallback` (the catalogue's separate Year field) when
    `raw` is absent or doesn't match any known pattern.
    """
    if not raw:
        return ParsedDate(int(year_fallback)) if year_fallback.isdigit() else None

    approximate = "[" in raw
    cleaned = raw.replace("[", " ").replace("]", " ").replace(".", " ").strip()
    after_only = bool(re.match(r"^po\b", cleaned, re.IGNORECASE))
    if after_only:
        cleaned = re.sub(r"^po\s+", "", cleaned, flags=re.IGNORECASE)

    m = _DATE_RE.match(re.sub(r"\s+", " ", cleaned).strip())
    if not m:
        return ParsedDate(int(year_fallback)) if year_fallback.isdigit() else None

    day_s, month_roman, year_s = m.groups()
    month = _ROMAN_MONTHS.get(month_roman) if month_roman else None
    if month_roman and month is None:
        return ParsedDate(int(year_fallback)) if year_fallback.isdigit() else None

    return ParsedDate(
        year=int(year_s),
        month=month,
        day=int(day_s) if day_s else None,
        after_only=after_only,
        approximate=approximate and not after_only,  # after_only already implies uncertainty
    )


def build_pages(record: dict) -> list[tuple[str, str, dict]]:
    """Return one (doc_name, text, metadata) per OCR page.

    Text = pure transcription — no metadata noise polluting embeddings. A
    per-page context header was tried and reverted: on multi-page docs it
    produced many near-identical title-only chunks that flooded title-like
    queries and starved the LLM of body text. Catalogue fields stay in metadata.

    Of those catalogue fields, AnythingLLM's raw-text collector only persists
    a fixed whitelist (title, docAuthor, description, docSource, chunkSource,
    published) — anything else sent as custom keys is silently dropped, and
    `published` can only hold a plain calendar date, not "year-only" or
    "after this date" precision. So the real dating info (see ParsedDate)
    is instead embedded as a one-line `[Datowanie: ...]` header directly in
    the transcribed text — the only channel that survives untouched into
    both the vector embedding and the LLM's retrieved context, precision and
    uncertainty markers intact. `published` still gets a coarse epoch anchor
    for AnythingLLM's own UI/citation display; publisher/place are folded
    into `docSource` since there's no dedicated field for those either.
    """
    title = (record.get("Tytuł") or record.get("Tytuł pełny") or "Unknown").lstrip("#").strip()
    year = str(record.get("Year", ""))
    description = (record.get("Contents") or record.get("Zawartość") or "").strip()
    publisher = record.get("Publisher", "")
    place = record.get("Place of Publication", "")
    parsed_date = parse_date_of_publication(record.get("Date of Publication"), year)

    # Scholarly commentary fields (General/Historical/Linguistic Commentary)
    # -- previously not captured anywhere. Embedded as an inline bracketed
    # marker (same mechanism as [Datowanie: ...] below) so it's actually
    # searchable/visible to the LLM, not just structured metadata
    # (AnythingLLM never sends structured payload fields to the model -- see
    # genericOpenAi/index.js's #appendContext(), which only uses each
    # chunk's embedded text). Commentary fields include real scholarly
    # bibliography references (e.g. to K. Zawadzki's "Poczatki prasy
    # polskiej") that exist nowhere else in this dataset as a structured
    # field. Validated in the "desc_trial" eval trial: 5x-repeated runs
    # across bibliografia/linguistic-note/historical-note questions were
    # 5/5 consistent and grounded, and the citation-check URL issue seen in
    # an earlier trial that ALSO added a [Identyfikator CBDU: N] marker
    # turned out to reproduce on the untouched main instance too -- i.e. it
    # was unrelated to either marker, so the identifier addition was
    # dropped as unproven while this commentary addition was kept. NOTE:
    # per direct measurement, ~88.6% of OCR pages exceed AnythingLLM's
    # 1500-char chunk size and get split into multiple chunks by
    # AnythingLLM's own splitter, AFTER this script runs -- so, like
    # [Datowanie: ...], this marker will only survive into the first
    # sub-chunk of a split page, not all of them.
    commentary_parts = []
    for label, key in [
        ("Komentarz ogólny", "General Commentary"),
        ("Komentarz historyczny", "Historical Commentary"),
        ("Komentarz językowy", "Linguistic Commentary"),
    ]:
        val = (record.get(key) or "").strip()
        if val:
            commentary_parts.append(f"{label}: {val}")
    commentary_text = " ".join(commentary_parts)

    webpage_url = record.get("webpage_url", "")
    # Also folded into the `description` metadata field (AnythingLLM's
    # document-manager UI shows this; it is metadata-only, NOT embedded --
    # see with_source()/build_pages() docstring) so it's visible there too,
    # independent of the embedded inline marker below.
    description_parts = [description] if description else []
    if commentary_text:
        description_parts.append(commentary_text)
    base_meta = {
        "title": title[:80],
        "description": "\n\n".join(description_parts),
        "docAuthor": record.get("Author") or record.get("Autor", "") or "no author specified",
        "docSource": ", ".join(filter(None, [publisher, place])) or "no source set",
    }
    # Only set `published` when the date is a genuinely confirmed exact
    # day+month+year (printed on the item, not bracketed/inferred, not a
    # "po" lower-bound) — AnythingLLM's document-manager UI renders this as
    # a real calendar date+time tooltip, so anything less than fully
    # confirmed would show a fabricated precise date to users. The honest,
    # fully-qualified label (including "po"/approximate markers) always
    # stays available via the `[Datowanie: ...]` line in the document text.
    if (
        parsed_date is not None
        and parsed_date.day
        and parsed_date.month
        and not parsed_date.approximate
        and not parsed_date.after_only
    ):
        base_meta["published"] = parsed_date.epoch_ms()

    def with_source(url: str, **extra) -> dict:
        # Citation display, decoded from AnythingLLM's own frontend bundle
        # (kg() in index-*.js): for chunkSource prefixes it doesn't have a
        # dedicated icon for, it falls through to the generic "link://"
        # case, which renders the header as the URL's host+pathname (NOT the
        # title) -- that's why citations were showing the raw PDF path
        # instead of a readable title. Prefixes it DOES recognize (github://,
        # gitlab://, confluence://, drupalwiki://, obsidian://, ...) render
        # the header as the actual title instead, while `href` is computed
        # identically in every case (just the URL after the prefix) -- so
        # switching prefix only changes the displayed icon, not the link
        # target. "drupalwiki://" was picked as the closest fit (a
        # wiki/reference-page icon) for a digital-library citation.
        # Second change: always point at the eprint's info page
        # (record.webpage_url, e.g. ".../id/eprint/17400/") instead of the
        # direct per-file PDF link -- more useful to click through to than a
        # raw PDF download.
        url = webpage_url or url
        meta = {**base_meta, "url": url, **extra}
        if url:
            meta["chunkSource"] = f"drupalwiki://{url}"
        return meta

    # Combined inline prefix: date stays exactly as before, Commentary is
    # ADDED alongside it, not replacing it. Computed once (none of these
    # vary per-page) and glued onto each page's own paragraph the same way
    # Datowanie always was, to avoid the chunker splitting it into a
    # standalone near-empty chunk (FIX #6).
    prefix_markers = []
    if parsed_date is not None:
        prefix_markers.append(f"[Datowanie: {parsed_date.label()}]")
    if commentary_text:
        prefix_markers.append(f"[{commentary_text}]")
    prefix = " ".join(prefix_markers)

    pages = []
    for fi in record.get("files", []):
        gemini = fi.get("ocr", {}).get("gemini_corrected", {})
        if not gemini:
            continue

        file_url = fi.get("url", "")
        total_pages = len(gemini)
        for img_path, page in sorted(gemini.items()):
            page_num = (
                img_path.rsplit("_page_", 1)[-1].replace(".png", "")
                if "_page_" in img_path
                else "1"
            )
            modern = (page.get("modern_transcription") or "").strip()
            diplomatic = (page.get("diplomatic_transcription") or "").strip()

            if not modern and not diplomatic:
                continue

            # FIX #6 (chunking bug, confirmed via direct Qdrant similarity
            # search): with the Datowanie line as its OWN paragraph (blank
            # line before body), AnythingLLM's chunker often splits it off
            # into a standalone near-empty chunk. For a multi-page x
            # multi-edition document, that produces many near-duplicate
            # "header only, no page content" chunks that can crowd out the
            # real content chunk for topN under title-style queries. Fix:
            # glue the label onto the SAME paragraph as the real content
            # (space, not blank line) so it can't be split off alone --
            # see HALLUCINATION_FIXES.md.
            parts = []
            if modern:
                parts.append(modern)
            if diplomatic and diplomatic != modern:
                parts.append(f"[Zapis oryginalny]\n{diplomatic}")
            body = "\n\n".join(parts)
            text = f"{prefix} {body}" if (body and prefix) else (prefix or body)

            if total_pages > 1:
                doc_name = f"{year} — {title} [s. {page_num}]"[:120]
            else:
                doc_name = f"{year} — {title}"[:120]

            pages.append((doc_name, text, with_source(file_url, page=page_num)))

    # Fallback: KORBA transcription (single-page, no split)
    if not pages and record.get("korba_transcription"):
        korba_text = str(record["korba_transcription"])
        if prefix:
            # FIX #6: glued onto same paragraph, see comment above.
            korba_text = f"{prefix} {korba_text}"
        pages.append((
            f"{year} — {title}"[:120],
            korba_text,
            with_source(webpage_url),
        ))

    return pages


def upload_documents(records: list[dict]) -> list[str]:
    locations = []
    total = len(records)

    for i, record in enumerate(records, 1):
        title = (record.get("Tytuł") or record.get("Tytuł pełny") or "Unknown").lstrip("#").strip()
        pages = build_pages(record)
        if not pages:
            continue

        rec_locs = []
        for doc_name, text, meta in pages:
            # strip control characters that break JSON parsing server-side
            clean_text = "".join(c for c in text if c >= " " or c in "\n\r\t")
            try:
                resp = api(
                    "POST",
                    "/document/raw-text",
                    json={"textContent": clean_text, "metadata": meta},
                )
                loc = resp["documents"][0]["location"]
                rec_locs.append(loc)
            except requests.HTTPError as exc:
                body = exc.response.text[:300] if exc.response is not None else ""
                print(f"    FAILED: {doc_name[:60]} — {exc} | {body}", file=sys.stderr)
            except Exception as exc:
                print(f"    FAILED: {doc_name[:60]} — {exc}", file=sys.stderr)

        locations.extend(rec_locs)
        print(f"  [{i:>3}/{total}] {title[:60]} ({len(rec_locs)} pages)")

    return locations


def embed_documents(workspace_slug: str, locations: list[str]) -> None:
    total = len(locations)
    for start in range(0, total, BATCH_SIZE):
        batch = locations[start : start + BATCH_SIZE]
        end = start + len(batch)
        try:
            api(
                "POST",
                f"/workspace/{workspace_slug}/update-embeddings",
                json={"adds": batch, "deletes": []},
            )
            print(f"  Embedded {end}/{total}")
        except Exception as exc:
            print(f"  FAILED batch [{start}:{end}]: {exc}", file=sys.stderr)


def main() -> None:
    wait_for_server()
    ensure_api_key()

    print(f"Loading {DATA_FILE} ...")
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    records = [
        r for r in data
        if (
            any("gemini_corrected" in fi.get("ocr", {}) for fi in r.get("files", []))
            or r.get("korba_transcription")
        )
        # Excludes Identifier "0" ("Literacka zabawa z okazji Dnia
        # Bibliotekarza i Bibliotek" / "Literackie bingo") -- a bogus
        # catalogue placeholder, not a real CBDU pamphlet: Contents field
        # literally reads "To jest druk testowy, do różnych eksperymentów"
        # and every catalogue field (Publisher/Format/Sheets/Signing) is
        # "Test object". Its actual OCR'd page content happens to be a real
        # 17th-century pamphlet (Battle of Kircholm, 1605) sitting under
        # this bogus metadata -- a separate data-quality bug upstream, not
        # something to paper over by keeping the fake record in our corpus.
        # Filtered on Publisher rather than a hardcoded Identifier so any
        # other stray test entries with the same marker are also excluded.
        and r.get("Publisher") != "Test object"
    ]
    print(f"Found {len(records)} records with OCR/transcription (out of {len(data)} total)\n")

    slug = WORKSPACE_NAME.lower().replace(" ", "-")

    if CLEAN:
        print("CLEAN_WORKSPACE=1 — removing all existing documents ...")
        clean_workspace(slug)
        print()

    workspace_slug = get_or_create_workspace()

    print(f"\nUploading documents ...")
    locations = upload_documents(records)
    print(f"\nUploaded {len(locations)} page-documents.")

    print(f"\nEmbedding into workspace '{workspace_slug}' ...")
    embed_documents(workspace_slug, locations)

    print(f"\nDone. Open http://localhost:3001 — workspace: '{WORKSPACE_NAME}'")


if __name__ == "__main__":
    main()
