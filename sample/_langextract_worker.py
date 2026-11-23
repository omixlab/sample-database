"""
Standalone LangExtract worker -- runs INSIDE the ``sample-langextract`` environment.

Executed as a subprocess by :mod:`sample.langextract_records`, never imported by the rest of the
project. It must therefore stay free of any ``sample.*`` import: langextract pulls its own
pydantic/httpx tree, while ``sample-extraction`` is pinned to numpy<2 for spacy, so the two
environments cannot share a process. See ``environments/sample-langextract.yml``.

Input  : a text file path plus the Ollama model id and server URL, on argv.
Output : one JSON object written to ``--out``, never to stdout.

The payload travels in a FILE rather than on stdout for the same reason the docling worker does it:
model and HTTP client libraries print to stdout through channels that ``logging``/``warnings``
suppression does not reach, and anything printed ahead of the JSON makes the result unparseable.

LOCAL ONLY. ``model_url`` always points at the caller-supplied Ollama server and ``model_id`` is
always an Ollama tag. No API key is read or required, and no remote provider is reachable from
here -- if Ollama is down the run fails, it does not silently fall back to a hosted model.
"""
import argparse
import json
import os
import sys


#: What the model is asked to find. Deliberately narrow: this backend exists to recover peptide
#: ACTIVITY MEASUREMENTS that the plain text layer flattens, which is the same job the docling table
#: backend does, so the two produce interchangeable records.
PROMPT = (
    "Extract every peptide that this document reports an experimental activity measurement for. "
    "For each peptide, extract its name and, when the document states it, its amino-acid sequence "
    "in one-letter code. For each measurement, extract the assay type (MIC, MBC, IC50, EC50, "
    "LD50, Hemolysis and so on), the numeric value exactly as printed, its unit exactly as "
    "printed, and the target organism or cell line it was measured against. "
    "Use the exact text of the document for every extraction; do not paraphrase, do not convert "
    "units, and do not infer a value that is not printed. Extract only peptides this document "
    "itself measured -- ignore peptides it merely cites from other work."
)


def _examples(lx):
    """
    :param lx: The imported ``langextract`` module.
    :returns: Few-shot examples in langextract's ``ExampleData``/``Extraction`` form.

    Grounded in the real shape of this corpus: a named peptide with a sequence and a MIC against a
    named organism, and a name-only peptide with an IC50 against a cell line. The attributes use the
    same field names the rest of this project persists, so the worker's output needs no renaming.
    """
    return [
        lx.data.ExampleData(
            text=("Magainin 2 (GIGKFLHSAKKFGKAFVGEIMNS) showed an MIC of 25 ug/mL against "
                  "Escherichia coli ATCC 25922."),
            extractions=[
                lx.data.Extraction(
                    extraction_class="peptide",
                    extraction_text="Magainin 2",
                    attributes={"peptide_sequence": "GIGKFLHSAKKFGKAFVGEIMNS"},
                ),
                lx.data.Extraction(
                    extraction_class="activity",
                    extraction_text="MIC of 25 ug/mL against Escherichia coli ATCC 25922",
                    attributes={
                        "peptide_name": "Magainin 2",
                        "activity_type": "MIC",
                        "value_raw": "25",
                        "activity_unit": "ug/mL",
                        "target_organism": "Escherichia coli",
                    },
                ),
            ],
        ),
        lx.data.ExampleData(
            text="Peptide JG436 inhibited MCF-7 cells with an IC50 of 12.5 uM.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="peptide",
                    extraction_text="JG436",
                    attributes={"peptide_sequence": ""},
                ),
                lx.data.Extraction(
                    extraction_class="activity",
                    extraction_text="IC50 of 12.5 uM",
                    attributes={
                        "peptide_name": "JG436",
                        "activity_type": "IC50",
                        "value_raw": "12.5",
                        "activity_unit": "uM",
                        "target_cell_line": "MCF-7",
                    },
                ),
            ],
        ),
    ]


def main() -> int:
    """:returns: Process exit status; 0 even when nothing is found, non-zero only on real failure."""
    parser = argparse.ArgumentParser(description="LangExtract peptide-activity worker.")
    parser.add_argument("text_path", help="Document text to extract from.")
    parser.add_argument("--out", required=True, help="Where to write the JSON payload.")
    parser.add_argument("--model-id", required=True, help="Ollama model tag, e.g. gemma4:latest.")
    parser.add_argument("--model-url", required=True, help="Local Ollama server URL.")
    parser.add_argument("--max-chars", type=int, default=200_000,
                        help="Truncate the document at this many characters.")
    parser.add_argument("--max-char-buffer", type=int, default=3000,
                        help="Characters per chunk sent to the model. Measured optimum on this "
                             "host; see sample.langextract_records.MAX_CHAR_BUFFER for the data.")
    parser.add_argument("--num-ctx", type=int, default=4096,
                        help="Ollama context window. Must comfortably exceed --max-char-buffer "
                             "plus the prompt and examples, or the chunk is silently truncated.")
    parser.add_argument("--max-workers", type=int, default=2,
                        help="Concurrent chunk requests. Bounded by what the Ollama server can "
                             "actually run at once, not by CPU.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 for a reproducible extraction.")
    parser.add_argument("--think", dest="think", action="store_true", default=None,
                        help="Let a reasoning family (gemma4, qwen3, deepseek-r1, phi4) emit its "
                             "chain of thought. langextract's Ollama provider does "
                             "`combined_kwargs.setdefault('think', False)`, so NOT passing this "
                             "leaves thinking off; passing it wins over that default. Enabling it "
                             "costs output budget: the provider caps thinking and content "
                             "together, so a trace can starve the answer and the document comes "
                             "back with zero records.")
    parser.add_argument("--no-think", dest="think", action="store_false",
                        help="Force thinking off, overriding the caller.")
    parser.add_argument("--keep-alive", type=int, default=3600,
                        help="Seconds Ollama should hold the model loaded after a request. MUST be "
                             "set here: langextract sends its own keep_alive on every request "
                             "(_DEFAULT_KEEP_ALIVE = 300s) and a per-request value OVERRIDES the "
                             "server's OLLAMA_KEEP_ALIVE, so setting 24h on the service has no "
                             "effect while this backend is the caller. Any gap longer than the "
                             "value then costs a full model reload -- cheap for a 3GB model, "
                             "expensive for a 9GB one.")
    # Kept equal to sample.langextract_records.REQUEST_TIMEOUT. The caller always passes the
    # value, so this default only applies when the worker is run by hand -- and a debugging run
    # that behaves differently from the pipeline is worse than no default at all.
    parser.add_argument("--request-timeout", type=int, default=600,
                        help="Per-HTTP-request timeout to Ollama. langextract's own default is "
                             "120s, which is NOT enough here: a cold model pays its load time "
                             "inside the first request (gemma4:12b alone took 21.5s to load, a "
                             "1.5B model timed out outright at 120s), and one slow chunk fails "
                             "the whole document.")
    args = parser.parse_args()

    try:
        import langextract as lx
    except Exception as e:  # noqa: BLE001 -- a missing library is a fallback signal, not a crash
        json.dump({"error": f"langextract unavailable: {type(e).__name__}: {e}",
                   "extractions": []}, open(args.out, "w", encoding="utf-8"))
        return 0

    try:
        with open(args.text_path, encoding="utf-8", errors="ignore") as handle:
            text = handle.read()[:args.max_chars]
    except OSError as e:
        json.dump({"error": f"unreadable text: {e}", "extractions": []},
                  open(args.out, "w", encoding="utf-8"))
        return 0

    if not text.strip():
        json.dump({"error": "", "extractions": []}, open(args.out, "w", encoding="utf-8"))
        return 0

    try:
        # fence_output and use_schema_constraints are deliberately NOT passed: Ollama does not
        # support output_schema, and upstream documents that the provider factory auto-configures
        # both. Passing them explicitly is what breaks the Ollama path.
        #
        # max_char_buffer is the single biggest cost lever and langextract's default (1000 chars)
        # is far too small here: measured, a 54kB paper became ~54 sequential model calls and took
        # 2m55s. Raising it to 6000 cuts that roughly six-fold. It is only safe alongside num_ctx:
        # the chunk plus the prompt and both examples must fit the context window, or the model
        # silently sees a truncated chunk and the extraction quietly degrades. 8192 tokens against
        # a 6000-character chunk leaves ample room (~1 token per 4 chars).
        result = lx.extract(
            text_or_documents=text,
            prompt_description=PROMPT,
            examples=_examples(lx),
            model_id=args.model_id,
            model_url=args.model_url,
            max_char_buffer=args.max_char_buffer,
            max_workers=args.max_workers,
            temperature=args.temperature,
            # Threaded through to Ollama's own options dict by the provider's
            # _build_request_options, which accepts num_ctx explicitly.
            #
            # "timeout" is the provider's per-request HTTP timeout, and it must be set explicitly:
            # langextract defaults it to 120s, and on a timeout it raises InferenceRuntimeError,
            # which this worker turns into "no records" for the ENTIRE document rather than for the
            # one slow chunk. Measured: qwen2.5-coder:1.5b returned 0 records on its first document
            # purely because the model's load time landed inside that 120s window, then produced 18
            # records on the next document once warm.
            # "keep_alive" likewise has to be passed, and for a reason that is easy to miss: the
            # provider mirrors it onto the request, and a per-request keep_alive beats the server's
            # OLLAMA_KEEP_ALIVE. Observed with it unset -- `ollama ps` counting down from five
            # minutes on a host whose service was configured for 24 hours.
            # "think" is passed only when the caller decided one way or the other. The provider
            # uses setdefault, so an absent key leaves its own False in place -- which is the
            # measured-good setting for structured extraction and must stay the behaviour when
            # nothing asks otherwise.
            language_model_params={"num_ctx": args.num_ctx, "timeout": args.request_timeout,
                                   "keep_alive": args.keep_alive,
                                   **({} if args.think is None else {"think": args.think})},
            show_progress=False,
        )
    except Exception as e:  # noqa: BLE001 -- any failure degrades to "no records", caller falls back
        json.dump({"error": f"{type(e).__name__}: {e}", "extractions": []},
                  open(args.out, "w", encoding="utf-8"))
        return 0

    out = []
    for extraction in getattr(result, "extractions", []) or []:
        interval = getattr(extraction, "char_interval", None)
        out.append({
            "extraction_class": getattr(extraction, "extraction_class", ""),
            "extraction_text": getattr(extraction, "extraction_text", ""),
            "attributes": dict(getattr(extraction, "attributes", None) or {}),
            # Source grounding: the character span this extraction was taken from. This is the
            # reason to prefer this backend -- every record can be checked against the document
            # rather than trusted, which is exactly what the text path could not offer.
            "start_char": getattr(interval, "start_pos", None) if interval else None,
            "end_char": getattr(interval, "end_pos", None) if interval else None,
        })
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"error": "", "extractions": out}, handle, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
