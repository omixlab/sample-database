import os
import json
import logging
import random
import re
import threading
import time
import concurrent.futures

logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)
logging.getLogger("google.genai._api_client").setLevel(logging.WARNING)
import fitz
try:
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.llm")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ContentValidationError(Exception):
    """
    Raised when model output is structurally valid JSON but its CONTENT is unusable.

    Distinct from a parse error on purpose: the message is written to be fed back to the model as
    correction guidance, so the caller can retry with a concrete explanation of what was wrong.
    """
    pass

def validate_extraction_logic(data: Any) -> bool:
    """
    Content checks on an LLM extraction result, beyond what a JSON schema can express.

    Catches the failure modes a schema cannot: placeholder text the model emitted instead of
    admitting it found nothing, nucleotide sequences returned where an amino-acid sequence was
    asked for, the prompt's own example sequence echoed back, and a value of exactly ``0.0``
    standing in for a missing measurement.

    :param data: Parsed model output to check, walked recursively.
    :raises ContentValidationError: On the first violation, with a message written to be fed
        back to the model as correction guidance.
    :returns: ``True`` when the structure passes every check.
    """
    invalid_placeholders = {"[insert", "tbd", "n/a", "unknown", "placeholder", "sample peptide", "example peptide", "examplepeptide", "peptide a", "peptide b", "peptide 1"}
    
    def check_node(node: Any, path: str = "root") -> None:
        """
        Recursively checks one node of the parsed structure.

        :param node: Value to check; dicts and lists are walked, scalars are validated.
        :param path: Dotted path to `node`, used to make the error message locate the problem.
        :raises ContentValidationError: On the first violation found.
        :returns: ``None``.
        """
        if isinstance(node, dict):
            for k, v in node.items():
                check_node(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                check_node(item, f"{path}[{i}]")
        elif isinstance(node, str):
            if "scratchpad" in path.lower():
                return
            node_lower = node.strip().lower()
            if not node_lower:
                raise ContentValidationError(f"Empty string at {path}. DO NOT return empty strings. If the data is missing, omit the field or use null, but keep the peptide record if the name is present.")
            if any(p in node_lower for p in invalid_placeholders):
                 raise ContentValidationError(f"Prohibited placeholder in {path}: {node}. DO NOT use placeholders like N/A. If the data is missing, omit the field or use null, but keep the peptide record if the name is present.")
            
            if path.endswith("peptide_sequence"):
                import re
                seq = node.strip().upper()
                if seq != "NULL" and seq != "NONE":
                    if not re.match(r'^[A-Z-]+$', seq):
                        raise ContentValidationError(f"Invalid peptide sequence format at {path}: '{node}'. MUST contain a valid 3-letter IUPAC amino acid sequence separated by dashes, or a 1-letter sequence. If you cannot extract a valid sequence, omit the field or use null, but do NOT omit the peptide entirely if the name is present.")
                    
                    # Check for DNA/RNA sequence hallucination
                    if len(seq) >= 6 and all(char in "ACGTU" for char in seq):
                        raise ContentValidationError(f"Invalid peptide sequence at {path}: '{node}'. This appears to be a DNA/RNA nucleotide gene sequence (consists entirely of A, C, G, T, U). You MUST extract the translated AMINO ACID peptide sequence, NOT the DNA sequence. If no amino acid sequence is provided, omit the peptide entirely.")
                    
                # Check for literal copy-pasting of prompt examples or alphabetical sequences
                if seq == "ACDEFGHIKLMNPQRSTVWY" or seq == "GIGKFLHSAKKFGKAFVGEIMNS" or seq.startswith("ACDEF"):
                    raise ContentValidationError(f"Hallucination detected at {path}: '{node}'. You generated an alphabetical or example sequence. You MUST extract the actual sequence from the document. If no peptide is found, return an empty array.")
        elif isinstance(node, (int, float)):
            if "value_lower" in path:
                if float(node) == 0.0:
                    raise ContentValidationError(f"Invalid value_lower of {node} at {path}. MIC/IC50/EC50 cannot be exactly 0.0. If the numerical value is absent in the document, use null. Do NOT default to 0.0.")

    check_node(data)
    return True

def clean_json(raw: Any) -> Any:
    """
    Cleans raw LLM output and extracts the JSON block.
    """
    if not isinstance(raw, str):
        return raw
    import re
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    
    import re
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        
    start_obj = raw.find('{')
    end_obj = raw.rfind('}')
    start_arr = raw.find('[')
    end_arr = raw.rfind(']')
    
    if start_obj != -1 and end_obj != -1 and (start_arr == -1 or start_obj < start_arr):
        raw = raw[start_obj:end_obj+1]
    elif start_arr != -1 and end_arr != -1:
        raw = raw[start_arr:end_arr+1]
        
    try:
        parsed = json.loads(raw, strict=False)
    except Exception:
        try:
            import json_repair
            parsed = json_repair.repair_json(raw, return_objects=True)
            if not parsed:
                raise ValueError("json_repair returned empty")
        except Exception as e:
            raise ValueError(f"JSON Decode Error and Repair Failed: {e}. Raw output: {raw[:200]}...")
    if isinstance(parsed, list):
        parsed = {"scratchpad": "Auto-recovered list output", "peptides": parsed}
    
    if isinstance(parsed, dict) and "scratchpad" in parsed:
        if not isinstance(parsed["scratchpad"], str):
            parsed["scratchpad"] = json.dumps(parsed["scratchpad"])
            
    if isinstance(parsed, dict) and "peptides" in parsed:
        if isinstance(parsed["peptides"], dict):
            parsed["peptides"] = []
            
    # Auto-clean strings
    def replace_null_strings(node: Any) -> None:
        """
        Rewrites the string literals models emit for absent values into real ``None``.

        :param node: Container to walk in place.
        :returns: ``None``.
        """
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, str) and v.strip().lower() in ["null", "none", "n/a"]:
                    node[k] = None
                else:
                    replace_null_strings(v)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                if isinstance(v, str) and v.strip().lower() in ["null", "none", "n/a"]:
                    node[i] = None
                else:
                    replace_null_strings(v)
                    
    replace_null_strings(parsed)

    # Auto-clean sequences so LLM isn't overly penalized for minor formatting
    if isinstance(parsed, dict) and "peptides" in parsed and isinstance(parsed["peptides"], list):
        import re
        three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        one_to_three = {v: k.capitalize() for k, v in three_to_one.items()}
        
        for pep in parsed["peptides"]:
            if not isinstance(pep, dict):
                continue
            seq = pep.get("peptide_sequence")
            name = pep.get("peptide_name", "")
            
            # If sequence is missing but the name contains a sequence-like string, try to use it
            if isinstance(seq, str):
                s = seq.strip()
                if s.upper() in ["NULL", "NONE", "N/A"]:
                    s = None
            else:
                s = None
                
            if not s and isinstance(name, str) and "-" in name:
                parts = name.split(":")
                s = parts[-1].strip()
                
            if s:
                # Strip common terminal mods
                s = re.sub(r'(?i)^(NH2|AC|FITC|PEG|H2N)[-_\s]*', '', s)
                s = re.sub(r'(?i)[-_\s]*(NH2|COOH|OH|AMIDE)$', '', s)
                
                parts = re.split(r'[-_\s]+', s)
                cleaned = []
                
                if len(parts) > 1 and any(len(p) >= 3 for p in parts):
                    # Already separated, probably 3-letter codes
                    for p in parts:
                        p_upper = p.upper()
                        # Extract potential stereochemistry
                        stereo = ""
                        if p_upper.startswith("D-") or p_upper.startswith("L-"):
                            stereo = p[:2]
                            p = p[2:]
                            p_upper = p.upper()
                            
                        if p_upper in three_to_one:
                            cleaned.append(stereo + p.capitalize())
                        elif len(p_upper) == 1 and p_upper in one_to_three:
                            cleaned.append(stereo + one_to_three[p_upper])
                        else:
                            cleaned.append(stereo + p.capitalize())
                else:
                    # Continuous string (either 1-letter or unseparated 3-letter)
                    s_upper = s.upper()
                    s_upper = re.sub(r'[^A-Z]', '', s_upper)
                    
                    is_unseparated_3_letter = False
                    if len(s_upper) >= 3 and len(s_upper) % 3 == 0:
                        chunks = [s_upper[i:i+3] for i in range(0, len(s_upper), 3)]
                        if all(c in three_to_one for c in chunks):
                            cleaned = [c.capitalize() for c in chunks]
                            is_unseparated_3_letter = True
                            
                    if not is_unseparated_3_letter:
                        # Assume 1-letter codes
                        for char in s_upper:
                            if char in one_to_three:
                                cleaned.append(one_to_three[char])
                                
                if cleaned:
                    pep["peptide_sequence"] = "-".join(cleaned)
                else:
                    pep["peptide_sequence"] = None
            else:
                pep["peptide_sequence"] = None

    return parsed

# Ollama model families that emit a separate chain-of-thought channel. Kept as an explicit list
# because the failure mode for a wrong guess is a hang, not an error (see the `think` handling in
# llm_call). Add a family here only after verifying it accepts the parameter.
_REASONING_MODEL_FAMILIES = ("gemma4", "qwen3", "deepseek-r1", "phi4")


def _is_reasoning_model(model_name: str) -> bool:
    """
    :param model_name: Ollama model tag, e.g. ``"gemma4:latest"``.
    :returns: ``True`` for families that emit a thinking trace, which must be disabled or it
        consumes the whole token budget and the completion comes back empty.
    """
    name = (model_name or "").lower()
    return any(fam in name for fam in _REASONING_MODEL_FAMILIES)


# Client-side cap on concurrent Ollama requests, matched to the server's own parallelism.
#
# Ollama accepts OLLAMA_NUM_PARALLEL requests at a time and queues the rest; when that queue fills
# it answers 503 "maximum pending requests exceeded". Observed on a real run: the extractor issued
# far more concurrent calls than the server would run, and the 503s landed on whichever layer
# happened to be calling -- disproportionately candidate triage, which fails open, so those
# documents silently kept every junk candidate. Throttling here means the pipeline waits for a
# slot instead of being told to go away, which is the difference between slower and wrong.
#
# Sized from the server's OLLAMA_NUM_PARALLEL when it is visible in this environment. The default
# is deliberately small: over-subscribing is what caused the problem.
_OLLAMA_MAX_INFLIGHT = max(1, int(os.getenv("OLLAMA_NUM_PARALLEL", "4")))
_OLLAMA_SLOTS = threading.BoundedSemaphore(_OLLAMA_MAX_INFLIGHT)
_OLLAMA_SLOT_WAIT = 120.0

# 503/backpressure is a transient "try later", not a failure -- retrying it keeps the verification
# layers doing their job instead of failing open and degrading the data.
_OVERLOAD_MARKERS = ("503", "server busy", "maximum pending requests", "no slots available")
_OVERLOAD_RETRIES = 4
_OVERLOAD_BACKOFF_BASE = 0.75

# Per-call deadline for the structured extraction/verification calls. Raised from 30s because that
# budget now has to cover waiting for a free slot and up to three backoff sleeps (~5s total) on top
# of the call itself; leaving it at 30s would turn contention into TimeoutErrors, which the circuit
# breakers count as real failures and which is exactly the fail-open behaviour being fixed here.
_LLM_CALL_TIMEOUT = float(os.getenv("SAMPLE_LLM_TIMEOUT", "90"))

# Thinking mode and its token budget are defined in sample.config, because the context rungs
# there depend on the same switch. Re-exported under the names this module has always used.
from sample.config import LLM_THINK as THINK_ENABLED          # noqa: E402
from sample.config import THINK_MIN_TOKENS, THINK_TOKEN_FACTOR  # noqa: E402


#: Per-layer call accounting: {layer: [calls, seconds, failures]}.
#:
#: Added because throughput on this host is bounded by the model server, and the question that
#: decides what to optimise -- WHICH layer is spending the GPU -- was being answered by
#: inference. Measured 2026-10-16: LLM calls/minute stayed flat at 24-30 while documents/minute
#: fell threefold, so cost per document rose rather than the server slowing down. That says the
#: answer is in the per-layer mix, and nothing was recording it.
_CALL_STATS: Dict[str, List[float]] = {}
_CALL_STATS_LOCK = threading.Lock()

#: How often the accumulated per-layer summary is logged, in calls.
_CALL_STATS_EVERY = int(os.getenv("SAMPLE_LLM_STATS_EVERY", "200"))


def record_call(layer: str, seconds: float, failed: bool = False) -> None:
    """
    Accounts one LLM call against its layer.

    :param layer: Short layer name, e.g. ``"triage"``.
    :param seconds: Wall-clock duration of the call.
    :param failed: Whether it raised.
    :returns: ``None``.
    """
    with _CALL_STATS_LOCK:
        entry = _CALL_STATS.setdefault(layer or "unlabelled", [0.0, 0.0, 0.0])
        entry[0] += 1
        entry[1] += seconds
        entry[2] += 1 if failed else 0
        total = sum(int(value[0]) for value in _CALL_STATS.values())
        due = _CALL_STATS_EVERY > 0 and total % _CALL_STATS_EVERY == 0
        snapshot = dict(_CALL_STATS) if due else None
    if snapshot:
        parts = []
        for name, (calls, elapsed, failures) in sorted(
                snapshot.items(), key=lambda item: -item[1][1]):
            share = 100.0 * elapsed / max(sum(v[1] for v in snapshot.values()), 1e-9)
            parts.append(f"{name}={int(calls)} calls/{elapsed:.0f}s ({share:.0f}% of LLM time"
                         f"{f', {int(failures)} failed' if failures else ''})")
        logger.info(f"LLM time by layer: {'; '.join(parts)}")


def call_stats() -> Dict[str, List[float]]:
    """:returns: A copy of the per-layer accounting, for a caller that wants to report it."""
    with _CALL_STATS_LOCK:
        return {name: list(value) for name, value in _CALL_STATS.items()}


def inflight_calls() -> int:
    """
    :returns: How many LLM calls this process currently has in flight, counted from the
        concurrency semaphore.

    Exists for the chunk barrier in :mod:`sample.pipeline_runner`, which must not reset the
    Ollama runtime while a request is still outstanding -- an unload during a generation gives
    that document a hard failure for no reason. The pool having drained is the primary
    guarantee; this is the independent check on it, and it is the one that would catch a call
    abandoned by the timeout path above, which by design outlives the caller that started it.

    Reads the semaphore's private counter because BoundedSemaphore exposes no public one.
    Diagnostic only: never branch correctness on it, since it is read without the lock.
    """
    return max(0, _OLLAMA_MAX_INFLIGHT - _OLLAMA_SLOTS._value)


def _is_overload_error(exc: BaseException) -> bool:
    """
    :param exc: Exception raised by the model backend.
    :returns: ``True`` if it indicates the server is overloaded and the call is worth retrying.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _OVERLOAD_MARKERS)


def _keep_alive() -> str:
    """
    :returns: The ``keep_alive`` value sent to Ollama, from ``OLLAMA_KEEP_ALIVE``. Keeping a large
        model resident avoids a multi-second reload between the pipeline's many short calls.
    """
    """
    Ollama keep_alive value, from OLLAMA_KEEP_ALIVE (see sample/config.py). Numeric strings are
    passed as seconds (Ollama accepts an int); duration strings like "1h"/"10m" pass through.
    """
    from sample.config import OLLAMA_KEEP_ALIVE
    raw = str(OLLAMA_KEEP_ALIVE).strip()
    try:
        return int(raw)
    except ValueError:
        return raw


def llm_call(messages: List[Dict[str, str]], temperature: float = 0.0,
             response_schema: Optional[Dict[str, Any]] = None, model: Optional[str] = None,
             max_tokens: Optional[int] = None, timeout: Optional[float] = None,
             layer: str = "", **kwargs: Any) -> str:
    """
    Unified LLM caller. Runs against the local Ollama model by default; a remote provider is used
    only when a caller names one explicitly and that provider's API key is present. Supports an
    optional timeout.

    :param layer: Which pipeline layer is calling, for the per-layer time accounting in
        :func:`record_call`. Optional and defaulted so an un-updated caller still works; such a
        call is accounted as ``"unlabelled"`` rather than dropped, since a layer missing from the
        summary would make the mix look cleaner than it is.
    """
    import os
    from sample.config import DEFAULT_EXTRACT_MODEL
    gemini_key = os.environ.get("GEMINI_API_KEY")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    # An omitted `model` must not route off-box. This defaulted to "gemini-flash", so any caller
    # that forgot the argument would have sent document text to a remote API; every current caller
    # passes DEFAULT_EXTRACT_MODEL, and this makes that the floor rather than a convention.
    target_model = model or DEFAULT_EXTRACT_MODEL

    # PER-DOCUMENT BUDGET. Extraction gives each document a wall-clock budget (see
    # sample.deadlines), and a document makes many LLM calls: triage, activity verification,
    # organism normalisation, activity recovery, validation, unit adjudication. Clamping here
    # rather than at each of those call sites is what makes the budget total instead of
    # per-call -- six helpers each honouring a 90s timeout is a 540s document, which is how a
    # "300s per paper" cap turns into no cap at all.
    #
    # Refusing outright once the budget is spent is the load-bearing half. A call issued with
    # seconds left cannot return in time, but Ollama still runs the generation to completion
    # after the client gives up, so it occupies a slot the NEXT document needs. That is the
    # mechanism by which an over-budget document steals throughput from documents that are
    # still within theirs.
    from sample import deadlines
    if deadlines.expired():
        # Accounted, with zero seconds: no request was made, so this must not inflate the
        # layer's share of LLM time -- but a layer quietly losing calls to the budget is
        # exactly what the summary needs to show, and an unrecorded refusal shows nothing.
        record_call(layer, 0.0, failed=True)
        raise deadlines.BudgetExhausted(
            f"document budget exhausted ({deadlines.elapsed():.0f}s of "
            f"{deadlines.budget():.0f}s used); not starting another {target_model} call")
    timeout = deadlines.clamp(timeout)

    def _do_call() -> str:
        """
        Issues the request itself, inside the concurrency semaphore.

        :returns: The model's raw completion text.
        """
        if "deepseek" in target_model.lower() and "ollama" not in target_model.lower():
            import requests
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {deepseek_key}"
            }
            # Remove images from messages for DeepSeek text API, send just text
            ds_messages = []
            for msg in messages:
                ds_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            payload = {
                "model": target_model,
                "messages": ds_messages,
                "temperature": temperature
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if response_schema:
                payload["response_format"] = {"type": "json_object"}
            
            try:
                resp = requests.post("https://api.deepseek.com/chat/completions", json=payload, headers=headers, timeout=timeout or 300)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"DeepSeek API failed: {e}. Falling back to Ollama.")
                pass

        if "gemini" in target_model.lower():
            from google import genai
            from google.genai import types
                
            # Default to gemini-2.5-flash if gemini is targeted
            m_name = target_model if "gemini" in target_model.lower() else "gemini-2.5-flash"
            
            # Handle shorthand names
            if m_name in ["gemini", "gemini-flash", "gemini flash", "gemini-pro", "gemini pro"]:
                if "pro" in m_name:
                    m_name = "gemini-2.5-pro"
                else:
                    m_name = "gemini-2.5-flash"
                
            import threading
            if not hasattr(llm_call, "local_data"):
                llm_call.local_data = threading.local()
            if not hasattr(llm_call.local_data, "client"):
                llm_call.local_data.client = genai.Client(api_key=gemini_key)
            client = llm_call.local_data.client
            
            # Combine messages for the new API
            single_prompt = "\n\n".join([msg["content"] for msg in messages])
                    
            config_kwargs = {"temperature": temperature}
            if response_schema:
                config_kwargs["response_mime_type"] = "application/json"
            if max_tokens:
                config_kwargs["max_output_tokens"] = max_tokens
                
            try:
                response = client.models.generate_content(
                    model=m_name,
                    contents=single_prompt,
                    config=types.GenerateContentConfig(**config_kwargs)
                )
                return response.text
            except Exception as e:
                # Fall back to ollama if Gemini fails (e.g. no key)
                print(f"Gemini API failed: {e}. Falling back to Ollama.")
                pass

        # Use local Ollama fallback
        import ollama
        ollama_client = ollama.Client(timeout=timeout) if timeout and timeout > 0 else ollama

        ollama_model = target_model if "gemini" not in target_model.lower() else "deepseek-r1:1.5b"
        
        # Adapt messages for Ollama (it supports {"role", "content", "images"})
        adapted_messages = []
        for msg in messages:
            adapted_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            if "images" in msg:
                adapted_msg["images"] = msg["images"]
            adapted_messages.append(adapted_msg)
            
        # Context window is fitted to the actual prompt instead of a blanket 8192.
        #
        # Ollama/llama.cpp allocates the KV cache from num_ctx up front, per parallel slot, and
        # holds it for as long as the model stays loaded -- so a fixed 8192 reserved the full
        # window for every call regardless of size, multiplied by OLLAMA_NUM_PARALLEL slots and
        # paid even by the many short calls this pipeline makes (an activity-verification prompt
        # is a few hundred tokens). Fitting the window to the request keeps that reservation
        # proportional to what is actually being asked.
        #
        # ~3.5 chars/token is a deliberately conservative estimate for English prose mixed with
        # JSON and chemical notation (real ratios run higher, so this over-estimates tokens and
        # therefore never under-allocates). The floor keeps small calls from thrashing; the
        # ceiling preserves the previous maximum so no existing caller's prompt can be truncated
        # by this change. An explicit num_ctx kwarg still wins outright.
        # PINNED, not fitted. The paragraph above describes the KV-cache saving a fitted window
        # gives; what it did not account for is that Ollama restarts the runner whenever a request
        # needs more context than the loaded one has. A continuously varying num_ctx therefore
        # rebuilt the runner over and over -- measured at 7 restarts in 10 minutes across three
        # sizes -- discarding the very cache the fitting was protecting. See
        # sample.config.DEFAULT_NUM_CTX.
        #
        # The prompt is still measured, but only to decide whether it FITS. A prompt that does not
        # is escalated rather than truncated: one reload is cheap, a silently truncated prompt is
        # a wrong answer. After escalating, the larger runner serves the small calls too, so this
        # converges upward instead of oscillating.
        if "num_ctx" in kwargs:
            num_ctx = kwargs["num_ctx"]
        else:
            from sample.config import DEFAULT_NUM_CTX, FALLBACK_NUM_CTX

            prompt_chars = sum(len(m.get("content") or "") for m in adapted_messages)
            needed = int(prompt_chars / 3.5) + (max_tokens or 512) + 256  # +256 chat/template margin
            # SNAP to a rung. Asking for `needed` itself -- even bounded -- reintroduces the
            # problem: a prompt wanting 4196 tokens would get a 4196-token window, the next
            # wanting 5100 another, and the runner rebuilds for each. Two sizes only, so it can be
            # rebuilt at most once per run.
            num_ctx = DEFAULT_NUM_CTX
            if needed > DEFAULT_NUM_CTX:
                num_ctx = FALLBACK_NUM_CTX
                logger.debug(
                    "prompt needs ~%d tokens, above the pinned num_ctx of %d; escalating to %d. "
                    "Costs at most one runner reload for the whole run.",
                    needed, DEFAULT_NUM_CTX, num_ctx)
            if needed > FALLBACK_NUM_CTX:
                # Above the top rung the prompt WILL be truncated by llama.cpp, silently. Said
                # loudly rather than quietly widened to an arbitrary size: the caller should chunk
                # its input (langextract already does, at LANGEXTRACT_MAX_CHAR_BUFFER), and a
                # third runner size would bring the restarts back.
                logger.warning(
                    "prompt needs ~%d tokens but the largest pinned window is %d; it will be "
                    "TRUNCATED. Chunk the input instead of relying on the context window.",
                    needed, FALLBACK_NUM_CTX)
        options = {"temperature": temperature, "num_ctx": num_ctx}
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Deterministic, structure-friendly sampling for temperature-0 work (every extraction,
        # verification and triage call in this pipeline). temperature=0 alone does NOT make Ollama
        # greedy: top_k defaults to 40 and a sampler still picks among candidates, which was
        # observed directly -- the same verification prompt returned "kept 3/3" and "kept 1/3" on
        # consecutive identical runs. Pinning top_k/top_p/seed makes a rerun reproduce a run, which
        # is what lets a corpus extraction be audited and re-derived.
        #
        # repeat_penalty is the important one for schema-constrained output: Ollama defaults it to
        # 1.1, which penalizes exactly the tokens a JSON array must repeat ("id", "supported", the
        # braces) once per element, progressively distorting later elements of a long verdict list.
        # Neutralized to 1.0 -- there is no rambling to suppress when the grammar already
        # constrains the output.
        if temperature == 0:
            options.update({"top_k": 1, "top_p": 1.0, "seed": 0, "repeat_penalty": 1.0})
            
        # Reasoning models write their chain of thought to message.thinking and the actual answer
        # to message.content, and num_predict caps the two combined. Every structured call this
        # pipeline makes asks for a short JSON verdict, so the model would spend the whole budget
        # thinking and return content="" with done_reason="length" -- which is exactly what the
        # "gemma4 returns an empty completion" symptom was. Measured on one verification prompt:
        # thinking enabled gave "" in 7.1s, think=False gave the correct {"supported": false} in
        # 0.3s. Chain of thought buys nothing for a boolean verdict under a fixed grammar.
        #
        # Gated by model family rather than passed unconditionally: sending `think` to a model
        # without thinking support does not raise, it HANGS (verified with qwen2.5-coder:1.5b,
        # which never returned), so it can't be guarded with try/except.
        api_kwargs = {
            "model": ollama_model,
            "messages": adapted_messages,
            "options": options,
            # Keeps the model resident between calls. With a 9.6GB model this is the difference
            # between a sub-second warm call and a multi-second reload from disk into VRAM, and
            # this pipeline makes many short calls separated by pure-Python work during which
            # Ollama's 5-minute default can evict it. See OLLAMA_KEEP_ALIVE in sample/config.py.
            "keep_alive": _keep_alive(),
        }
        if _is_reasoning_model(ollama_model):
            api_kwargs["think"] = THINK_ENABLED
            if THINK_ENABLED and options.get("num_predict"):
                # The budget has to grow with the trace, or enabling thinking just reproduces the
                # empty-completion bug: num_predict caps thinking + content TOGETHER, so a 384-
                # token budget spent on reasoning leaves nothing for the answer. This is the
                # difference between think=True working and think=True silently returning "".
                options["num_predict"] = max(int(options["num_predict"] * THINK_TOKEN_FACTOR),
                                             THINK_MIN_TOKENS)
        
        # Do not enforce structured JSON formatting on DeepSeek or Qwen3 reasoning models, as it
        # prevents them from generating <think> reasoning trace blocks or causes hangs. This used
        # to blanket-exclude every "qwen"-named model, silently no-op'ing response_schema for
        # non-reasoning Qwen models too (e.g. sample/llm_validator.py has been passing
        # response_schema for "qwen3.5:9b" -- itself still excluded here, unchanged -- and
        # recover_peptide_llm's default "qwen2.5-coder:1.5b" was never able to use it at all).
        # Thinking mode is a Qwen3-specific feature; qwen2.5 has no such mode to break. Verified
        # directly: qwen2.5-coder:1.5b with format=schema returns valid, parseable JSON in ~2s
        # across repeated trials, no hangs.
        if response_schema and "deepseek" not in ollama_model.lower() and "qwen3" not in ollama_model.lower():
            api_kwargs["format"] = response_schema
            
        # Wait for a free slot rather than adding to a queue the server is already refusing, and
        # retry the overload response itself with exponential backoff + jitter. Jitter matters
        # because several worker threads hit the wall at the same instant; without it they would
        # retry in lockstep and re-create the burst that caused the 503.
        last_exc = None
        for attempt in range(_OVERLOAD_RETRIES):
            if not _OLLAMA_SLOTS.acquire(timeout=_OLLAMA_SLOT_WAIT):
                raise TimeoutError(
                    f"Timed out after {_OLLAMA_SLOT_WAIT:.0f}s waiting for a free Ollama slot "
                    f"({_OLLAMA_MAX_INFLIGHT} in flight)."
                )
            try:
                response = ollama_client.chat(**api_kwargs)
                return response["message"]["content"]
            except Exception as e:
                last_exc = e
                if not _is_overload_error(e) or attempt == _OVERLOAD_RETRIES - 1:
                    raise
            finally:
                _OLLAMA_SLOTS.release()

            delay = _OVERLOAD_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.4)
            logger.debug(
                f"Ollama busy ({last_exc}); retry {attempt + 1}/{_OVERLOAD_RETRIES - 1} in {delay:.1f}s"
            )
            time.sleep(delay)
        raise last_exc

    # Accounted on both paths, and a failure is recorded as a failure rather than dropped: a
    # layer whose calls mostly fail would otherwise look cheap in the summary.
    started = time.monotonic()
    try:
        result = _llm_call_timed(_do_call, timeout)
    except BaseException:
        record_call(layer, time.monotonic() - started, failed=True)
        raise
    record_call(layer, time.monotonic() - started)
    return result


def _llm_call_timed(_do_call: Any, timeout: Optional[float]) -> str:
    """
    Applies the optional timeout around one prepared call.

    :param _do_call: The zero-argument closure that issues the request.
    :param timeout: Seconds, or ``None`` for no bound.
    :returns: The model's completion text.
    """
    if timeout is not None and timeout > 0:
        import concurrent.futures
        # NOT a `with` block, and that is the entire point of this shape. A
        # ThreadPoolExecutor's __exit__ calls shutdown(wait=True), which blocks until the
        # submitted task finishes -- so `future.result(timeout=...)` would raise TimeoutError
        # and then the context manager would sit there waiting for the very call that timed
        # out. The timeout was reported at the right moment and enforced at none: callers were
        # billed the full duration of a call they had already given up on, which is why a
        # per-document cap built on these timeouts appeared to do nothing.
        #
        # shutdown(wait=False) returns immediately. The abandoned thread still runs to
        # completion (nothing can interrupt it), but it no longer holds the caller, and the
        # Ollama slot it occupies is released by the semaphore in _do_call's `finally` either
        # way. cancel_futures has no effect on an already-started task and is passed only so a
        # queued one -- impossible here with max_workers=1 and a single submit, but cheap to be
        # correct about -- is dropped rather than run after everyone stopped caring.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            return executor.submit(_do_call).result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
    else:
        return _do_call()
_RECOVERY_COOLDOWN_SECONDS = 30.0


class _CircuitBreaker:
    """
    Trips only after repeated *consecutive* failures, and reports how much work it skipped.

    Replaces a plain "unavailable until T" timestamp that each LLM layer kept independently. That
    design conflated two very different situations -- a backend that is down, and a single call
    that timed out -- and treated both as "disable this layer for 30 seconds". Because the state
    is global and the pipeline runs several worker threads, one slow call silently disabled a
    whole verification layer for every document processed in the next 30 seconds. Candidate triage
    was the visible casualty: it fails open by design, so the skipped documents kept all their
    junk candidates and nothing in the logs said why.

    A transient timeout now costs one call, not a window of them. The breaker opens only when
    `threshold` calls fail in a row (a real outage), and when it closes it logs the number of
    calls that were skipped, so a degraded run is visible instead of silently producing worse data.
    """

    def __init__(self, name: str, threshold: int = 3,
                 cooldown: float = _RECOVERY_COOLDOWN_SECONDS) -> None:
        """
        :param name: Layer label used in log messages.
        :param threshold: Consecutive failures required to open the breaker.
        :param cooldown: Seconds to stay open before retrying.
        """
        self._name = name
        self._threshold = threshold
        self._cooldown = cooldown
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._open_until = 0.0
        self._skipped = 0

    def is_open(self) -> bool:
        """:returns: ``True`` if calls should be skipped right now."""
        with self._lock:
            if self._open_until and time.monotonic() >= self._open_until:
                skipped, self._open_until, self._skipped = self._skipped, 0.0, 0
                self._consecutive_failures = 0
                logger.warning(
                    f"{self._name}: resuming after backoff ({skipped} call(s) skipped while unavailable)."
                )
            if self._open_until:
                self._skipped += 1
                return True
        return False

    def record_success(self) -> None:
        """:returns: ``None``. Resets the consecutive-failure counter."""
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self, error: Optional[BaseException] = None) -> None:
        """
        Counts a failure and opens the breaker once `threshold` land in a row.

        A :class:`sample.deadlines.BudgetExhausted` is not counted. It means this document ran
        out of its own time, not that the backend is unhealthy -- and because one over-budget
        document refuses several calls back to back, counting them would open the breaker and
        disable the layer for every document in flight alongside it.
        """
        from sample.deadlines import BudgetExhausted
        if isinstance(error, BudgetExhausted):
            return
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold and not self._open_until:
                self._open_until = time.monotonic() + self._cooldown
                logger.warning(
                    f"{self._name}: {self._consecutive_failures} consecutive failures "
                    f"(last: {error}); backing off for {self._cooldown:.0f}s."
                )


_TRIAGE_BREAKER = _CircuitBreaker("LLM candidate triage")
_VERIFY_BREAKER = _CircuitBreaker("LLM activity verification")
_ORGNORM_BREAKER = _CircuitBreaker("LLM organism normalization")
_RECOVERY_BREAKER = _CircuitBreaker("LLM activity recovery")
_VALIDATION_BREAKER = _CircuitBreaker("LLM extraction validation")   # step 13, sample/llm_validator.py
#: Its own breaker rather than sharing _VALIDATION_BREAKER: unit adjudication runs on ~4.6% of
#: documents, so a run of failures there would otherwise open the breaker that gates extraction
#: validation and silently disable a layer that has nothing to do with units.
_UNIT_BREAKER = _CircuitBreaker("LLM concentration-unit adjudication")

_ACTIVITY_RECOVERY_SCHEMA = {
    "type": "object",
    "properties": {
        "biological_activities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_organism": {"type": "string"},
                    "target_group": {"type": "string"},
                    "activity_type": {"type": "string"},
                    "value_raw": {"type": "string"},
                    "activity_unit": {"type": "string"},
                    "supporting_quote": {"type": "string"},
                },
                "required": ["target_organism", "activity_type", "value_raw", "activity_unit", "supporting_quote"],
            },
        }
    },
    "required": ["biological_activities"],
}



def _verdict_schema(array_name: str, verdict_field: str, n_items: int) -> dict:
    """
    Per-call schema for a verdict array.

    `n_items` is accepted but deliberately NOT encoded as minItems/maxItems: Ollama 0.24.0's
    JSON-schema-to-grammar conversion cannot express array length bounds and returns an EMPTY
    completion when they are present, which -- because this layer fails open -- silently accepted
    every candidate. Verified directly: the identical call returns valid JSON without the bounds
    and an empty string with them. The count is kept in the signature so the caller's intent stays
    visible, and completeness is enforced in the prompt and by the caller's own accounting.
    """
    return {
        "type": "object",
        "properties": {
            array_name: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        verdict_field: {"type": "boolean"},
                    },
                    "required": ["id", verdict_field],
                },
            }
        },
        "required": [array_name],
    }

# Bounds chosen to keep this one call cheap and its context window small (see llm_call's fitted
# num_ctx). 40 candidates x ~90 chars of evidence is roughly 1.2k tokens of prompt -- a document
# with more borderline candidates than this is dominated by OCR noise anyway, and truncating the
# list keeps a pathological document from allocating a large context.
# Ballot size. Lowered from 40 after measuring accuracy decay on long lists: on a 45-candidate
# ballot the model rejected LL-37 and admitted five obvious buffer entries, while on ballots of
# ~10 it was perfect on the same material. Documents rarely exceed this, so the extra call only
# affects the OCR-noisy outliers -- which are exactly the ones whose verdicts need to be good.
_TRIAGE_MAX_CANDIDATES = 20
_TRIAGE_SNIPPET_CHARS = 90


def triage_candidates_llm(candidates: list, doc_id: str = "", model: str = None) -> set:
    """
    Judges which name-only candidates are actually peptides/proteins studied in the paper, as
    opposed to the open-ended junk classes that share their surface shape: bench chemicals
    (BCNU/carmustine), microbial strain codes (CH-112), instrument and column models (DU-8, RP-8),
    plasmids (pUC19), cell lines, and OCR fragments of a real name run together with a table
    column ("FK-5650" from "FK-565" + its dose).

    Those classes are why the blocklist approach kept losing ground: each is unbounded, so
    enumerating them is endless, but all of them are trivially separable by anyone (or anything)
    that can read the sentence they appear in. This asks the model that question directly.

    Applied ONLY to candidates with no recovered sequence -- a candidate carrying a real amino-acid
    sequence is already strongly grounded, and running it past a small model risks discarding good
    data on a bad generation.

    ALL candidates for a document go in ONE call, and the function fails OPEN (on error, timeout,
    or an unusable response it returns every candidate as accepted). Same reasoning as
    verify_activities_llm: deterministic filtering upstream is what establishes correctness, and a
    verification layer that fails closed is how the NLI gate silently emptied the database.

    `candidates` is a list of (name, evidence_snippet). Returns the set of accepted names.
    """
    accepted = {name for name, _ in candidates}
    if not candidates:
        return accepted
    if _TRIAGE_BREAKER.is_open():
        return accepted

    # Candidates past the prompt-size cap are judged in further calls rather than waved through.
    # They used to be added back unconditionally as "unjudged", which meant a document with more
    # than _TRIAGE_MAX_CANDIDATES name-only candidates kept every one beyond the cap -- and the
    # documents with the most candidates are exactly the OCR-noisy ones with the most junk. The
    # batches are independent one-call judgements, so this costs an extra call only for the
    # documents that actually overflow.
    if len(candidates) > _TRIAGE_MAX_CANDIDATES:
        kept = set()
        for start in range(0, len(candidates), _TRIAGE_MAX_CANDIDATES):
            kept |= triage_candidates_llm(
                candidates[start:start + _TRIAGE_MAX_CANDIDATES], doc_id=doc_id, model=model)
        return kept & accepted

    from sample.config import DEFAULT_TRIAGE_MODEL
    target_model = model or DEFAULT_TRIAGE_MODEL

    subset = candidates
    lines = []
    for i, (name, snippet) in enumerate(subset):
        snippet = " ".join((snippet or "").split())[:_TRIAGE_SNIPPET_CHARS]
        lines.append(f'{i}. "{name}" -- context: ...{snippet}...')

    # Written for DEFAULT_TRIAGE_MODEL (a 1.5B model), where the labelled examples are load-bearing
    # rather than decorative: with only an abstract description of the rejection classes it kept a
    # plasmid (pUC19), a non-peptide chemotherapeutic (BCNU/carmustine) and an OCR fragment
    # ("FK-5650"), defaulting to "yes" for anything shaped like a compound code. Naming each class
    # with a concrete instance and its reason took it from 2/8 to 7/8. Do not port this prompt to a
    # larger model without re-measuring -- gemma4 returns an empty completion for it (see
    # DEFAULT_TRIAGE_MODEL in sample/config.py).
    system = (
        "You decide whether a name from a scientific paper is a PEPTIDE or PROTEIN that the paper "
        "studies as a therapeutic/antimicrobial agent.\n"
        "Answer is_peptide=true ONLY for peptides, proteins, or peptide-nature antibiotics.\n"
        "Answer is_peptide=false for everything else. Examples of false:\n"
        '- "BCNU" -> a small-molecule chemotherapeutic (carmustine), not a peptide\n'
        '- "doxorubicin" -> a small-molecule drug, not a peptide\n'
        '- "pUC19" -> a plasmid vector\n'
        '- "CH-112" -> a bacterial strain code\n'
        '- "ATCC 25922" -> a culture-collection number\n'
        '- "MCF-7" -> a cell line\n'
        '- "RP-8" -> an HPLC column\n'
        '- "DMSO" -> a solvent; "FBS" -> a reagent; "RPMI-1640" -> a culture medium\n'
        '- "FK-5650" -> a garbled OCR fragment: a real name ("FK-565") fused to a table number\n'
        # The classes below were the most frequent junk names in a 22,050-document corpus audit and
        # were all accepted by the previous prompt. Each names a real thing, which is why an
        # abstract "is it a peptide?" question was not enough -- the model has to be told that a
        # property, a class and a component are none of them a peptide's NAME.
        '- "cationic", "apoptotic", "anionic" -> adjectives describing a property, not a name\n'
        '- "DE3", "BL21", "PAO1", "K12" -> expression hosts and reference strains\n'
        '- "MB-231" -> a fragment of the cell line MDA-MB-231\n'
        '- "pyruvate", "glutamate" -> metabolites/media components\n'
        '- "lectin", "cathepsin", "RNase" -> a protein CLASS or enzyme family, not one peptide\n'
        '- "oligo", "combi" -> clipped lab words (oligonucleotide, combination)\n'
        '- "transferrin", "heparin" -> real molecules, but not peptide therapeutics under study\n'
        "Examples of true:\n"
        '- "magainin 2", "LL-37", "nisin A", "defensin HNP-1", "polymyxin B", "bacitracin"\n'
        '- "SP4080", "FK-565" -> paper-local codes for a specific peptide under study\n'
        "Answer only with the JSON schema given."
    )
    user = "\n".join(lines) + (
        '\n\nReturn a verdict for EVERY id above, including the ones that are not peptides. '
        'Return {"peptides":[{"id":<int>,"is_peptide":<bool>}]}'
    )

    try:
        raw = llm_call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, model=target_model, max_tokens=384, timeout=_LLM_CALL_TIMEOUT,
            response_schema=_verdict_schema("peptides", "is_peptide", len(subset)),
            layer="triage",
        )
    except Exception as e:
        logger.warning(f"[{doc_id}] LLM candidate triage unavailable ({e}); keeping all candidates.")
        _TRIAGE_BREAKER.record_failure(e)
        return accepted
    _TRIAGE_BREAKER.record_success()

    try:
        parsed = clean_json(raw)
    except Exception:
        parsed = None
    verdicts = parsed.get("peptides") if isinstance(parsed, dict) else None
    if not isinstance(verdicts, list):
        # No parseable ballot at all -- transport or generation failure. Fail open.
        return accepted

    # Read as a complete ballot: an id the model reports true is kept, an id it reports false is
    # rejected, and -- the case that matters -- an id it does not mention at all is also rejected.
    #
    # Omission is a real answer here, not missing data. Asked to classify a list, gemma4 reports
    # only the members it considers peptides and leaves the rest out entirely (for three
    # candidates it returned a single {"id":2,"is_peptide":true}). Treating an unmentioned id as
    # "no verdict, keep it" therefore accepted every non-peptide and made triage a no-op.
    #
    # An EMPTY list is likewise a real answer -- "none of these is a peptide" -- and is how this
    # model says so, given that it reports only the affirmative members. Treating empty as a
    # non-answer meant a batch consisting entirely of junk was kept in full, which is the batch
    # where triage matters most: on a 43-junk ballot split into batches, 25 junk candidates
    # survived purely because their batches contained nothing to affirm.
    #
    # Fail-open is preserved where it belongs and only there: a response that does not parse, or
    # is not a list at all, keeps everything (handled above). A well-formed ballot is trusted.
    # The exposure is bounded -- only name-only candidates reach triage, catalogued peptides are
    # exempted by the caller, and sequence-bearing candidates never enter the pool.
    affirmed = set()
    for v in verdicts:
        if not isinstance(v, dict) or v.get("is_peptide") is not True:
            continue
        try:
            idx = int(v.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(subset):
            affirmed.add(subset[idx][0])

    if not affirmed:
        logger.debug(f"[{doc_id}] Candidate triage rejected all {len(subset)} candidate(s).")

    # Every candidate in this call was on the ballot (batching above guarantees it), so the ballot
    # is complete and the affirmed set is the answer -- including when it is empty.
    return affirmed & accepted



_ORGANISM_NORMALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "organisms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "scientific_name": {"type": "string"},
                },
                "required": ["id", "scientific_name"],
            },
        }
    },
    "required": ["organisms"],
}

_ORGNORM_MAX_NAMES = 30


def normalize_organisms_llm(names: list, doc_id: str = "", model: str = None) -> dict:
    """
    Maps raw organism strings that NCBI did not recognize to candidate binomial scientific names,
    e.g. "staph aureus" -> "Staphylococcus aureus", "the yeast" -> "Saccharomyces cerevisiae".

    Only a *suggestion*: every value returned here is re-checked against NCBI by the caller, and
    anything that still fails to resolve is discarded. That ordering matters -- the model is used
    to guess a spelling, never to authorize a name into the database.

    Batched (one call per document) and bounded, and returns {} on any failure so the caller
    simply falls through to discarding the unresolvable names.
    """
    if not names:
        return {}
    if _ORGNORM_BREAKER.is_open():
        return {}

    from sample.config import DEFAULT_EXTRACT_MODEL
    target_model = model or DEFAULT_EXTRACT_MODEL

    subset = list(names)[:_ORGNORM_MAX_NAMES]
    listing = "\n".join(f'{i}. "{n}"' for i, n in enumerate(subset))

    system = (
        "You convert organism names from scientific papers into their formal NCBI binomial "
        "scientific name. Examples: \"staph aureus\" -> \"Staphylococcus aureus\"; \"human\" -> "
        "\"Homo sapiens\"; \"baker's yeast\" -> \"Saccharomyces cerevisiae\"; \"P. aeruginosa\" -> "
        "\"Pseudomonas aeruginosa\". If a string is NOT an organism at all (a cell line, a "
        "chemical, an assay word, a piece of a longer word), return an empty string for it. "
        "Answer only with the JSON schema given."
    )
    user = listing + (
        '\n\nReturn {"organisms":[{"id":<int>,"scientific_name":"<binomial or empty string>"}]}'
    )

    try:
        raw = llm_call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, model=target_model, max_tokens=384, timeout=_LLM_CALL_TIMEOUT,
            response_schema=_ORGANISM_NORMALIZE_SCHEMA,
            layer="organism_norm",
        )
    except Exception as e:
        logger.warning(f"[{doc_id}] LLM organism normalization unavailable ({e}).")
        _ORGNORM_BREAKER.record_failure(e)
        return {}
    _ORGNORM_BREAKER.record_success()

    try:
        parsed = clean_json(raw)
    except Exception:
        parsed = None
    entries = parsed.get("organisms") if isinstance(parsed, dict) else None
    if not isinstance(entries, list):
        return {}

    out = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            idx = int(entry.get("id"))
        except (TypeError, ValueError):
            continue
        suggestion = (entry.get("scientific_name") or "").strip()
        if suggestion and 0 <= idx < len(subset):
            out[subset[idx]] = suggestion
    return out



def verify_activities_llm(activities: list, context_text: str, peptide_id: str = "",
                          doc_id: str = "", model: str = None) -> list:
    """
    Verifies classically-extracted activities against their source text using the local LLM
    (Ollama/qwen, per sample/config.py's DEFAULT_EXTRACT_MODEL), replacing the BERT/NLI
    entailment gate that used to do this.

    Why the change: the NLI gate (since removed) was measured to reject 100% of extracted
    activities (0 of 84 across 25 real documents) while carrying no usable signal -- on hand-built
    pairs, TRUE claims scored 0.015-0.053 entailment and FALSE claims 0.034-0.354, i.e. overlapping
    with the top score belonging to a false claim. An instruction-following LLM judges "does this text actually state that this
    peptide had this value against this target?" directly, which is the question being asked,
    rather than the surface-level sentence-pair entailment NLI was trained for.

    All of a peptide's activities go in ONE call (numbered, verdicts returned by id) rather than
    one call per activity -- per-activity calls were what made the old LLM *fallback* path the
    pipeline's dominant cost.

    Fails OPEN (returns activities unchanged) when the LLM is unavailable or returns something
    unusable. That direction is deliberate: the deterministic grounding upstream (value parsed and
    present in its own sentence, organism validated, clause-based pairing, and the
    known-type-or-organism gate) is what establishes correctness, and this is a second opinion on
    top of it. Failing closed is exactly how the NLI gate silently emptied the database.
    """
    if not activities or not context_text:
        return activities
    if _VERIFY_BREAKER.is_open():
        return activities

    from sample.config import DEFAULT_EXTRACT_MODEL
    from sample.preprocessing import raw_value_and_unit
    target_model = model or DEFAULT_EXTRACT_MODEL

    lines = []
    for i, act in enumerate(activities):
        lines.append(
            f"{i}. peptide={peptide_id or 'the peptide'} | activity={act.get('activity_type') or 'unspecified'} "
            f"| target={act.get('target_organism') or 'unspecified'} "
            # Raw value with the unit it was printed in, since the model checks it against the
            # source text -- see preprocessing.raw_value_and_unit.
            f"| value={' '.join(str(x) for x in raw_value_and_unit(act) if x)}".strip()
        )
    claims = "\n".join(lines)

    system = (
        "You verify whether biological activity claims are actually supported by a scientific text. "
        "A claim is supported ONLY if the text states that this peptide/compound was measured to have "
        "that value against that target. Reject claims where the number is a reagent or buffer "
        "concentration, a molecular weight, a wavelength, a page/table/figure number, a dose of a "
        "different compound, a column heading, or a symbol-key legend rather than a measured result "
        "for this peptide.\n"
        # Claims arriving with activity=unspecified are the ones that survived extraction without a
        # named assay, and they are where the remaining false positives concentrate: the number is
        # typically the concentration a treatment was APPLIED at, in a sentence that also reports
        # an outcome ("...was downregulated after treatment with 2 mg/ml of AS2-1"). Deterministic
        # rules cannot separate those reliably because the outcome language is genuine -- it just
        # belongs to a different number than the one captured.
        'When activity=unspecified, be strict: the text must report that number as the RESULT that '
        'was measured. A concentration the peptide was applied AT -- "treated with", "in the '
        'presence of", "at a dose of", the salt or medium the assay was run in -- is not a result, '
        "even when the same sentence describes an effect.\n"
        "Answer only with the JSON schema given."
    )
    # 2500 chars (~700 tokens), not the whole 4000-token context window. The claims being checked
    # were all extracted from sentences inside this window, and a tighter excerpt both fits the
    # fitted num_ctx (see llm_call) and keeps the judgement focused on the text that actually
    # states the measurement rather than burying it in surrounding methods prose.
    user = (
        f"TEXT:\n{context_text[:2500]}\n\n"
        f"CLAIMS:\n{claims}\n\n"
        "For each claim id, is it supported by the TEXT? Return {\"verdicts\":[{\"id\":<int>,\"supported\":<bool>}]}"
    )

    try:
        raw = llm_call(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0, model=target_model, max_tokens=512, timeout=_LLM_CALL_TIMEOUT,
            response_schema=_verdict_schema("verdicts", "supported", len(activities)),
            layer="activity_verify",
        )
    except Exception as e:
        logger.warning(f"[{doc_id}] LLM verification unavailable ({e}); keeping activities unverified.")
        _VERIFY_BREAKER.record_failure(e)
        return activities
    _VERIFY_BREAKER.record_success()

    try:
        parsed = clean_json(raw)
    except Exception:
        parsed = None
    verdicts = parsed.get("verdicts") if isinstance(parsed, dict) else None
    if not isinstance(verdicts, list) or not verdicts:
        logger.debug(f"[{doc_id}] LLM verification returned no usable verdicts; keeping activities.")
        return activities

    rejected = set()
    for v in verdicts:
        if isinstance(v, dict) and v.get("supported") is False:
            try:
                rejected.add(int(v.get("id")))
            except (TypeError, ValueError):
                continue

    # A verdict list that rejects everything is far more likely a confused/degenerate response
    # than a document in which every classically-grounded activity is wrong -- treat it as a
    # non-answer rather than letting one bad generation wipe the document's data.
    if len(rejected) >= len(activities):
        logger.debug(f"[{doc_id}] LLM verification rejected all {len(activities)} activities; ignoring verdict.")
        return activities

    return [act for i, act in enumerate(activities) if i not in rejected]


def recover_peptide_llm(context_text: str, target_seq: str, doc_id: str = "", model: str = None) -> list:
    """
    Uses a local LLM (via llm_call -- the same Gemini/DeepSeek/Ollama-routing call used
    everywhere else in the pipeline) to recover activities for a specific peptide that survived
    candidate discovery but had no NLI-verified activity.

    This used to hardcode a direct OpenAI-client call to a vLLM server assumed to be running at
    http://localhost:8000/v1 serving 'Qwen/Qwen2.5-Coder-7B-Instruct' -- infrastructure that was
    never actually part of this project's setup (no Makefile target, service, or install step
    starts it), so every call failed with a connection error. The LLM that actually is running
    in this environment is Ollama (already the default backend for the rest of the pipeline, per
    sample/config.py's DEFAULT_EXTRACT_MODEL), so this now goes through the same `llm_call` path
    instead of assuming unmanaged infrastructure that isn't there.

    Backs off for _RECOVERY_COOLDOWN_SECONDS after a failed call instead of retrying a dead LLM
    backend on every single candidate across every worker thread.
    """
    if _RECOVERY_BREAKER.is_open():
        return []

    from sample.config import DEFAULT_EXTRACT_MODEL
    target_model = model or DEFAULT_EXTRACT_MODEL

    prompt = f"""You are extracting biological activity data for exactly ONE named peptide from a
scientific text: "{target_seq}".

STRICT RULES (small models tend to invent plausible-looking data if you don't enforce these):
1. Only report an activity if the text explicitly states a measured value (MIC, IC50, ZOI, etc.)
   for something with this EXACT name: "{target_seq}".
2. Do NOT report a value that belongs to a different peptide, a control/reference substance, an
   assay/methods description, or a citation to another study -- even if "{target_seq}" is
   mentioned nearby. Example: "similar to gramicidin S (Smith 1990)" is a comparison, NOT a
   measured result for gramicidin S in THIS text -- exclude it.
3. For every activity you report, copy the exact supporting sentence from the text verbatim into
   "supporting_quote". Do not paraphrase or summarize it. This sentence must literally contain
   "{target_seq}" and the reported value.
4. If the text does not explicitly report a measured value for "{target_seq}", return
   {{"biological_activities": []}}. An empty result is the more common and often correct answer --
   do not force a match just to produce output.

Return ONLY a valid JSON object matching this schema, with no other text or markdown:
{{
  "biological_activities": [
    {{
      "target_organism": "string (e.g., E. coli, cancer cell line)",
      "target_group": "string (bacteria, fungi, cancer, virus, unknown)",
      "activity_type": "string (MIC, IC50, ZOI, etc.)",
      "value_raw": "string (the exact number, e.g., '10.5' or '>50')",
      "activity_unit": "string (e.g., ug/mL, uM, mm)",
      "supporting_quote": "string (the exact sentence from the text stating this, verbatim)"
    }}
  ]
}}

Text:
{context_text}
"""
    messages = [
        {"role": "system", "content": "You are an expert biological data extractor. Output only raw JSON. You never invent data -- an empty result is correct when the text doesn't support a match."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = llm_call(messages, temperature=0.0, model=target_model, max_tokens=1024, timeout=_LLM_CALL_TIMEOUT,
        layer="peptide_recover",
                        response_schema=_ACTIVITY_RECOVERY_SCHEMA)
        data = clean_json(raw)
        activities = data.get("biological_activities", []) if isinstance(data, dict) else []
        if not isinstance(activities, list):
            return []
        _RECOVERY_BREAKER.record_success()
        return _verify_grounded_activities(activities, context_text, target_seq, doc_id)
    except Exception as e:
        logger.warning(f"[{doc_id}] LLM recovery failed: {e}")
        _RECOVERY_BREAKER.record_failure(e)
        return []


def _verify_grounded_activities(activities: list, context_text: str, target_seq: str, doc_id: str) -> list:
    """
    Programmatic grounding check -- does not trust the model's "supporting_quote" at face value.
    Rejects an activity unless its quote (a) actually mentions the target peptide by name and
    (b) is a genuine (not fabricated) excerpt of the source text, checked via a normalized
    substring match rather than exact equality to tolerate minor whitespace/OCR differences.
    Measured directly against an adversarial case (a peptide mentioned only in a citation to
    another study, with no real activity data): without this check, three different local models
    each fabricated an activity anyway by attaching an unrelated nearby number to it.

    A genuine, verbatim-in-source quote is not sufficient by itself, though: a real pasted
    extraction example ("SV-4814") showed the model can find an authentic sentence naming the
    peptide and still attach a value that isn't actually in that sentence (a page number sitting
    next to the name), plus a self-contradictory target_group ("bacteria" for a human cell line
    it had itself just named). So beyond the quote-groundedness check above, this also verifies
    the claimed value and organism are themselves stated in that same quote, and never trusts the
    model's own target_group label.
    """
    from sample.nlp import parse_activity_value_fast, is_valid_organism_match, infer_activity_group, get_cellosaurus_kp

    def normalize(s: Any) -> str:
        """
        :param s: Text to fold.
        :returns: Lower-cased, whitespace-collapsed text, for substring comparison against a
            supporting quote.
        """
        return re.sub(r'\s+', ' ', str(s).strip().lower())

    def _numbers_in(s: Any) -> list:
        """
        :param s: Text to scan.
        :returns: Every number in it, with thousands separators removed.
        """
        return [float(m) for m in re.findall(r'\d+(?:\.\d+)?', s.replace(',', ''))]

    context_norm = normalize(context_text)
    target_norm = target_seq.strip().lower()
    verified = []
    for act in activities:
        if not isinstance(act, dict):
            continue
        quote = act.pop("supporting_quote", "") if isinstance(act, dict) else ""
        quote_norm = normalize(quote)
        if not quote_norm or target_norm not in quote_norm:
            logger.debug(f"[{doc_id}] Rejected ungrounded activity for '{target_seq}': quote doesn't name the peptide ({quote!r})")
            continue
        # A genuine verbatim excerpt should appear as a contiguous substring of the source text;
        # check a meaningful prefix rather than the whole quote to tolerate minor truncation.
        probe = quote_norm[:60] if len(quote_norm) >= 60 else quote_norm
        if probe not in context_norm:
            logger.debug(f"[{doc_id}] Rejected ungrounded activity for '{target_seq}': quote not found verbatim in source text ({quote!r})")
            continue

        # The claimed value must actually be the one stated in the quote -- checked as numeric
        # equivalence (parsed via the same operator/range/±-aware parser used elsewhere), not a
        # raw digit-substring match, which is brittle to formatting ("1,500" vs "1500") and can
        # still pass a fabricated value that happens to share formatting with a real one nearby.
        # A value that isn't parseable as a number at all (e.g. "potent") is rejected outright --
        # a real measured activity is always numeric; this was measured directly during a model
        # evaluation where a larger model fabricated value_raw="potent" for a sentence that was
        # actually a citation to another study, exactly the case the prompt tells it to exclude.
        _, claimed_val, _ = parse_activity_value_fast(act.get("value_raw", ""))
        if claimed_val is None:
            logger.debug(f"[{doc_id}] Rejected ungrounded activity for '{target_seq}': value_raw {act.get('value_raw')!r} isn't a parseable number")
            continue
        quote_numbers = _numbers_in(quote_norm)
        if not any(abs(claimed_val - n) <= max(abs(claimed_val), 1e-9) * 1e-6 for n in quote_numbers):
            logger.debug(f"[{doc_id}] Rejected ungrounded activity for '{target_seq}': claimed value {act.get('value_raw')!r} not found in quote ({quote!r})")
            continue

        # The claimed organism must also be named in the quote -- the same standard already
        # applied to the peptide's own name above.
        organism = normalize(act.get("target_organism", ""))
        if not organism or organism not in quote_norm:
            logger.debug(f"[{doc_id}] Rejected ungrounded activity for '{target_seq}': organism {act.get('target_organism')!r} not found in quote ({quote!r})")
            continue

        # Never trust the model's own target_group label -- recompute it deterministically from
        # the now quote-verified organism, via the same pipeline the classical extraction path
        # uses, so both paths share one source of truth for organism/cell-line classification.
        is_cell_line = is_valid_organism_match(act.get("target_organism", ""), quote) and bool(
            get_cellosaurus_kp().extract_keywords(act.get("target_organism", ""))
        )
        act["target_group"] = infer_activity_group(act.get("target_organism", ""), act.get("activity_type", ""), is_cell_line=is_cell_line)

        verified.append(act)
    return verified

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text from a PDF file using PyMuPDF (fitz).
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        logger.debug(f"Failed to extract text from {pdf_path}: {e}")
        
    # Strip soft hyphens across newlines
    text = text.replace('\xad\n', '')
    text = text.replace('\xad', '')
    text = text.replace('-\n', '')
    
    # Issue 1: Join line breaks between valid 3-letter or 1-letter amino acid sequences
    # For 3-letter codes split by newline (e.g. GLY-\nALA -> GLY-ALA, GLY\nALA -> GLYALA)
    import re
    text = re.sub(r'([A-Z]{3}-?)\s*\n\s*([A-Z]{3})', r'\1\2', text)
    # For 1-letter codes split by newline (e.g. ACDEF\nGHIK -> ACDEFGHIK)
    text = re.sub(r'([A-Z]{4,}-?)\s*\n\s*([A-Z]{4,})', r'\1\2', text)
    
    return text
