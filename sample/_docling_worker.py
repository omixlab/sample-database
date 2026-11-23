"""
Standalone docling table worker -- runs INSIDE the ``sample-docling`` environment.

This file is executed as a subprocess by :mod:`sample.docling_tables`, never imported by the rest
of the project. It must therefore stay free of any ``sample.*`` import: ``sample-docling`` resolves
numpy 2.x and transformers 5.17, while ``sample-extraction`` is pinned to numpy<2 for spacy, so the
two environments cannot share a process. See ``environments/sample-docling.yml``.

Input  : a PDF path plus flags, on argv.
Output : one JSON object on stdout.

The output carries the raw cell grid rather than a flattened DataFrame. Flattening loses exactly
what the caller needs: which cells are header cells, which row of a two-row header they came from,
and how far they span. A table whose peptides are its COLUMNS (assay tables routinely put the
organisms down the side) is indistinguishable from a normal one once the header is a single joined
string, and the caller would read organism names as peptide names.

Page text is deliberately NOT returned. The caller builds context windows with PyMuPDF, which it
already has; shipping every page's text through JSON would dominate the payload for no gain.
"""
import argparse
import json
import logging
import sys
import warnings


def _convert(pdf_path, use_ocr, tesseract_cmd, ocr_lang, args):
    """
    Convert one PDF and return the plain-data structure that goes to stdout.

    :param pdf_path: The PDF to convert.
    :param use_ocr: Whether to run OCR. When on, docling renders each page to an image at
        ``scale`` and OCRs the whole page with tesseract, ignoring any text layer -- the
        PDF -> PNG -> tesseract path. That is the only way to read a scanned document: measured
        over 60 sampled patents (1,400 pages), 36.7% have a text layer of exactly ZERO characters,
        so pdfplumber and docling's text path both return nothing for them. An earlier note here
        claimed 99.7% of the corpus has a healthy text layer; that held for papers only and was
        wrong for patents, which is why the patent corpus yielded almost nothing.
    :param tesseract_cmd: Absolute path to the ``tesseract`` binary. Required when ``use_ocr``.
    :param ocr_lang: Tesseract language codes.
    :returns: ``{"tables": [...], "text": "..."}`` -- each table carrying its page number and cell
        grid, plus the whole document as Markdown. The text is what the LangExtract backend reads,
        and on a scanned document it is OCR output that exists nowhere else.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (PdfPipelineOptions, TableFormerMode,
                                                    TesseractCliOcrOptions)

    opts = PdfPipelineOptions()
    opts.do_table_structure = True

    # Run TableFormer on the GPU when there is one, with CPU threads alongside it.
    #
    # Worth doing because the table pass is 82% of the screening step's time (measured: 429 of
    # 521s over 24 documents) and the screening step runs with the LangExtract backend disabled --
    # so the RTX 3060 is completely idle while TableFormer grinds through pages on CPU. The
    # refined step is where the GPU is contended; screening is not.
    #
    # Falls back to CPU silently when torch has no CUDA build or no device is visible, because a
    # host without a GPU must still run this step rather than fail in it.
    try:
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        import torch

        device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
        opts.accelerator_options = AcceleratorOptions(device=device,
                                                      num_threads=max(1, args.num_threads))
    except Exception as e:  # noqa: BLE001 -- accelerator selection must never sink a conversion
        print(f"accelerator selection unavailable ({type(e).__name__}: {e}); using defaults",
              file=sys.stderr)
    # ACCURATE over FAST: the structure model is the whole reason this worker exists, and the
    # measured cost of the accurate mode on this corpus is ~1.14 s/page, which the caller's gating
    # already budgets for.
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True

    if use_ocr:
        # Tesseract, never docling's default RapidOCR. RapidOCR ignores both the docling cache and
        # the HuggingFace cache and downloads PP-OCRv6 weights from modelscope.cn at first use --
        # verified by running with the network blocked, where it raised DownloadFileException. This
        # project must run fully locally, so the OCR engine has to be the local tesseract binary.
        opts.do_ocr = True
        opts.ocr_options = TesseractCliOcrOptions(tesseract_cmd=tesseract_cmd, lang=list(ocr_lang),
                                                  force_full_page_ocr=True)
    else:
        opts.do_ocr = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    document = converter.convert(pdf_path).document

    tables = []
    for table in document.tables:
        prov = getattr(table, "prov", None) or []
        # page_no is load-bearing downstream: the caller builds the context window from that page's
        # text, and a record whose context does not contain its own sequence is DELETED by the
        # extraction pipeline's third gate. A table with no provenance is reported as page None so
        # the caller can skip it rather than silently attach it to the wrong page.
        page_no = getattr(prov[0], "page_no", None) if prov else None

        data = getattr(table, "data", None)
        raw_cells = list(getattr(data, "table_cells", None) or [])
        cells = []
        for cell in raw_cells:
            text = (getattr(cell, "text", "") or "").strip()
            if not text:
                continue
            cells.append({
                "r0": getattr(cell, "start_row_offset_idx", None),
                "r1": getattr(cell, "end_row_offset_idx", None),
                "c0": getattr(cell, "start_col_offset_idx", None),
                "c1": getattr(cell, "end_col_offset_idx", None),
                "header": bool(getattr(cell, "column_header", False)),
                "text": text,
            })
        if not cells:
            continue
        tables.append({
            "page_no": page_no,
            "num_rows": getattr(data, "num_rows", None),
            "num_cols": getattr(data, "num_cols", None),
            "cells": cells,
        })

    # The document TEXT, not just the table grids. This is the payload's reason for existing on a
    # scanned PDF: the conversion has already paid for rendering and OCR-ing every page, and
    # throwing the text away is what left 36.7% of patents with no readable content for any
    # downstream extractor. Markdown rather than plain text because it keeps table rows on one
    # line with their cells delimited, so a measurement and the peptide it belongs to stay
    # adjacent in the character stream LangExtract reads and grounds its spans against.
    try:
        text = document.export_to_markdown()
    except Exception:  # noqa: BLE001 -- tables are still worth returning if the export fails
        try:
            text = document.export_to_text()
        except Exception:  # noqa: BLE001
            text = ""
    return {"tables": tables, "text": text or ""}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract table cell grids from a PDF via docling.")
    parser.add_argument("pdf_path")
    parser.add_argument("--ocr", action="store_true",
                        help="Run Tesseract OCR (for PDFs with no usable text layer).")
    parser.add_argument("--tesseract-cmd", default=None,
                        help="Absolute path to the tesseract binary; required with --ocr.")
    parser.add_argument("--ocr-lang", default="eng",
                        help="Comma-separated Tesseract language codes.")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="CPU threads this conversion may use, sized by the caller from the "
                             "conversion semaphore. Passed to docling's AcceleratorOptions so it "
                             "applies to the models, not just to BLAS.")
    parser.add_argument("--out", default=None,
                        help="Write the JSON payload to this file instead of stdout. Preferred: "
                             "stdout cannot be kept clean (see below).")
    args = parser.parse_args(argv)

    if args.ocr and not args.tesseract_cmd:
        json.dump({"error": "--ocr requires --tesseract-cmd (the binary is not on PATH here)"},
                  sys.stdout)
        return 2

    # docling and its dependencies write to stdout at import and conversion time, and silencing
    # them is NOT sufficient. TableFormer's MatchingPostProcessor prints its orphan-cell recovery
    # warnings through its own writer, and the weight loader prints a tqdm progress bar; neither
    # goes through `logging` or `warnings`, so both survive the calls below and land in front of the
    # JSON. json.loads then fails with "Extra data: line 1 column 5" and the caller sees zero tables
    # -- silently, on exactly the ragged tables this worker exists to recover. One paper whose six
    # tables were real returned nothing that way.
    #
    # So the payload gets its own channel (--out) rather than competing with whatever a third-party
    # library decides to print. The noise suppression below stays as a courtesy for stdout mode, but
    # correctness no longer depends on it being complete.
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)

    try:
        payload = _convert(args.pdf_path, args.ocr, args.tesseract_cmd,
                           [s for s in args.ocr_lang.split(",") if s], args)
    except Exception as exc:                                    # noqa: BLE001 - reported, not hidden
        # The caller cannot see this process's traceback, so the reason travels in the payload.
        # Returning an error document beats exiting non-zero with a bare stderr dump: the caller
        # logs one line naming the document and the cause, and carries on with the other backend.
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        _emit(payload, args.out)
        return 1

    _emit(payload, args.out)
    return 0


def _emit(payload, out_path):
    """
    Write the JSON payload where the caller can read it unambiguously.

    :param payload: The document to serialise.
    :param out_path: A file to write, or None for stdout.
    """
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
