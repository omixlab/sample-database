# sAMPle: A web server and pipeline of antimicrobial peptides

# Bind sqlite3 before anything else in this package can import a heavy native library.
#
# The conda environments link a recent ICU (libicui18n.so.78), which needs CXXABI_1.3.15. The
# environment ships a libstdc++ that provides it (CXXABI_1.3.17), but several pip-installed native
# extensions in the extraction stack pull in the SYSTEM /lib/x86_64-linux-gnu/libstdc++.so.6,
# which stops at CXXABI_1.3.14. Whichever loads first wins for the whole process, so importing
# `sample.extractor` before `sqlite3` left the loader unable to resolve `_sqlite3`:
#
#     ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
#                  (required by .../libicui18n.so.78)
#
# and every taxonomy lookup, prediction store and release-database read failed. Importing sqlite3
# here makes the order deterministic no matter which submodule a caller reaches for first. It is
# stdlib and already a dependency of most of this package, so it costs nothing.
import sqlite3 as _sqlite3_preload  # noqa: F401

__version__ = "1.0.0"
