"""Enable xtrack's JIT C-kernel compilation on this Windows + MSVC toolchain.

xtrack compiles its tracking kernels on first use via ``cffi`` → the platform C
compiler. On Windows that path is broken in two layers (full diagnosis in
``docs/CONVENTIONS.md``):

1. **xobjects drops compiler flags on Windows.** In
   ``xobjects/context_cpu.py::compile_kernel`` the ``os.name == "nt"`` branch sets
   ``xtr_compile_args = []`` (literal comment ``# TODO: to be handled properly``),
   discarding both the ``-I<site-packages>`` include flag (→ ``C1083``: cannot find
   ``xtrack/multisetter/multisetter.h``) and the ``-DXO_CONTEXT_CPU*`` context
   defines (→ ``C1189``).
2. **xtrack's own C source is not MSVC-clean.** Even with (1) restored, MSVC
   ``cl.exe`` rejects xtrack source with ``C2166: l-value specifies const object``
   (e.g. ``track_misalignments.h``). GCC/Clang accept it — xsuite is developed on
   Linux, so MSVC's stricter front-end is the outlier.

This module monkeypatches the distutils MSVC compiler (the one ``cffi`` uses) to:

- compile with **clang-cl** instead of ``cl.exe`` — clang-cl is a drop-in,
  cl-compatible front-end that reproduces the *reference* toolchain's GCC/Clang
  behaviour (clearing the ``C2166``), while still emitting MSVC-ABI objects that
  the MSVC linker links;
- put ``site-packages`` back on the include path (layer 1a);
- restore the ``XO_CONTEXT_CPU`` / ``XO_CONTEXT_CPU_SERIAL`` defines (layer 1b);
- drop ``/GL`` + ``/LTCG`` — clang-cl's ``/GL`` emits LLVM bitcode that the MSVC
  linker's link-time code generation cannot consume.

It is a **no-op** on non-Windows platforms and whenever clang-cl cannot be found,
so reference cross-checks skip gracefully (rather than erroring) on machines
without LLVM. Install clang-cl with ``winget install LLVM.LLVM`` (or the VS "C++
Clang tools" component); set ``ACCSIM_CLANG_CL`` to point at a specific binary.
"""

from __future__ import annotations

import os
import shutil
import sysconfig

# Common install locations for clang-cl when it is not already on PATH.
_CLANG_CL_CANDIDATES = (
    r"C:\Program Files\LLVM\bin\clang-cl.exe",
    r"C:\Program Files (x86)\LLVM\bin\clang-cl.exe",
)

_applied = False


def _find_clang_cl() -> str | None:
    """Locate a clang-cl executable, or return None if unavailable."""
    override = os.environ.get("ACCSIM_CLANG_CL")
    if override:
        return override if os.path.isfile(override) else None
    on_path = shutil.which("clang-cl")
    if on_path:
        return on_path
    return next((c for c in _CLANG_CL_CANDIDATES if os.path.isfile(c)), None)


def apply() -> str | None:
    """Patch the MSVC compiler to build xtrack kernels with clang-cl.

    Returns the clang-cl path if the patch was applied, otherwise ``None`` (a
    no-op: non-Windows, or clang-cl not found). Idempotent.
    """
    global _applied
    if os.name != "nt":
        return None
    clang_cl = _find_clang_cl()
    if clang_cl is None:
        return None
    if _applied:
        return clang_cl

    # Resolve the concrete MSVC compiler class cffi will instantiate. Done inside
    # the Windows guard because new_compiler(compiler="msvc") is Windows-specific.
    from distutils.ccompiler import new_compiler

    site_packages = sysconfig.get_paths()["purelib"]
    msvc_cls = type(new_compiler(compiler="msvc"))
    orig_initialize = msvc_cls.initialize

    def patched_initialize(self, plat_name=None):  # type: ignore[no-untyped-def]
        orig_initialize(self, plat_name)
        self.cc = clang_cl  # layer 2: GCC-compatible front-end clears C2166
        if site_packages not in self.include_dirs:  # layer 1a: find multisetter.h
            self.include_dirs.append(site_packages)
        for define in ("/DXO_CONTEXT_CPU", "/DXO_CONTEXT_CPU_SERIAL"):  # layer 1b
            if define not in self.compile_options:
                self.compile_options.append(define)
        # clang-cl /GL bitcode is incompatible with the MSVC linker's /LTCG.
        self.compile_options = [opt for opt in self.compile_options if opt != "/GL"]
        for key, flags in list(self._ldflags.items()):
            self._ldflags[key] = [flag for flag in flags if flag != "/LTCG"]

    msvc_cls.initialize = patched_initialize
    _applied = True
    return clang_cl
