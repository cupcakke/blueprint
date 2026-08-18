#!/usr/bin/env python3
"""Generate the Futhark FFI surface from the compiler-emitted manifest.

The Futhark compiler writes a `<prog>.json` manifest next to the generated C
library.  That manifest is the authoritative, machine-readable description of
the C ABI actually produced by the exact compiler version that ran.  Anything
hand-written against a remembered ABI is a guess; the manifest is ground truth.

Futhark changed its C ABI in 0.26.1: entry points returning tuples stopped
implicitly unpacking them into multiple out-parameters and instead return a
single opaque `futhark_opaque_tupN_*` value that must be projected.  Bindings
that assume the wrong convention still *link* (C has no cross-TU signature
checking for these), then corrupt the stack at run time.  Deriving both the
Zig externs and the C static assertions from the manifest makes that class of
mismatch impossible to express.

Emits:
  * futhark_abi.zig  -- extern declarations + opaque handle types
  * futhark_abi_check.c -- _Static_assert per entry point against the real header
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

TUPLE_UNPACK_BOUNDARY = (0, 26, 1)

SCALAR_C = {
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "f16": "uint16_t",
    "f32": "float",
    "f64": "double",
    "bool": "bool",
}

SCALAR_ZIG = {
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "u8": "u8",
    "u16": "u16",
    "u32": "u32",
    "u64": "u64",
    "f16": "u16",
    "f32": "f32",
    "f64": "f64",
    "bool": "bool",
}


class ManifestError(RuntimeError):
    pass


def parse_version(raw: str) -> Tuple[int, ...]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", raw or "")
    if not match:
        raise ManifestError(f"cannot parse Futhark version from {raw!r}")
    return tuple(int(g) for g in match.groups())


def unpacks_tuples(version: Tuple[int, ...]) -> bool:
    """True when this compiler flattens tuple returns into out-params."""
    return version < TUPLE_UNPACK_BOUNDARY


def scalar_of(t: str) -> str | None:
    return t if t in SCALAR_C else None


def array_struct(manifest: Dict[str, Any], t: str) -> str:
    info = manifest["types"].get(t)
    if info is None or info.get("kind") != "array":
        raise ManifestError(f"type {t!r} is not a manifest array type")
    ctype = info["ctype"].replace("*", "").strip()
    if not ctype.startswith("struct "):
        raise ManifestError(f"unexpected ctype {info['ctype']!r} for {t!r}")
    return ctype[len("struct ") :]


def tuple_struct_name(outputs: List[Dict[str, Any]]) -> str:
    """Reconstruct Futhark's generated name for a returned tuple type.

    Futhark names these `tupN_` followed by each component rendered as
    `arrRd_<elem>` for rank-R arrays or the bare scalar name.
    """
    parts = []
    for out in outputs:
        t = out["type"]
        if scalar_of(t):
            parts.append(t)
        else:
            rank = t.count("[]")
            elem = t.replace("[]", "")
            parts.append(f"arr{rank}d_{elem}")
    return f"tup{len(outputs)}_" + "_".join(parts)


def c_param(manifest: Dict[str, Any], t: str, *, out: bool, const: bool) -> str:
    s = scalar_of(t)
    if s:
        return f"{SCALAR_C[s]} *" if out else f"const {SCALAR_C[s]}"
    struct = array_struct(manifest, t)
    if out:
        return f"struct {struct} **"
    return f"const struct {struct} *"


def zig_param(manifest: Dict[str, Any], t: str, *, out: bool) -> str:
    s = scalar_of(t)
    if s:
        return f"?*{SCALAR_ZIG[s]}" if out else SCALAR_ZIG[s]
    struct = array_struct(manifest, t)
    if out:
        return f"?*?*struct_{struct}"
    return f"?*const struct_{struct}"


def entry_signature(
    manifest: Dict[str, Any], entry: Dict[str, Any], flatten: bool
) -> Tuple[List[str], List[str]]:
    """Return (c_param_types, zig_param_decls) excluding the context arg."""
    outputs = entry["outputs"]
    inputs = entry["inputs"]

    c_params: List[str] = []
    zig_params: List[str] = []

    if len(outputs) > 1 and not flatten:
        name = tuple_struct_name(outputs)
        c_params.append(f"struct futhark_opaque_{name} **")
        zig_params.append(f"    out: ?*?*struct_futhark_opaque_{name},")
    else:
        for i, out in enumerate(outputs):
            c_params.append(c_param(manifest, out["type"], out=True, const=False))
            zig_params.append(f"    out{i}: {zig_param(manifest, out['type'], out=True)},")

    for i, inp in enumerate(inputs):
        c_params.append(c_param(manifest, inp["type"], out=False, const=True))
        raw = inp.get("name") or f"in{i}"
        nm = re.sub(r"[^A-Za-z0-9_]", "_", raw)
        if nm in {"type", "error", "test", "fn", "const", "var", "align", "export"}:
            nm = nm + "_"
        zig_params.append(f"    {nm}: {zig_param(manifest, inp['type'], out=False)},")

    return c_params, zig_params


def collect_structs(manifest: Dict[str, Any], flatten: bool) -> List[str]:
    structs = {"futhark_context_config", "futhark_context"}
    for info in manifest["types"].values():
        if info.get("kind") == "array":
            structs.add(info["ctype"].replace("*", "").strip()[len("struct ") :])
    if not flatten:
        for entry in manifest["entry_points"].values():
            if len(entry["outputs"]) > 1:
                structs.add("futhark_opaque_" + tuple_struct_name(entry["outputs"]))
    return sorted(structs)


def gen_zig(manifest: Dict[str, Any], flatten: bool, source: str) -> str:
    L: List[str] = []
    version = manifest["version"].strip().splitlines()[0].strip()
    L.append("// @generated by tools/futhark_abi_gen.py -- DO NOT EDIT BY HAND.")
    L.append(f"// Source manifest: {source}")
    L.append(f"// Futhark version: {version}   backend: {manifest['backend']}")
    L.append(
        "// Tuple returns are "
        + ("flattened into out-parameters." if flatten else "returned as opaque values.")
    )
    L.append("")
    L.append("pub const futhark_version: []const u8 = \"" + version + "\";")
    L.append("pub const futhark_backend: []const u8 = \"" + manifest["backend"] + "\";")
    L.append(
        "pub const tuple_outputs_flattened: bool = " + ("true" if flatten else "false") + ";"
    )
    L.append("")

    for struct in collect_structs(manifest, flatten):
        L.append(f"pub const struct_{struct} = opaque {{}};")
    L.append("")

    for name in sorted(manifest["entry_points"]):
        entry = manifest["entry_points"][name]
        _, zig_params = entry_signature(manifest, entry, flatten)
        L.append(f"pub extern \"c\" fn {entry['cfun']}(")
        L.append("    ctx: ?*struct_futhark_context,")
        L.extend(zig_params)
        L.append(") c_int;")
        L.append("")

    # Array constructors / destructors / readers, straight from the manifest ops.
    for tname in sorted(manifest["types"]):
        info = manifest["types"][tname]
        if info.get("kind") != "array":
            continue
        struct = array_struct(manifest, tname)
        elem = SCALAR_ZIG[info["elemtype"]]
        rank = int(info["rank"])
        ops = info["ops"]
        dims = ", ".join(f"dim{i}: i64" for i in range(rank))
        L.append(
            f"pub extern \"c\" fn {ops['new']}(ctx: ?*struct_futhark_context, "
            f"data: ?[*]const {elem}, {dims}) ?*struct_{struct};"
        )
        L.append(
            f"pub extern \"c\" fn {ops['free']}(ctx: ?*struct_futhark_context, "
            f"arr: ?*struct_{struct}) c_int;"
        )
        L.append(
            f"pub extern \"c\" fn {ops['values']}(ctx: ?*struct_futhark_context, "
            f"arr: ?*struct_{struct}, data: ?[*]{elem}) c_int;"
        )
        L.append(
            f"pub extern \"c\" fn {ops['shape']}(ctx: ?*struct_futhark_context, "
            f"arr: ?*struct_{struct}) ?[*]const i64;"
        )
        if "values_raw" in ops:
            L.append(
                f"pub extern \"c\" fn {ops['values_raw']}(ctx: ?*struct_futhark_context, "
                f"arr: ?*struct_{struct}) ?*anyopaque;"
            )
        L.append("")

    if not flatten:
        for name in sorted(manifest["entry_points"]):
            entry = manifest["entry_points"][name]
            outs = entry["outputs"]
            if len(outs) <= 1:
                continue
            tname = tuple_struct_name(outs)
            L.append(
                f"pub extern \"c\" fn futhark_free_opaque_{tname}"
                f"(ctx: ?*struct_futhark_context, obj: ?*struct_futhark_opaque_{tname}) c_int;"
            )
            for i, out in enumerate(outs):
                L.append(
                    f"pub extern \"c\" fn futhark_project_opaque_{tname}_{i}("
                    f"ctx: ?*struct_futhark_context, "
                    f"out: {zig_param(manifest, out['type'], out=True)}, "
                    f"obj: ?*const struct_futhark_opaque_{tname}) c_int;"
                )
            L.append("")

    return "\n".join(L) + "\n"


def gen_c_check(manifest: Dict[str, Any], flatten: bool, header: str, source: str) -> str:
    L: List[str] = []
    version = manifest["version"].strip().splitlines()[0].strip()
    L.append("/* @generated by tools/futhark_abi_gen.py -- DO NOT EDIT BY HAND. */")
    L.append(f"/* Source manifest: {source} (Futhark {version}, {manifest['backend']}) */")
    L.append(f'#include "{header}"')
    L.append("#include <stdbool.h>")
    L.append("#include <stdint.h>")
    L.append("")
    L.append("/* Each assertion compares the real generated prototype against the")
    L.append("   signature implied by the manifest. A mismatch is a compile error, so")
    L.append("   stale generated C can never be linked against these bindings. */")
    L.append("")

    for name in sorted(manifest["entry_points"]):
        entry = manifest["entry_points"][name]
        c_params, _ = entry_signature(manifest, entry, flatten)
        params = ", ".join(["struct futhark_context *"] + c_params)
        L.append(f"typedef int (*abi_{name}_t)({params});")
    L.append("")

    for name in sorted(manifest["entry_points"]):
        entry = manifest["entry_points"][name]
        L.append(
            "_Static_assert(__builtin_types_compatible_p("
            f"__typeof__(&{entry['cfun']}), abi_{name}_t),"
        )
        L.append(f'               "{name}: generated C ABI does not match manifest");')
    L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--zig-out", type=Path)
    ap.add_argument("--c-out", type=Path)
    ap.add_argument("--header", default=None)
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify existing outputs are up to date; nonzero exit if not",
    )
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    for key in ("version", "backend", "entry_points", "types"):
        if key not in manifest:
            raise ManifestError(f"manifest missing {key!r}")

    version = parse_version(manifest["version"])
    flatten = unpacks_tuples(version)
    header = args.header or (args.manifest.stem + ".h")
    src = args.manifest.name

    outputs = []
    if args.zig_out:
        outputs.append((args.zig_out, gen_zig(manifest, flatten, src)))
    if args.c_out:
        outputs.append((args.c_out, gen_c_check(manifest, flatten, header, src)))

    stale = False
    for path, text in outputs:
        if args.check:
            current = path.read_text() if path.exists() else None
            if current != text:
                sys.stderr.write(f"stale: {path}\n")
                stale = True
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path}")

    if args.check and stale:
        sys.stderr.write(
            "Generated Futhark bindings are out of date. Re-run tools/futhark_abi_gen.py.\n"
        )
        return 1
    if args.check:
        print("Futhark bindings are up to date.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ManifestError as exc:
        sys.stderr.write(f"error: {exc}\n")
        sys.exit(2)
