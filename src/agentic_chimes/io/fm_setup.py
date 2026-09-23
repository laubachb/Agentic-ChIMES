"""Parser/writer for ChIMES `fm_setup.in` files.

Grammar characterized directly from `codes/chimes_lsq-LLfork/src/chimes_lsq.C`
and the golden fixtures `test_suite-lsq/{test_4atoms.2,special3b}/fm_setup.in`:
two sections (CONTROL VARIABLES, TOPOLOGY VARIABLES), `# KEYWORD #` directive
lines followed by a value on the next non-blank line, an `NATMTYP`-sized
atom-type table, a `PAIRIDX` table (read until the next directive/block),
optional `EXCLUDE {3,4}B INTERACTION: <n>` and `SPECIAL {3,4}B
{S_MINIM,S_MAXIM}: {ALL <v>|SPECIFIC <n>}` blocks, terminated by `# ENDFILE #`.

`parse`/`render` round-trip *semantically* (re-parsing a rendering reproduces
the same structured dict) rather than byte-for-byte, since real fm_setup.in
files in the wild vary in whitespace/alignment.
"""

from __future__ import annotations

import re
from typing import Optional

_KEYWORD_RE = re.compile(r"^#\s*([A-Za-z0-9_]+)\s*#")

_SINGLE_VALUE_KEYS = {
    "TRJFILE": ("trjfile", str),
    "WRAPTRJ": ("wraptrj", "bool"),
    "NFRAMES": ("nframes", int),
    "NLAYERS": ("nlayers", int),
    "FITCOUL": ("fitcoul", "bool"),
    "FITSTRS": ("fitstrs", str),
    "FITENER": ("fitener", str),
    "FITPOVR": ("fitpovr", "bool"),
    "CHBTYPE": ("chbtype", str),
    "FCUTTYP": ("fcuttyp", str),
    "SPLITFI": ("splitfi", "bool"),
    "USENEIG": ("useneig", "bool"),
    "SKPFRMS": ("skpfrms", str),
}


def _to_bool(s: str) -> bool:
    return s.strip().lower() == "true"


def _bool_str(b: bool) -> str:
    return "true" if b else "false"


def _strip_inline_comment(line: str) -> str:
    idx = line.find("!")
    return (line[:idx] if idx != -1 else line).strip()


def _tokenize(text: str) -> list:
    return [s.strip() for s in text.splitlines() if s.strip()]


def parse(text: str) -> dict:
    lines = _tokenize(text)
    n = len(lines)
    i = 0
    data: dict = {
        "atom_types": [],
        "pairs": [],
        "special_blocks": [],
    }
    natmtyp: Optional[int] = None

    while i < n:
        line = lines[i]
        m = _KEYWORD_RE.match(line)

        if m:
            kw = m.group(1).upper()

            if kw == "ENDFILE":
                break

            if kw == "PAIRTYP":
                i += 1
                toks = _strip_inline_comment(lines[i]).split()
                i += 1
                data["pairtyp"] = toks[0]
                nums = toks[1:]
                data["order2"] = int(nums[0])
                data["order3"] = int(nums[1])
                if len(nums) >= 3:
                    data["order4"] = int(nums[2])
                if len(nums) >= 5:
                    data["cheby_min"] = float(nums[3])
                    data["cheby_max"] = float(nums[4])
                continue

            if kw == "NATMTYP":
                i += 1
                natmtyp = int(_strip_inline_comment(lines[i]))
                data["natmtyp"] = natmtyp
                i += 1
                continue

            if kw == "TYPEIDX":
                i += 1
                if natmtyp is None:
                    raise ValueError("TYPEIDX table encountered before NATMTYP was set")
                for _ in range(natmtyp):
                    toks = lines[i].split()
                    i += 1
                    data["atom_types"].append(
                        {"idx": int(toks[0]), "symbol": toks[1], "charge": float(toks[2]), "mass": float(toks[3])}
                    )
                continue

            if kw == "PAIRIDX":
                i += 1
                while i < n and not _KEYWORD_RE.match(lines[i]) and not lines[i].upper().startswith(("EXCLUDE", "SPECIAL")):
                    toks = lines[i].split()
                    i += 1
                    entry = {
                        "idx": int(toks[0]),
                        "type1": toks[1],
                        "type2": toks[2],
                        "s_minim": float(toks[3]),
                        "s_maxim": float(toks[4]),
                        "s_delta": float(toks[5]),
                        "morse_lambda": float(toks[6]),
                    }
                    if len(toks) > 7:
                        entry["useovrp"] = _to_bool(toks[7])
                    if len(toks) > 10:
                        entry["nijbins"] = int(toks[8])
                        entry["nikbins"] = int(toks[9])
                        entry["njkbins"] = int(toks[10])
                    data["pairs"].append(entry)
                continue

            if kw in _SINGLE_VALUE_KEYS:
                dest, kind = _SINGLE_VALUE_KEYS[kw]
                i += 1
                raw = _strip_inline_comment(lines[i])
                i += 1
                data[dest] = _to_bool(raw) if kind == "bool" else kind(raw)
                continue

            # unknown directive: skip its value line defensively
            i += 1
            if i < n and not _KEYWORD_RE.match(lines[i]):
                i += 1
            continue

        elif line.upper().startswith("EXCLUDE"):
            parts = line.replace(":", " ").split()
            order = 3 if parts[1].upper().startswith("3") else 4
            count = int(parts[-1])
            i += 1
            rows = [lines[i + k].split() for k in range(count)]
            i += count
            data[f"exclude_{order}b"] = rows
            continue

        elif line.upper().startswith("SPECIAL"):
            header = line.replace(":", " ").split()
            order = 3 if header[1].upper().startswith("3") else 4
            bound = header[2].upper()
            mode = header[3].upper()
            block = {"order": order, "bound": bound, "mode": mode}
            i += 1
            if mode == "ALL":
                block["value"] = float(header[4])
            else:  # SPECIFIC
                count = int(header[4])
                block["rows"] = [lines[i + k].split() for k in range(count)]
                i += count
            data["special_blocks"].append(block)
            continue

        else:
            # stray/unrecognized line (e.g. a free-text header comment) -- skip
            i += 1
            continue

    return data


def render(p: dict) -> str:
    atom_types = p["atom_types"]
    pairs = p["pairs"]
    natmtyp = p.get("natmtyp", len(atom_types))

    lines = [
        "####### CONTROL VARIABLES #######",
        "",
        "# TRJFILE #",
        f"\t{p['trjfile']}",
        "# WRAPTRJ #",
        f"\t{_bool_str(p.get('wraptrj', True))}",
        "# NFRAMES #",
        f"\t{p['nframes']}",
        "# NLAYERS #",
        f"\t{p.get('nlayers', 1)}",
        "# FITCOUL #",
        f"\t{_bool_str(p.get('fitcoul', False))}",
        "# FITSTRS #",
        f"\t{p.get('fitstrs', 'false')}",
        "# FITENER #",
        f"\t{p.get('fitener', 'false')}",
        "# FITPOVR #",
        f"\t{_bool_str(p.get('fitpovr', False))}",
    ]

    pairtyp_line = f"{p.get('pairtyp', 'CHEBYSHEV')} {p['order2']} {p['order3']}"
    if p.get("order4") is not None:
        pairtyp_line += f" {p['order4']} {p.get('cheby_min', -1)} {p.get('cheby_max', 1)}"
    lines += ["# PAIRTYP #", f"\t{pairtyp_line}", "# CHBTYPE #", f"\t{p.get('chbtype', 'MORSE')}"]

    if "splitfi" in p:
        lines += ["# SPLITFI #", f"\t{_bool_str(p['splitfi'])}"]
    if "useneig" in p:
        lines += ["# USENEIG #", f"\t{_bool_str(p['useneig'])}"]
    if "skpfrms" in p:
        lines += ["# SKPFRMS #", f"\t{p['skpfrms']}"]

    lines += [
        "",
        "####### TOPOLOGY VARIABLES #######",
        "",
        "# NATMTYP #",
        f"\t{natmtyp}",
        "",
        "# TYPEIDX #\t# ATM_TYP #\t# ATMCHRG #\t# ATMMASS #",
    ]
    for a in atom_types:
        lines.append(f"{a['idx']}\t{a['symbol']}\t{a['charge']}\t{a['mass']}")

    lines += [
        "",
        "# PAIRIDX #\t# ATM_TY1 #\t# ATM_TY1 #\t# S_MINIM #\t# S_MAXIM #\t# S_DELTA #\t"
        "# MORSE_LAMBDA #\t# USEOVRP #\t# NIJBINS #\t# NIKBINS #\t# NJKBINS #",
    ]
    for pr in pairs:
        row = (
            f"{pr['idx']}\t{pr['type1']}\t{pr['type2']}\t{pr['s_minim']}\t{pr['s_maxim']}\t"
            f"{pr['s_delta']}\t{pr['morse_lambda']}"
        )
        if "useovrp" in pr or "nijbins" in pr:
            row += (
                f"\t{_bool_str(pr.get('useovrp', False))}\t{pr.get('nijbins', 0)}\t"
                f"{pr.get('nikbins', 0)}\t{pr.get('njkbins', 0)}"
            )
        lines.append(row)

    lines += ["", "# FCUTTYP #", f"\t{p.get('fcuttyp', 'CUBIC')}", ""]

    for order in (3, 4):
        rows = p.get(f"exclude_{order}b")
        if rows:
            lines.append(f"EXCLUDE {order}B INTERACTION: {len(rows)}")
            lines.extend(" ".join(row) for row in rows)
            lines.append("")

    for block in p.get("special_blocks", []):
        header = f"SPECIAL {block['order']}B {block['bound']}: "
        if block["mode"] == "ALL":
            lines.append(header + f"ALL {block['value']}")
        else:
            rows = block["rows"]
            lines.append(header + f"SPECIFIC {len(rows)}")
            lines.extend(" ".join(row) for row in rows)
        lines.append("")

    lines += ["# ENDFILE #", ""]
    return "\n".join(lines)
