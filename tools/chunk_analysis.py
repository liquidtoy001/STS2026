"""Analyse chunk studies (tools/chunk_study.py + tools/asr_wer.py).

analyze  one study folder: corpus WER per condition, the words the TTS must wait for before it
         can speak, the paired 95% interval of WER(condition) - WER(full), matched-wait
         comparisons ctx_N_kK vs naive_{N+K}, and the trade-off plot (analysis.json, tradeoff.png).
compare  several checkpoints on the same sentences: WER per condition and paired 95% intervals of
         each checkpoint minus the reference (and chunk minus control when both are given);
         optionally the same for tools/eval_tts010.py folders (whole held-out sentences).

Intervals: 2,000 bootstrap resamples of the sentences (random.Random(0)), the same resample for
every condition and checkpoint, 2.5th and 97.5th percentiles of the corpus-WER difference.

python tools/chunk_analysis.py analyze output/chunk-study/R0_40k
python tools/chunk_analysis.py compare --study R0=<dir> chunk=<dir> control=<dir> \
    --eval R0=<dir> chunk=<dir> control=<dir> --out comparison.json
"""

import argparse
import json
import random
from pathlib import Path

BOOTSTRAP = 2000
LABELS = {"full": "whole sentence"}
LABELS.update({f"naive_{n}": f"independent {n}-word chunks" for n in (1, 2, 4, 8)})
LABELS.update({f"ctx_{n}_k{k}": f"context-aware {n}-word chunks, lookahead {k}"
               for n in (1, 2, 4, 8) for k in (0, 2, 4)})


def rows_by_channel(asr):
    """{channel: {sentence id: (word errors, reference words)}} from an asr.json record."""
    rows = {}
    for u in asr["utterances"]:
        rows.setdefault(u["channel"], {})[u["id"]] = (u["word_s"] + u["word_d"] + u["word_i"], u["word_n"])
    return rows


def wer(rows, sample):
    return sum(rows[i][0] for i in sample) / sum(rows[i][1] for i in sample)


def resamples(ids):
    rng = random.Random(0)
    return [[rng.choice(ids) for _ in ids] for _ in range(BOOTSTRAP)]


def paired(a, b, ids, boots):
    """WER(a) - WER(b) over ids and its bootstrap 95% interval."""
    diffs = sorted(wer(a, s) - wer(b, s) for s in boots)
    return {"diff": wer(a, ids) - wer(b, ids),
            "ci95": [diffs[int(0.025 * len(boots))], diffs[int(0.975 * len(boots)) - 1]]}


def wait_words(channel, words):
    """Mean number of words that must arrive before the first audio."""
    if channel == "full":
        return sum(words.values()) / len(words)
    parts = channel.split("_")
    n, k = int(parts[1]), int(parts[2][1:]) if len(parts) > 2 else 0
    return sum(min(n + k, count) for count in words.values()) / len(words)


def analyze(study):
    study = Path(study)
    rows = rows_by_channel(json.loads((study / "asr.json").read_text(encoding="utf-8")))
    timing = json.loads((study / "timing.json").read_text(encoding="utf-8"))
    ids = sorted(rows["full"])
    words = {i: rows["full"][i][1] for i in ids}
    boots = resamples(ids)
    conditions = {}
    for channel in rows:
        if channel == "natural":
            continue
        ci = [0, 0] if channel == "full" else paired(rows[channel], rows["full"], ids, boots)["ci95"]
        conditions[channel] = {"wer": wer(rows[channel], ids), "wait_words": wait_words(channel, words),
                               "vs_full_ci95": ci,
                               "duration_ratio": timing["summary"].get(channel, {}).get("duration_ratio_mean")}
    pairs = [(f"ctx_{n}_k{k}", f"naive_{n + k}") for n in (1, 2, 4, 8) for k in (0, 2, 4) if f"naive_{n + k}" in rows]
    matched = {f"{a} vs {b}": paired(rows[a], rows[b], ids, boots) for a, b in pairs if a in rows}
    result = {"sentences": len(ids), "natural_wer": wer(rows["natural"], ids) if "natural" in rows else None,
              "conditions": conditions, "matched_wait": matched}
    (study / "analysis.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot(result, study / "tradeoff.png")
    return result


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = result["conditions"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for prefix, marker, label in (("naive", "o", "independent chunks"), ("ctx", "s", "context-aware chunks")):
        for k in ((None,) if prefix == "naive" else (0, 2, 4)):
            points = sorted((v["wait_words"], 100 * v["wer"]) for c, v in out.items()
                            if c.startswith(prefix) and (k is None or c.endswith(f"_k{k}")))
            if points:
                ax.plot(*zip(*points), marker=marker, label=label if k is None else f"{label}, lookahead {k}")
    ax.scatter([out["full"]["wait_words"]], [100 * out["full"]["wer"]], marker="*", s=150, color="k",
               label="whole sentence", zorder=5)
    if result["natural_wer"] is not None:
        ax.axhline(100 * result["natural_wer"], ls=":", color="grey", label="real recordings")
    ax.set_xlabel("Words the TTS must wait for before speaking")
    ax.set_ylabel("Whisper WER (%)")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compare_channels(models, channels, ids, reference="R0"):
    boots = resamples(ids)
    out = {}
    for channel in channels:
        entry = {name: {"wer": wer(rows[channel], ids)} for name, rows in models.items()}
        for name in models:
            if name != reference:
                entry[name]["vs_" + reference] = paired(models[name][channel], models[reference][channel], ids, boots)
        if "chunk" in models and "control" in models:
            entry["chunk"]["vs_control"] = paired(models["chunk"][channel], models["control"][channel], ids, boots)
        out[channel] = entry
    return out


def compare(study_dirs, eval_dirs, reference="R0"):
    result = {"reference": reference}
    if study_dirs:
        asr = {n: json.loads((Path(d) / "asr.json").read_text(encoding="utf-8")) for n, d in study_dirs.items()}
        models = {n: rows_by_channel(a) for n, a in asr.items()}
        channels = [c for c in models[reference] if c != "natural" and all(c in m for m in models.values())]
        ids = sorted(models[reference]["full"])
        if any(sorted(m[c]) != ids for m in models.values() for c in channels):
            raise ValueError("The studies do not cover the same sentences for every condition.")
        timing = {n: json.loads((Path(d) / "timing.json").read_text(encoding="utf-8"))["summary"]
                  for n, d in study_dirs.items()}
        natural = next((m["natural"] for m in models.values() if "natural" in m), None)
        result["chunk_study"] = {
            "sentences": len(ids), "natural_wer": wer(natural, ids) if natural else None,
            "asr": {n: a.get("asr") for n, a in asr.items()},
            "conditions": compare_channels(models, channels, ids, reference),
            "duration_ratio": {c: {n: timing[n][c]["duration_ratio_mean"] for n in models} for c in channels},
        }
    if eval_dirs:
        models = {n: rows_by_channel(json.loads((Path(d) / "asr.json").read_text(encoding="utf-8")))
                  for n, d in eval_dirs.items()}
        metrics = {n: json.loads((Path(d) / "metrics.json").read_text(encoding="utf-8")) for n, d in eval_dirs.items()}
        ids = sorted(models[reference]["synthesized"])
        result["eval_tts010"] = {
            "sentences": len(ids), "natural_wer": wer(models[reference]["natural"], ids),
            "synthesized": compare_channels(models, ["synthesized"], ids, reference)["synthesized"],
            "metrics": {n: {"checkpoint": m["checkpoint"], "heldout_full": m["heldout_full"],
                            "heldout_tts": m["heldout_tts"]} for n, m in metrics.items()},
        }
    return result


def markdown(result):
    """Tables for docs: every model's WER and the paired differences in points [95% interval]."""
    pct = lambda v: f"{100 * v:.1f}"
    diff = lambda d: f"{100 * d['diff']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}]"
    reference, lines = result["reference"], []
    for section in ("eval_tts010", "chunk_study"):
        if section not in result:
            continue
        block = result[section]
        conditions = {"synthesized": block["synthesized"]} if section == "eval_tts010" else block["conditions"]
        names = list(next(iter(conditions.values())))
        others = [n for n in names if n != reference]
        with_control = "chunk" in names and "control" in names
        columns = [f"{n} - {reference}" for n in others] + (["chunk - control"] if with_control else [])
        lines.append("| Condition | " + " | ".join(f"{n} WER (%)" for n in names) + " | "
                     + " | ".join(f"{c} [95% CI]" for c in columns) + " |")
        lines.append("|---|" + "---:|" * (len(names) + len(columns)))
        for channel, entry in conditions.items():
            cells = [pct(entry[n]["wer"]) for n in names] + [diff(entry[n]["vs_" + reference]) for n in others]
            if with_control:
                cells.append(diff(entry["chunk"]["vs_control"]))
            label = "whole held-out sentence" if section == "eval_tts010" else LABELS.get(channel, channel)
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        if block.get("natural_wer") is not None:
            lines.append(f"\nLJSpeech recordings of the same sentences: {pct(block['natural_wer'])}% WER.")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("analyze", help="one study folder")
    one.add_argument("study")
    many = commands.add_parser("compare", help="several checkpoints, paired per sentence")
    many.add_argument("--study", nargs="*", default=[], help="NAME=chunk-study folder (the first-named reference: R0)")
    many.add_argument("--eval", nargs="*", default=[], help="NAME=tools/eval_tts010.py folder")
    many.add_argument("--reference", default="R0")
    many.add_argument("--out", required=True, help="comparison JSON; a Markdown table is written next to it")
    args = parser.parse_args()
    if args.command == "analyze":
        result = analyze(args.study)
        print(json.dumps({c: {"wer": round(100 * v["wer"], 1), "wait": round(v["wait_words"], 1),
                              "vs_full_ci95": [round(100 * x, 1) for x in v["vs_full_ci95"]]}
                          for c, v in result["conditions"].items()}, indent=0))
    else:
        parse = lambda items: dict(item.split("=", 1) for item in items)
        result = compare(parse(args.study), parse(args.eval), args.reference)
        out = Path(args.out)
        out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        out.with_suffix(".md").write_text(markdown(result), encoding="utf-8")
        print(markdown(result))


if __name__ == "__main__":
    main()
