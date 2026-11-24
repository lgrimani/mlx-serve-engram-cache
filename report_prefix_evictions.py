"""Summarize diagnosed cold misses in one rank0 log, excluding launch warmup."""
import argparse
from collections import Counter
from pathlib import Path


def report(path):
    victims = {}
    evictions = {}
    totals = Counter()
    recovered = Counter()
    candidates = {}
    previous = 0
    details = []
    with path.open(errors="replace") as stream:
        for number, line in enumerate(stream, 1):
            if not line.startswith("[prefix-diag] "):
                continue
            tokens = line.split()
            fields = dict(word.split("=", 1) for word in tokens[1:] if "=" in word)
            event = tokens[1]
            if event == "victim":
                victims[fields["entry"]] = fields["rule"]
            elif event == "eviction":
                trigger = line.split("trigger=", 1)[1].split(" retained_for_comparison=", 1)[0]
                rule = victims.pop(fields["entry"], trigger)
                evictions[fields["entry"]] = (trigger, rule)
                totals["evictions"] += 1
            elif event == "evicted_compare" and fields["whole_prefix_compatible"] == "true":
                request = fields["request"]
                boundary = int(fields["boundary"])
                if boundary > candidates.get(request, (0, None))[0]:
                    candidates[request] = boundary, fields["entry"]
            elif event.startswith("request="):
                seq = int(fields["request"])
                if seq <= previous:
                    victims.clear()
                    evictions.clear()
                    candidates.clear()
                    totals.clear()
                    recovered.clear()
                    details.clear()
                previous = seq
                totals["requests"] += 1
                cold = fields["restored"] == "0" and fields["bypass"] == "false"
                totals["cold"] += cold
                match = candidates.pop(str(seq), None)
                if cold and match:
                    boundary, entry = match
                    trigger, rule = evictions.get(entry, ("unknown", "unknown"))
                    recovered[trigger, rule] += 1
                    details.append((seq, int(fields["tokens"]), boundary, trigger, rule, number))
    print(path)
    print(f"{totals['requests']} requests, {totals['evictions']} evictions, {totals['cold']} cold requests, "
          f"{len(details)} cold requests with a confirmed evicted match")
    for seq, size, boundary, trigger, rule, line in details:
        print(f"  request {seq}: {boundary:,}/{size:,} tokens reusable; {trigger} / {rule} (line {line})")
    for (trigger, rule), count in recovered.most_common():
        print(f"  {trigger} / {rule}: {count} affected requests")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="coordinator log paths, one per server run")
    for log in parser.parse_args().logs:
        report(log)
