"""Report IPA length distribution in train_list.txt — PLBERT overflows >510 chars."""
from pathlib import Path

p = Path("/workspace/bol_run/training/train_list.txt")
lens = []
for line in p.read_text().splitlines():
    parts = line.split("|")
    if len(parts) == 3:
        lens.append(len(parts[1]))

s = sorted(lens)
n = len(lens)
print(f"total utterances: {n}")
print(f"max IPA chars:    {max(lens)}")
print(f"min IPA chars:    {min(lens)}")
print(f"mean:             {sum(lens) / n:.1f}")
print(f"p50/p90/p99/p100: {s[n // 2]}/{s[int(n * 0.9)]}/{s[int(n * 0.99)]}/{max(lens)}")
print(f"over 510:         {sum(1 for x in lens if x > 510)}")
print(f"over 400:         {sum(1 for x in lens if x > 400)}")
print(f"over 300:         {sum(1 for x in lens if x > 300)}")
print(f"over 200:         {sum(1 for x in lens if x > 200)}")
