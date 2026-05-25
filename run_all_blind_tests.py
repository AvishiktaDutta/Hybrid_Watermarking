import os
import re
import subprocess
from pathlib import Path

COV_DIR = Path("coverImages")
OUT_DIR = Path("watermarked_outputs")
WM = Path("watermark.png")

images = sorted([p for p in COV_DIR.glob("*.png")])
results = []

for img in images:
    stem = img.stem
    print(f"\n=== Testing: {stem} ===")
    # embed
    cmd_embed = ["python", "embed.py", "--host", str(img), "--watermark", str(WM), "--output-dir", str(OUT_DIR)]
    r = subprocess.run(cmd_embed, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        results.append((stem, "embed_failed"))
        continue

    wm_img = OUT_DIR / f"{stem}_watermarked.png"
    restored = OUT_DIR / f"{stem}_restored.png"
    extracted = OUT_DIR / f"{stem}_extracted.png"

    # extract
    cmd_ext = ["python", "extraction.py", "--watermarked", str(wm_img), "--meta-dir", str(OUT_DIR), "--restored-host", str(restored), "--extracted-watermark", str(extracted)]
    r = subprocess.run(cmd_ext, capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        results.append((stem, "extract_failed"))
        continue

    # verify
    cmd_ver = ["python", "verify.py", "--input", str(img), "--watermarked-dir", str(OUT_DIR), "--restored-host", str(restored), "--extracted-watermark", str(extracted)]
    r = subprocess.run(cmd_ver, capture_output=True, text=True)
    out = r.stdout
    print(out)

    # parse metrics
    def parse(metric_name, text):
        m = re.search(rf"{metric_name}:\s*([-0-9.eiInf]+)", text)
        return m.group(1) if m else None

    host_psnr = parse("PSNR", out)
    host_ber = parse("BER", out)
    wm_ber = None
    # extract watermark BER from watermark section
    wm_section = out.split("Watermark vs Extracted Watermark")[-1] if "Watermark vs Extracted Watermark" in out else out
    m = re.search(r"BER:\s*([0-9.eE+-]+)", wm_section)
    if m:
        wm_ber = m.group(1)

    restored_section = out.split("Host vs Restored Host")[-1] if "Host vs Restored Host" in out else ""
    restored_psnr = None
    m = re.search(r"PSNR:\s*([-0-9.eEInf]+)", restored_section)
    if m:
        restored_psnr = m.group(1)

    results.append((stem, host_psnr, host_ber, wm_ber, restored_psnr))

# print summary
print("\n=== Summary ===")
print("image,host_psnr,host_ber,wm_ber,restored_psnr")
for row in results:
    print(",".join([str(x) if x is not None else "" for x in row]))

# save CSV
with open("watermarked_outputs/blind_test_summary.csv", "w") as f:
    f.write("image,host_psnr,host_ber,wm_ber,restored_psnr\n")
    for row in results:
        f.write(",".join([str(x) if x is not None else "" for x in row]) + "\n")

print("\nSaved summary to watermarked_outputs/blind_test_summary.csv")
