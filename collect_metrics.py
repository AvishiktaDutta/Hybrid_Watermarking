import re
import subprocess
from pathlib import Path

COV_DIR = Path('coverImages')
OUT_DIR = Path('watermarked_outputs')
images = sorted([p for p in COV_DIR.glob('*.png')])
rows = []
for img in images:
    stem = img.stem
    restored = OUT_DIR / f"{stem}_restored.png"
    extracted = OUT_DIR / f"{stem}_extracted.png"
    cmd = ['python','verify.py','--input',str(img),'--watermarked-dir',str(OUT_DIR),'--restored-host',str(restored),'--extracted-watermark',str(extracted)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = r.stdout
    # parse
    def find(label, text):
        m = re.search(rf"{label}:\s*([-0-9.eEInf]+)", text)
        return m.group(1) if m else ''
    # host section
    host_psnr = find('PSNR', out.split('Host vs Watermarked')[1].split('\n')[0]) if 'Host vs Watermarked' in out else ''
    host_ssim = ''
    m = re.search(r"SSIM:\s*([-0-9.eEInf]+)", out)
    if m:
        host_ssim = m.group(1)
    # more robust parsing per section
    host_section = out.split('Host vs Watermarked')[-1].split('\n\n')[0]
    host_psnr = find('PSNR', host_section)
    host_ssim = find('SSIM', host_section)
    host_ber = find('BER', host_section)

    wm_section = out.split('Watermark vs Extracted Watermark')[-1].split('\n\n')[0]
    wm_psnr = find('PSNR', wm_section)
    wm_ssim = find('SSIM', wm_section)
    wm_ber = find('BER', wm_section)

    restored_section = out.split('Host vs Restored Host')[-1].split('\n\n')[0] if 'Host vs Restored Host' in out else ''
    restored_psnr = find('PSNR', restored_section)
    restored_ssim = find('SSIM', restored_section)
    restored_ber = find('BER', restored_section)

    rows.append((stem, host_psnr, host_ssim, host_ber, wm_psnr, wm_ssim, wm_ber, restored_psnr, restored_ssim, restored_ber))

# write CSV
with open(OUT_DIR / 'detailed_metrics.csv', 'w') as f:
    f.write('image,host_psnr,host_ssim,host_ber,wm_psnr,wm_ssim,wm_ber,restored_psnr,restored_ssim,restored_ber\n')
    for r in rows:
        f.write(','.join(r) + '\n')
print('Saved to', OUT_DIR / 'detailed_metrics.csv')
