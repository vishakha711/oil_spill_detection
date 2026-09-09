from pathlib import Path
import shutil

ROOT=Path("C:/oilspill_project/data")
SOURCE=ROOT/"raw"/"dartis_download"

for d in ["oil/coast","oil/water"]+[f"no_oil/coast/c{i:02d}" for i in range(5)]+[f"no_oil/water/c{i:02d}" for i in range(12)]:
    (ROOT/d).mkdir(parents=True,exist_ok=True)

for p in SOURCE.glob("*.jpg"):
    name=p.name

    if name.startswith("oc-"):
        dest=ROOT/"oil/coast"
    elif name.startswith("ow-"):
        dest=ROOT/"oil/water"
    elif name.startswith("nc-"):
        dest=ROOT/"no_oil"/"coast"/f"c{int(name.split('-')[2]):02d}"
    elif name.startswith("nw-"):
        dest=ROOT/"no_oil"/"water"/f"c{int(name.split('-')[2]):02d}"
    else:
        continue

    if not (dest/name).exists():
        shutil.copy2(p,dest/name)

print("Dataset organization completed successfully.")