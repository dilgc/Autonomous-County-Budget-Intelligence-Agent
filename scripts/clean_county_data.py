# clean_county_data.py
# takes the raw comptroller zip and outputs clean csvs for the agent

import pandas as pd
import os, glob, zipfile, shutil, sys

script_dir = os.path.dirname(os.path.abspath(__file__))
proj = os.path.dirname(script_dir)
data_dir = os.path.join(proj, "data")
clean_dir = os.path.join(data_dir, "clean")
tmp = os.path.join(data_dir, "_raw")

# find the zip
zips = glob.glob(os.path.join(data_dir, "*.zip"))
if not zips:
    print("no zip found in data/ folder")
    print("grab it from: https://wwe1.osc.state.ny.us/localgov/findata/financial-data-for-local-governments.cfm")
    sys.exit(1)

print(f"using {os.path.basename(zips[0])}")

os.makedirs(tmp, exist_ok=True)
with zipfile.ZipFile(zips[0], 'r') as z:
    z.extractall(tmp)

# only need 2018-2024
years_we_want = range(2018, 2025)
csvs = sorted(glob.glob(os.path.join(tmp, "*.csv")))

frames = []
for f in csvs:
    fname = os.path.basename(f)
    try:
        yr = int(fname.split("_")[0])
    except ValueError:
        continue
    if yr not in years_we_want:
        continue
    chunk = pd.read_csv(f)
    if chunk.empty:
        print(f"  {fname} is empty, skipping")
        continue
    frames.append(chunk)
    print(f"  {fname}: {len(chunk)} rows")

df = pd.concat(frames, ignore_index=True)
print(f"\ntotal: {len(df)} rows before cleaning")

# fix county names - some are all caps, some have extra spaces
df['COUNTY'] = df['COUNTY'].str.strip().str.title()
df['ENTITY_NAME'] = df['ENTITY_NAME'].str.strip()
df['ACCOUNT_CODE_NARRATIVE'] = df['ACCOUNT_CODE_NARRATIVE'].str.strip()
df['LEVEL_1_CATEGORY'] = df['LEVEL_1_CATEGORY'].str.strip()
df['LEVEL_2_CATEGORY'] = df['LEVEL_2_CATEGORY'].str.strip()
df['ACCOUNT_CODE_SECTION'] = df['ACCOUNT_CODE_SECTION'].str.strip()

df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')

before = len(df)
df = df.dropna(subset=['AMOUNT'])
if before - len(df) > 0:
    print(f"dropped {before - len(df)} rows w/ bad amounts")

# dont need these
df = df.drop(columns=['SNAPSHOT_DATE', 'MUNICIPAL_CODE', 'CLASS_DESCRIPTION'], errors='ignore')

n_counties = df['COUNTY'].nunique()
yrs = sorted(df['CALENDAR_YEAR'].unique())
print(f"after cleaning: {len(df)} rows, {n_counties} counties, {yrs[0]}-{yrs[-1]}")

# save
os.makedirs(clean_dir, exist_ok=True)

df.to_csv(os.path.join(clean_dir, "clean_county_data.csv"), index=False)

exp = df[df['ACCOUNT_CODE_SECTION'] == 'EXPENDITURE']
exp.to_csv(os.path.join(clean_dir, "clean_county_expenditures.csv"), index=False)

rev = df[df['ACCOUNT_CODE_SECTION'] == 'REVENUE']
rev.to_csv(os.path.join(clean_dir, "clean_county_revenue.csv"), index=False)

print(f"\nsaved to data/clean/:")
print(f"  clean_county_data.csv ({len(df)} rows)")
print(f"  clean_county_expenditures.csv ({len(exp)} rows)")
print(f"  clean_county_revenue.csv ({len(rev)} rows)")

shutil.rmtree(tmp)
print("done")
