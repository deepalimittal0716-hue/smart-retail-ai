import pandas as pd
import shutil
import os

df = pd.read_csv('data/styles.csv', on_bad_lines='skip')

mapping = {
    'Apparel': 'Clothing',
    'Footwear': 'Shoes',
}

sub_mapping = {
    'Bags': 'Bags',
}

os.makedirs('data/products/Clothing', exist_ok=True)
os.makedirs('data/products/Shoes', exist_ok=True)
os.makedirs('data/products/Bags', exist_ok=True)

counts = {'Clothing': 0, 'Shoes': 0, 'Bags': 0}
MAX_PER_CLASS = 300

for _, row in df.iterrows():
    img_id = row['id']
    src = f"data/images/{img_id}.jpg"
    if not os.path.exists(src):
        continue

    target_class = None
    if row.get('subCategory') in sub_mapping:
        target_class = sub_mapping[row['subCategory']]
    elif row.get('masterCategory') in mapping:
        target_class = mapping[row['masterCategory']]

    if target_class and counts[target_class] < MAX_PER_CLASS:
        dst = f"data/products/{target_class}/{img_id}.jpg"
        shutil.copy(src, dst)
        counts[target_class] += 1

    if all(c >= MAX_PER_CLASS for c in counts.values()):
        break

print("Done! Copied:", counts)