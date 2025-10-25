import pandas as pd

# Read the CSV files
df = pd.read_csv('radiology_detail.csv')
radiology = pd.read_csv("radiology.csv")

# Step 1: Filter radiology_detail to get only chest-related notes (excluding CT, MR, etc.)
filtered_df = df[
    df['field_value'].str.contains('chest', case=False, na=False) &
    ~df['field_value'].str.contains('CT|line|U\\.S\\.|MR', case=False, na=False)
]

# Step 2: Get distinct note_ids
ids = filtered_df['note_id'].drop_duplicates()
print(f"Distinct chest-related note_ids: {len(ids)}")

# Step 3: Filter radiology for matching note_ids and text containing 'findings:' + 'pulmonary nodule'
filter_radiology = radiology[
    radiology['note_id'].isin(ids) &
    radiology['text'].str.contains('findings:', case=False, na=False) &
    radiology['text'].str.contains('pulmonary nodule', case=False, na=False)
]

# Step 4: Drop duplicates and get count
filter_radiology = filter_radiology.drop_duplicates(subset='note_id')
print(f"Distinct pulmonary nodule findings: {len(filter_radiology)}")

filter_radiology.to_csv("findings_nodule.csv", index=False)
