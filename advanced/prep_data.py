import pandas as pd
import os
import time

#Take starting timestamp
t0 = time.time()

# Constants
DATASET_FRACTION = 0.05 # Use 5% of the data
DATA_FOLDER = "lpf_data"

# clear databuffer
full_df = None
print("\nRefreshing Master Dataset\n")


# cycle thru all csv files in data folder
for root,dirs,files in os.walk(f"{DATA_FOLDER}/"):
    for file in files:
       if file.endswith(".csv"):
            full_file = f"{DATA_FOLDER}/{file}"
            print("Opening File: " + full_file)
            df = pd.read_csv(full_file, low_memory=False)
            dataset = df.sample(frac=DATASET_FRACTION, random_state=0)
            # Add the weight
            dataset['Weight'] = file.replace('g.csv','.00')
            # Drop the name and times data
            dataset = dataset.drop(columns=['Name'])
            # Make all values a floating point value
            dataset = dataset.apply(pd.to_numeric)

            if full_df is None:
                full_df = dataset
            else:
                full_df = full_df.append(dataset, ignore_index=True)


print("Saving Master File: weights_dataset.csv")
# float_format='{:f}'.format, encoding='utf-8'
full_df.to_csv('weights_dataset.csv', index=False, header=None)

t1 = time.time()
total_time = t1-t0
print("\nTotal Processing Time (s): {}".format(total_time))
