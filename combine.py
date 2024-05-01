import pandas as pd
import glob

# Define directory containing CSV files
data_dir = "dataset/"

# Empty list to store all DataFrames
all_dfs = []

csv_files = glob.glob(f"{data_dir}/*.csv")

for filename in csv_files:
    try:
        df = pd.read_csv(filename)

        df['city'] = filename.split("heat index_daily_mean_")[-1].split("_01012024-04232024.csv")[0]

        all_dfs.append(df)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Skipping.")
    except pd.errors.ParserError: # kapag hindi mabasa ng pandas
        print(f"Error: Parsing error encoutered while reading '{filename}'. Skipping")

if not all_dfs:
    print("No CSV files found in the specified directory.")
else:
    # Concatenate all DataFrame from `all_dfs` list
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Specify the output filename for the combined CSV
    output_filename = "city_heat_index.csv"

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_filename, index=False)

    print(f"CSV files successfully combined into '{output_filename}'.")
