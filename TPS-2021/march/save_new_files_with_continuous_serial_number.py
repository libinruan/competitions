sub_files = []
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.startswith("submission_") and file.endswith(".csv"):
            sub_files.append(file)
sub_files
if len(sub_files) == 0:
    sub_files = ["submission_00.csv"]
sub_files.sort()
last_sub_file = sub_files[-1]
last_id = int(last_sub_file.split("_")[-1].split(".")[0])
curr_id = str(last_id + 1).zfill(2)
curr_sub_fn = "submission_" + curr_id + ".csv"  # file name
test_df["target"] = y_pred
test_df[["target"]].to_csv(curr_sub_fn)