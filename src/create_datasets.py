import argparse
from pathlib import Path
import random
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Custom preprocessing functions (same imports as before)
from preprocessing import (
    process_metadata,
    process_test_data,
    lat_lon_to_bin,
    create_folds,
    agg_yield,
    process_blues,
    feat_eng_weather,
    feat_eng_soil,
    feat_eng_target,
    extract_target,
    create_field_location,
)

# ------------------------------
# ARGUMENTS
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cv", type=int, choices={0, 1, 2}, required=True)
parser.add_argument("--fold", type=int, choices={0, 1, 2, 3, 4}, required=True)
parser.add_argument("--seed", type=int, required=True)
args = parser.parse_args()

# ------------------------------
# CROSS VALIDATION CONFIG
# ------------------------------
if args.cv == 0:
    print("Using CV0")
    YTRAIN_YEAR, YVAL_YEAR, YTEST_YEAR = 2020, 2021, 2022
elif args.cv == 1:
    print("Using CV1")
    YTRAIN_YEAR, YVAL_YEAR, YTEST_YEAR = 2021, 2021, 2022
elif args.cv == 2:
    print("Using CV2")
    YTRAIN_YEAR, YVAL_YEAR, YTEST_YEAR = 2021, 2021, 2022

print("Using fold", args.fold)

# ------------------------------
# PATH CONFIG — FLAT OUTPUTS
# ------------------------------
OUTPUT_PATH = Path(".")  # all outputs in main directory
TRAIT_PATH = "1_Training_Trait_Data_2014_2021.csv"
TEST_PATH = "1_Submission_Template_2022.csv"
META_TRAIN_PATH = "2_Training_Meta_Data_2014_2021.csv"
META_TEST_PATH = "2_Testing_Meta_Data_2022.csv"

META_COLS = [
    "Env",
    "weather_station_lat",
    "weather_station_lon",
    "treatment_not_standard",
]
CAT_COLS = ["Env", "Hybrid"]

LAT_BIN_STEP = 1.2
LON_BIN_STEP = LAT_BIN_STEP * 3

# ------------------------------
# MAIN PIPELINE
# ------------------------------
if __name__ == "__main__":
    print("Loading metadata...")
    meta = process_metadata(META_TRAIN_PATH)
    meta_test = process_metadata(META_TEST_PATH)

    print("Loading test data...")
    test = process_test_data(TEST_PATH)
    xtest = test.merge(meta_test[META_COLS], on="Env", how="left").drop(
        ["Field_Location"], axis=1
    )
    df_sub = xtest.reset_index()[["Env", "Hybrid"]]

    print("Loading training trait data...")
    trait = pd.read_csv(TRAIT_PATH)
    trait = trait.merge(meta[META_COLS], on="Env", how="left")
    trait = create_field_location(trait)
    trait = agg_yield(trait)

    print("Loading environmental and soil data...")
    weather = pd.read_csv("4_Training_Weather_Data_2014_2021.csv")
    weather_test = pd.read_csv("4_Testing_Weather_Data_2022.csv")
    soil = pd.read_csv("3_Training_Soil_Data_2015_2021.csv")
    soil_test = pd.read_csv("3_Testing_Soil_Data_2022.csv")
    ec = pd.read_csv("6_Training_EC_Data_2014_2021.csv").set_index("Env")
    ec_test = pd.read_csv("6_Testing_EC_Data_2022.csv").set_index("Env")

    # ------------------------------
    # FOLDS AND SPLITS
    # ------------------------------
    print("Creating folds...")
    random.seed(args.seed)
    df_folds = create_folds(
        trait, val_year=YVAL_YEAR, cv=args.cv, fillna=False, random_state=args.seed
    )
    xval = (
        df_folds[df_folds["fold"] == args.fold]
        .drop("fold", axis=1)
        .reset_index(drop=True)
    )
    xtrain = (
        df_folds[df_folds["fold"] == 99].drop("fold", axis=1).reset_index(drop=True)
    )

    # ------------------------------
    # BLUES
    # ------------------------------
    print("Merging BLUEs...")
    blues = pd.read_csv("blues.csv")  # read from main dir (no output/blues.csv)
    xtrain = xtrain.merge(blues, on=["Env", "Hybrid"], how="right")
    xtrain = process_blues(xtrain)
    xval = xval.merge(blues, on=["Env", "Hybrid"], how="right")
    xval = process_blues(xval)

    # ------------------------------
    # FEATURE ENGINEERING
    # ------------------------------
    print("Engineering features...")
    xtrain = feat_eng_weather(xtrain, weather, year=YTRAIN_YEAR)
    xval = feat_eng_weather(xval, weather, year=YVAL_YEAR)
    xtest = feat_eng_weather(xtest, weather_test, year=YTEST_YEAR)

    xtrain = feat_eng_soil(xtrain, soil)
    xval = feat_eng_soil(xval, soil)
    xtest = feat_eng_soil(xtest, soil_test)

    xtrain = feat_eng_target(xtrain, ec)
    xval = feat_eng_target(xval, ec)
    xtest = feat_eng_target(xtest, ec_test)

    # ------------------------------
    # TARGET EXTRACTION
    # ------------------------------
    print("Extracting targets...")
    ytrain = extract_target(xtrain)
    yval = extract_target(xval)

    # ------------------------------
    # BINNING LAT/LON
    # ------------------------------
    print("Binning lat/lon features...")
    xtrain = lat_lon_to_bin(xtrain, LAT_BIN_STEP, LON_BIN_STEP)
    xval = lat_lon_to_bin(xval, LAT_BIN_STEP, LON_BIN_STEP)
    xtest = lat_lon_to_bin(xtest, LAT_BIN_STEP, LON_BIN_STEP)

    # ------------------------------
    # TRUNCATED SVD (OPTIONAL DIM REDUCTION)
    # ------------------------------
    print("Running SVD reduction (if applicable)...")
    numeric_cols = xtrain.select_dtypes(include=["number"]).columns
    svd = TruncatedSVD(n_components=min(10, len(numeric_cols) - 1))
    svd.fit(xtrain[numeric_cols].fillna(0))

    xtrain_svd = svd.transform(xtrain[numeric_cols].fillna(0))
    xval_svd = svd.transform(xval[numeric_cols].fillna(0))
    xtest_svd = svd.transform(xtest[numeric_cols].fillna(0))

    xtrain_svd = pd.DataFrame(
        xtrain_svd, columns=[f"svd_{i}" for i in range(xtrain_svd.shape[1])]
    )
    xval_svd = pd.DataFrame(
        xval_svd, columns=[f"svd_{i}" for i in range(xval_svd.shape[1])]
    )
    xtest_svd = pd.DataFrame(
        xtest_svd, columns=[f"svd_{i}" for i in range(xtest_svd.shape[1])]
    )

    # Combine back
    xtrain = pd.concat([xtrain.reset_index(drop=True), xtrain_svd], axis=1)
    xval = pd.concat([xval.reset_index(drop=True), xval_svd], axis=1)
    xtest = pd.concat([xtest.reset_index(drop=True), xtest_svd], axis=1)

    # ------------------------------
    # WRITE OUTPUTS — FLAT STRUCTURE
    # ------------------------------
    print("Saving flattened outputs...")
    xtrain.reset_index().to_csv(
        f"xtrain_cv{args.cv}_fold{args.fold}_seed{args.seed}.csv", index=False
    )
    xval.reset_index().to_csv(
        f"xval_cv{args.cv}_fold{args.fold}_seed{args.seed}.csv", index=False
    )
    xtest.reset_index().to_csv(
        f"xtest_cv{args.cv}_fold{args.fold}_seed{args.seed}.csv", index=False
    )
    ytrain.reset_index().to_csv(
        f"ytrain_cv{args.cv}_fold{args.fold}_seed{args.seed}.csv", index=False
    )
    yval.reset_index().to_csv(
        f"yval_cv{args.cv}_fold{args.fold}_seed{args.seed}.csv", index=False
    )

    print("\n Finished successfully.")
