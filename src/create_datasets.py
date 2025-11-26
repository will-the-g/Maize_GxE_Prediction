import argparse
from pathlib import Path
import random

import pandas as pd
from sklearn.decomposition import TruncatedSVD

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
    create_field_location
)


parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, choices={0, 1, 2}, required=True)
parser.add_argument('--fold', type=int, choices={0, 1, 2, 3, 4}, required=True)
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()

if args.cv == 0:
    print('Using CV0')
    YTRAIN_YEAR = 2020
    YVAL_YEAR = 2021
    YTEST_YEAR = 2022
elif args.cv == 1:
    print('Using CV1')
    YTRAIN_YEAR = 2021
    YVAL_YEAR = 2021
    YTEST_YEAR = 2022
elif args.cv == 2:
    print('Using CV2')
    YTRAIN_YEAR = 2021
    YVAL_YEAR = 2021
    YTEST_YEAR = 2022
print('Using fold', args.fold)

OUTPUT_PATH = Path('.')
TRAIT_PATH = '1_Training_Trait_Data_2014_2021.csv'
TEST_PATH = '1_Submission_Template_2022.csv'
META_TRAIN_PATH = '2_Training_Meta_Data_2014_2021.csv'
META_TEST_PATH = '2_Testing_Meta_Data_2022.csv'

META_COLS = ['Env', 'weather_station_lat', 'weather_station_lon', 'treatment_not_standard']
CAT_COLS = ['Env', 'Hybrid']

LAT_BIN_STEP = 1.2
LON_BIN_STEP = LAT_BIN_STEP * 3


if __name__ == '__main__':

    meta = process_metadata(META_TRAIN_PATH)
    meta_test = process_metadata(META_TEST_PATH)

    test = process_test_data(TEST_PATH)
    xtest = test.merge(meta_test[META_COLS], on='Env', how='left').drop(['Field_Location'], axis=1)
    df_sub = xtest.reset_index()[['Env', 'Hybrid']]

    trait = pd.read_csv(TRAIT_PATH)
    trait = trait.merge(meta[META_COLS], on='Env', how='left')
    trait = create_field_location(trait)

    trait = agg_yield(trait)

    weather = pd.read_csv('4_Training_Weather_Data_2014_2021.csv')
    weather_test = pd.read_csv('4_Testing_Weather_Data_2022.csv')

    soil = pd.read_csv('3_Training_Soil_Data_2015_2021.csv')
    soil_test = pd.read_csv('3_Testing_Soil_Data_2022.csv')

    ec = pd.read_csv('6_Training_EC_Data_2014_2021.csv').set_index('Env')
    ec_test = pd.read_csv('6_Testing_EC_Data_2022.csv').set_index('Env')

    random.seed(args.seed)
    df_folds = create_folds(trait, val_year=YVAL_YEAR, cv=args.cv, fillna=False, random_state=args.seed)
    xval = df_folds[df_folds['fold'] == args.fold].drop('fold', axis=1).reset_index(drop=True)
    xtrain = df_folds[df_folds['fold'] == 99].drop('fold', axis=1).reset_index(drop=True)
    print('val to train ratio:', len(set(xval['Hybrid'])) / len(set(xtrain['Hybrid'])))

    if args.cv == 0:
        candidates = list(set(df_folds['Hybrid']) - set(xval['Hybrid']))
        selected = random.choices(candidates, k=int(len(candidates) * 0.6))
        xtrain = xtrain[xtrain['Hybrid'].isin(selected + xval['Hybrid'].tolist())].reset_index(drop=True)
        print('val to train ratio:', len(set(xval['Hybrid'])) / len(set(xtrain['Hybrid'])))
        assert set(xtrain['Field_Location']) == set(xval['Field_Location'])
        assert set(xtrain['Year']) & set(xval['Year']) == set()
    elif args.cv == 1:
        xtrain = xtrain[~xtrain['Hybrid'].isin(xval['Hybrid'])].reset_index(drop=True)
        assert set(xtrain['Field_Location']) == set(xval['Field_Location'])
        assert set(xtrain['Hybrid']) & set(xval['Hybrid']) == set()
    else:
        xtrain = xtrain[~xtrain['Loc_Hybrid'].isin(xval['Loc_Hybrid'])].reset_index(drop=True)
        assert set(xtrain['Loc_Hybrid']) & set(xval['Loc_Hybrid']) == set()
        del xtrain['Loc_Hybrid'], xval['Loc_Hybrid']

    del xtrain['Field_Location'], xval['Field_Location']
    del xtrain['Year'], xval['Year']

    blues = pd.read_csv('blues.csv')
    xtrain = xtrain.merge(blues, on=['Env', 'Hybrid'], how='right')
    xtrain = process_blues(xtrain)
    xval = xval.merge(blues, on=['Env', 'Hybrid'], how='right')
    xval = process_blues(xval)

    weather_feats = feat_eng_weather(weather)
    weather_test_feats = feat_eng_weather(weather_test)
    xtrain = xtrain.merge(weather_feats, on='Env', how='left')
    xval = xval.merge(weather_feats, on='Env', how='left')
    xtest = xtest.merge(weather_test_feats, on='Env', how='left')


    xtrain = xtrain.merge(feat_eng_soil(soil), on='Env', how='left')
    xval = xval.merge(feat_eng_soil(soil), on='Env', how='left')
    xtest = xtest.merge(feat_eng_soil(soil_test), on='Env', how='left')

    xtrain_ec = ec[ec.index.isin(xtrain['Env'])].copy()
    xval_ec = ec[ec.index.isin(xval['Env'])].copy()
    xtest_ec = ec_test[ec_test.index.isin(xtest['Env'])].copy()

    n_components = 15
    svd = TruncatedSVD(n_components=n_components, n_iter=20, random_state=args.seed)
    svd.fit(xtrain_ec)
    print('SVD explained variance:', svd.explained_variance_ratio_.sum())

    xtrain_ec = pd.DataFrame(svd.transform(xtrain_ec), index=xtrain_ec.index)
    component_cols = [f'EC_svd_comp{i}' for i in range(xtrain_ec.shape[1])]
    xtrain_ec.columns = component_cols
    xval_ec = pd.DataFrame(svd.transform(xval_ec), columns=component_cols, index=xval_ec.index)
    xtest_ec = pd.DataFrame(svd.transform(xtest_ec), columns=component_cols, index=xtest_ec.index)

    xtrain = xtrain.merge(xtrain_ec, on='Env', how='left')
    xval = xval.merge(xval_ec, on='Env', how='left')
    xtest = xtest.merge(xtest_ec, on='Env', how='left')

    xtrain = create_field_location(xtrain)
    xval = create_field_location(xval)
    xtest = create_field_location(xtest)
    xtrain = xtrain.merge(feat_eng_target(trait, ref_year=YTRAIN_YEAR, lag=2), on='Field_Location', how='left')
    xval = xval.merge(feat_eng_target(trait, ref_year=YVAL_YEAR, lag=2), on='Field_Location', how='left')
    xtest = xtest.merge(feat_eng_target(trait, ref_year=YTEST_YEAR, lag=2), on='Field_Location', how='left')
    del xtrain['Field_Location'], xval['Field_Location'], xtest['Field_Location']

    for dfs in [xtrain, xval, xtest]:
        dfs['T2M_std_spring_X_weather_station_lat'] = dfs['T2M_std_spring'] * dfs['weather_station_lat']
        dfs['T2M_std_fall_X_weather_station_lat'] = dfs['T2M_std_fall'] * dfs['weather_station_lat']
        dfs['T2M_min_fall_X_weather_station_lat'] = dfs['T2M_min_fall'] * dfs['weather_station_lat']
        dfs['weather_station_lat'] = dfs['weather_station_lat'].apply(lambda x: lat_lon_to_bin(x, LAT_BIN_STEP))
        dfs['weather_station_lon'] = dfs['weather_station_lon'].apply(lambda x: lat_lon_to_bin(x, LON_BIN_STEP))


    xtrain = xtrain[~xtrain['Yield_Mg_ha'].isnull()].reset_index(drop=True)
    xval = xval[~xval['Yield_Mg_ha'].isnull()].reset_index(drop=True)
    xtest = xtest[~xtest['Yield_Mg_ha'].isnull()].reset_index(drop=True)

    xtrain = xtrain.set_index(['Env', 'Hybrid'])
    xval = xval.set_index(['Env', 'Hybrid'])
    xtest = xtest.set_index(['Env', 'Hybrid'])

    ytrain = extract_target(xtrain)
    yval = extract_target(xval)
    _ = extract_target(xtest)


    for col in [x for x in xtrain.columns if x not in CAT_COLS]:
        mean = xtrain[col].mean()
        xtrain.fillna({col: mean}, inplace=True)
        xval.fillna({col: mean}, inplace=True)
        xtest.fillna({col: mean}, inplace=True)


    assert xtrain.index.names == ['Env', 'Hybrid']
    assert xval.index.names == ['Env', 'Hybrid']

    xtrain.reset_index().to_csv(OUTPUT_PATH / f'cv{args.cv}_xtrain_fold{args.fold}_seed{args.seed}.csv', index=False)
    xval.reset_index().to_csv(OUTPUT_PATH / f'cv{args.cv}_xval_fold{args.fold}_seed{args.seed}.csv', index=False)
    ytrain.reset_index().to_csv(OUTPUT_PATH / f'cv{args.cv}_ytrain_fold{args.fold}_seed{args.seed}.csv', index=False)
    yval.reset_index().to_csv(OUTPUT_PATH / f'cv{args.cv}_yval_fold{args.fold}_seed{args.seed}.csv', index=False)
