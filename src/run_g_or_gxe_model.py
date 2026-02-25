import gc
import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.decomposition import TruncatedSVD

# Add the src directory to the path so we can import local modules
sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import create_field_location
from evaluate import create_df_eval, avg_rmse, feat_imp


parser = argparse.ArgumentParser()
parser.add_argument('--cv', type=int, choices={0, 1, 2}, required=True)
parser.add_argument('--fold', type=int, choices={0, 1, 2, 3, 4}, required=True)
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--model', choices={'G', 'GxE'}, required=True)
parser.add_argument('--A', action='store_true', default=False)
parser.add_argument('--D', action='store_true', default=False)
parser.add_argument('--E', action='store_true', default=False)
parser.add_argument('--svd', action='store_true', default=False)
parser.add_argument('--n_components', type=int, default=100)
parser.add_argument('--lag_features', action='store_true', default=False)
args = parser.parse_args()

PREFIX = f'cv{args.cv}_'

if args.model == 'G':
    outfile = f'{PREFIX}oof_g_model_fold{args.fold}_seed{args.seed}'
    print('Using G model.')
else:
    print('Using GxE model.')
    outfile = f'{PREFIX}oof_gxe_model_fold{args.fold}_seed{args.seed}'

# Build complete output filename upfront based on all arguments
# This ensures oof_outfile and pred_train_outfile are set correctly even if an exception occurs
if args.A:
    outfile = f'{outfile}_A'
if args.D:
    outfile = f'{outfile}_D'
if args.E:
    outfile = f'{outfile}_E'
if args.lag_features:
    outfile = f'{outfile}_lag_features'
if args.svd:
    outfile = f'{outfile}_svd{args.n_components}comps'

oof_outfile = f'{outfile}.csv'
pred_train_outfile = oof_outfile.replace('oof_', 'pred_train_')


def preprocess_g(df, kinship, individuals: list):
    individuals = [x.replace('Hybrid', '') if x.startswith('Hybrid') else x for x in individuals]
    df.columns = [x[:len(x) // 2].rstrip('_') for x in df.columns]  # fix duplicated column names
    df.index = df.columns
    #print("Index: \n", df.index)
    #print("Individuals: \n ", individuals[:20])
    df = df[df.index.isin(individuals)]  # filter rows
    df = df[[col for col in df.columns if col in individuals]]  # filter columns
    df.index.name = 'Hybrid'
    df.columns = [f'{x}_{kinship}' for x in df.columns]
    return df


def preprocess_kron(df, kinship):
    print(f"Dataframe: {df.head()}")
    print(f"Columns: {df.columns}")
    df[['Env', 'Hybrid']] = df['id'].str.split(':', expand=True)
    df = df.drop('id', axis=1).set_index(['Env', 'Hybrid'])
    df.columns = [f'{x}_{kinship}' for x in df.columns]
    # print(df.info(), '\n')
    # df[df.columns] = np.array(df.values, dtype=np.float32)  # downcast is too slow
    # print(df.info(), '\n')
    return df


def prepare_gxe(kinship):
    print(f"KINSHIP: {kinship}")
    kron = pd.read_feather(Path(f'{PREFIX}kronecker_{kinship}.arrow'))
    print(f"KRON: {kron}")
    kron = preprocess_kron(kron, kinship=kinship)
    return kron


if __name__ == '__main__':

    try:
        # load targets
        print('Loading target data...')
        ytrain = pd.read_csv(Path(f'{PREFIX}ytrain_fold{args.fold}_seed{args.seed}.csv'))
        yval = pd.read_csv(Path(f'{PREFIX}yval_fold{args.fold}_seed{args.seed}.csv'))
        individuals = ytrain['Hybrid'].unique().tolist() + yval['Hybrid'].unique().tolist()
        individuals = list(dict.fromkeys(individuals))  # take unique but preserves order (python 3.7+)
        yval["Hybrid"] = yval["Hybrid"].str.replace(r'^Hybrid', '', regex=True)

        # load kinships or kroneckers
        print('Loading kinships/kroneckers...')
        kinships = []
        kroneckers = []
        if args.A:
            print('Using A matrix.')
            if args.model == 'G':
                A = pd.read_csv('kinship_additive.txt', sep='\t')
                print("A:\n ", A)
                A = preprocess_g(A, 'A', individuals)
                print("A2:\n ", A)
                kinships.append(A)
            else:
                kroneckers.append(prepare_gxe('additive'))

        if args.D:
            print('Using D matrix.')
            if args.model == 'G':
                D = pd.read_csv('kinship_dominant.txt', sep='\t')
                D = preprocess_g(D, 'D', individuals)
                kinships.append(D)
            else:
                kroneckers.append(prepare_gxe('dominant'))

        if args.E:
            if args.model == 'G':
                print('Using E matrix.')
                Etrain = pd.read_csv(Path(f'{PREFIX}xtrain_fold{args.fold}_seed{args.seed}.csv'))
                Eval = pd.read_csv(Path(f'{PREFIX}xval_fold{args.fold}_seed{args.seed}.csv'))
            else:
                raise Exception('G+E+GxE is not implemented.')

        print('Using fold', args.fold)

        if (args.model == 'G' and len(kinships) == 0) or (args.model == 'GxE' and len(kroneckers) == 0):
            raise Exception('Choose at least one matrix.')

        # concat dataframes and bind target
        print('Merging data...')
        ytrain["Hybrid"] = ytrain["Hybrid"].str.replace(r'^Hybrid', '', regex=True)
        if args.model == 'G':
            K = pd.concat(kinships, axis=1)
            merged = pd.merge(ytrain, K, on='Hybrid', how='left')
            merged_clean = merged.dropna()
            xtrain = merged_clean.set_index(['Env', 'Hybrid'])
            xval = pd.merge(yval, K, on='Hybrid', how='left').dropna().set_index(['Env', 'Hybrid'])
            del kinships
            gc.collect()
        else:
            kron = pd.concat(kroneckers, axis=1)
            del kroneckers
            xtrain = pd.merge(ytrain, kron, on=['Env', 'Hybrid'], how='inner')
            xval = pd.merge(yval, kron, on=['Env', 'Hybrid'], how='inner')
            del kron
            gc.collect()

        # split x, y
        print('Splitting features and target...')
        ytrain = xtrain['Yield_Mg_ha']
        del xtrain['Yield_Mg_ha']
        yval = xval['Yield_Mg_ha']
        del xval['Yield_Mg_ha']
        gc.collect()

        # include E matrix if requested
        if args.E:
            Etrain = Etrain.dropna(axis=1)
            Eval = Eval.dropna(axis=1)
            Etrain["Hybrid"] = Etrain["Hybrid"].str.replace(r'^Hybrid', '', regex=True)
            Eval["Hybrid"] = Eval["Hybrid"].str.replace(r'^Hybrid', '', regex=True)
            xtrain = xtrain.merge(Etrain, on=['Env', 'Hybrid'], how='left').copy().set_index(['Env', 'Hybrid'])
            xval = xval.merge(Eval, on=['Env', 'Hybrid'], how='left').copy().set_index(['Env', 'Hybrid'])
            lag_cols = xtrain.filter(regex='_lag', axis=1).columns
            if len(lag_cols) > 0:
                xtrain = xtrain.drop(lag_cols, axis=1)
                xval = xval.drop(lag_cols, axis=1)

        # bind lagged yield features
        no_lags_cols = [x for x in xtrain.columns.tolist() if x not in ['Env', 'Hybrid']]

        if args.lag_features:
            print('Loading lag features...')
            xtrain_lag = pd.read_csv(
                Path(f'{PREFIX}xtrain_fold{args.fold}_seed{args.seed}.csv'),
                usecols=lambda x: 'yield_lag' in x or x in ['Env', 'Hybrid'],
            ).set_index(['Env', 'Hybrid'])
            xval_lag = pd.read_csv(
                Path(f'{PREFIX}xval_fold{args.fold}_seed{args.seed}.csv'),
                usecols=lambda x: 'yield_lag' in x or x in ['Env', 'Hybrid'],
            ).set_index(['Env', 'Hybrid'])

            xtrain_lag.index = xtrain_lag.index.set_levels(
                xtrain_lag.index.levels[1].str.replace("Hybrid", "", regex=True), level=1
            )
            xval_lag.index = xval_lag.index.set_levels(
                xval_lag.index.levels[1].str.replace("Hybrid", "", regex=True), level=1
            )

            xtrain = xtrain.merge(xtrain_lag, on=['Env', 'Hybrid'], how='inner').copy()
            xval = xval.merge(xval_lag, on=['Env', 'Hybrid'], how='inner').copy()

        if args.model == 'GxE':
            if 'Env' in xtrain.columns and 'Hybrid' in xtrain.columns:
                xtrain = xtrain.set_index(['Env', 'Hybrid'])
                xval = xval.set_index(['Env', 'Hybrid'])

        if not args.svd:
            print('Training model without SVD...')
            # add factor
            xtrain = xtrain.reset_index()
            xtrain = create_field_location(xtrain)
            xtrain['Field_Location'] = xtrain['Field_Location'].astype('category')
            xtrain = xtrain.set_index(['Env', 'Hybrid'])
            xval = xval.reset_index()
            xval = create_field_location(xval)
            xval['Field_Location'] = xval['Field_Location'].astype('category')
            xval = xval.set_index(['Env', 'Hybrid'])

            # include E matrix if requested
            if args.E:
                lag_cols = xtrain.filter(regex='_lag', axis=1).columns
                if len(lag_cols) > 0:
                    xtrain = xtrain.drop(lag_cols, axis=1)
                    xval = xval.drop(lag_cols, axis=1)
                xtrain = xtrain.merge(Etrain, on=['Env', 'Hybrid'], how='left').set_index(['Env', 'Hybrid'])
                xval = xval.merge(Eval, on=['Env', 'Hybrid'], how='left').set_index(['Env', 'Hybrid'])
                del Etrain, Eval
                gc.collect()

            print('Using full set of features.')
            print('# Features:', xtrain.shape[1])

            # fit
            print('Fitting model...')
            model = lgbm.LGBMRegressor(random_state=args.seed, max_depth=3)
            model.fit(xtrain, ytrain)

            # predict
            print('Making predictions...')
            ypred_train = model.predict(xtrain)
            ypred = model.predict(xval)

            # validate
            print('Evaluating predictions...')
            df_eval_train = create_df_eval(xtrain, ytrain, ypred_train)
            df_eval = create_df_eval(xval, yval, ypred)
            _ = avg_rmse(df_eval)

        else:
            print('Using svd.')
            print('# Components:', args.n_components)
            print('Fitting SVD...')
            svd = TruncatedSVD(n_components=args.n_components, random_state=args.seed)
            print("Null Values: ", xtrain[no_lags_cols].isna().sum().sum())
            svd.fit(xtrain[no_lags_cols])  # fit but without lagged yield features

            print('Explained variance:', svd.explained_variance_ratio_.sum())

            # transform from the fitted svd
            print('Transforming features with SVD...')
            svd_cols = [f'svd{i}' for i in range(args.n_components)]
            xtrain_svd = pd.DataFrame(
                svd.transform(xtrain[no_lags_cols]),
                columns=svd_cols,
                index=xtrain[no_lags_cols].index,
            )
            xval_svd = pd.DataFrame(
                svd.transform(xval[no_lags_cols]),
                columns=svd_cols,
                index=xval[no_lags_cols].index,
            )
            del svd
            gc.collect()

            # bind lagged yield features if needed
            if args.lag_features:
                print('Merging lag features with SVD components...')
                xtrain = xtrain_svd.merge(xtrain_lag, on=['Env', 'Hybrid'], how='inner').copy()
                del xtrain_svd, xtrain_lag
                xval = xval_svd.merge(xval_lag, on=['Env', 'Hybrid'], how='inner').copy()
                del xval_svd, xval_lag
                gc.collect()
            else:
                xtrain = xtrain_svd.copy()
                del xtrain_svd
                xval = xval_svd.copy()
                del xval_svd
                gc.collect()

        if args.svd:
            print('Training model with SVD features...')
            # add factor
            xtrain = xtrain.reset_index()
            xtrain = create_field_location(xtrain)
            xtrain['Field_Location'] = xtrain['Field_Location'].astype('category')
            xtrain = xtrain.set_index(['Env', 'Hybrid'])
            xval = xval.reset_index()
            xval = create_field_location(xval)
            xval['Field_Location'] = xval['Field_Location'].astype('category')
            xval = xval.set_index(['Env', 'Hybrid'])

            print('Fitting model...')
            model = lgbm.LGBMRegressor(random_state=args.seed, max_depth=3)
            model.fit(xtrain, ytrain)

            # predict
            print('Making predictions...')
            ypred_train = model.predict(xtrain)
            ypred = model.predict(xval)

            # validate
            print('Evaluating predictions...')
            df_eval_train = create_df_eval(xtrain, ytrain, ypred_train)
            df_eval = create_df_eval(xval, yval, ypred)
            _ = avg_rmse(df_eval)

            # feature importance
            print('Computing feature importance...')
            df_feat_imp = feat_imp(model)
            feat_imp_outfile = f'{outfile.replace("oof", "feat_imp")}.csv'
            df_feat_imp.to_csv(feat_imp_outfile, index=False)

    except Exception as e:
        print(f'\n\n*** FATAL ERROR OCCURRED ***')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {e}')
        import traceback
        traceback.print_exc()
        print(f'*** Creating empty output files to prevent transfer failure ***\n')

    # Always create output files (even if empty) to prevent transfer errors
    try:
        print('Writing file:', oof_outfile, '\n')
        if 'df_eval' in locals():
            df_eval.to_csv(oof_outfile, index=False)
            print('Successfully wrote:', oof_outfile)
        else:
            # Create empty file with headers if df_eval doesn't exist
            pd.DataFrame(
                columns=['Env', 'Hybrid', 'Yield_Mg_ha', 'Yield_Mg_ha_pred']
            ).to_csv(oof_outfile, index=False)
            print(f'Created empty file: {oof_outfile} (script failed before generating predictions)')
    except Exception as e:
        print(f'ERROR writing {oof_outfile}: {e}')
        # Create empty file to prevent transfer error
        Path(oof_outfile).touch()

    try:
        print('Writing file:', pred_train_outfile, '\n')
        if 'df_eval_train' in locals():
            df_eval_train.to_csv(pred_train_outfile, index=False)
            print('Successfully wrote:', pred_train_outfile)
        else:
            # Create empty file with headers if df_eval_train doesn't exist
            pd.DataFrame(
                columns=['Env', 'Hybrid', 'Yield_Mg_ha', 'Yield_Mg_ha_pred']
            ).to_csv(pred_train_outfile, index=False)
            print(f'Created empty file: {pred_train_outfile} (script failed before generating predictions)')
    except Exception as e:
        print(f'ERROR writing {pred_train_outfile}: {e}')
        # Create empty file to prevent transfer error
        Path(pred_train_outfile).touch()
