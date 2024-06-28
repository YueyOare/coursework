import os
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, \
    Birch, OPTICS
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, mutual_info_score, \
    normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score
from itertools import product
from sklearn.decomposition import PCA


def load_data(num_file, cat_file):
    num_data = pd.read_csv(num_file)
    cat_data = pd.read_csv(cat_file)
    return pd.merge(num_data, cat_data, on='Id')


def load_experiment_results(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep='\t')
    else:
        return pd.DataFrame(
            # columns=['model_name', 'method', 'data', 'parameters', 'RI', 'ARI', 'MI', 'NMI', 'AMI', 'homogeneity',
            #          'completeness', 'v_measure', 'DBI', 'CHI', 'Silhouette', 'n_clusters', 'has_outliers'])
            columns=['model_name', 'method', 'data', 'pca_params', 'parameters', 'RI', 'ARI', 'MI', 'NMI', 'AMI',
                     'homogeneity', 'completeness', 'v_measure', 'DBI', 'CHI', 'Silhouette', 'n_clusters', 'has_outliers'])


def calculate_additional_metrics(data, labels_pred):
    metrics = {}

    # Davies-Bouldin Index
    try:
        dbi = davies_bouldin_score(data, labels_pred)
        metrics['DBI'] = dbi
    except:
        print("Error calculating Davies-Bouldin Index")
        metrics['DBI'] = np.nan

    # Calinski-Harabasz Index
    try:
        chi = calinski_harabasz_score(data, labels_pred)
        metrics['CHI'] = chi
    except:
        print("Error calculating Calinski-Harabasz Index")
        metrics['CHI'] = np.nan

    # Silhouette Coefficient
    try:
        silhouette = silhouette_score(data, labels_pred)
        metrics['Silhouette'] = silhouette
    except:
        print("Error calculating Silhouette Coefficient")
        metrics['Silhouette'] = np.nan

    try:
        unique_clusters = np.unique([label for label in labels_pred if label >= 0])
        metrics['n_clusters'] = len(unique_clusters)
        metrics['has_outliers'] = len(unique_clusters) != len(np.unique(labels_pred))
    except:
        print("Error calculating n_clusters")
        metrics['n_clusters'] = np.nan
        metrics['has_outliers'] = np.nan

    return metrics


def get_model_instance(method, params):
    if method == 'KMeans':
        return KMeans(**params)
    elif method == 'DBSCAN':
        return DBSCAN(**params, n_jobs=-1)
    elif method == 'HDBSCAN':
        return HDBSCAN(**params, n_jobs=-1)
    elif method == 'AffinityPropagation':
        return AffinityPropagation(**params)
    elif method == 'MeanShift':
        return MeanShift(**params, n_jobs=-1)
    elif method == 'SpectralClustering':
        return SpectralClustering(**params, n_jobs=-1)
    elif method == 'AgglomerativeClustering':
        return AgglomerativeClustering(**params)
    elif method == 'Birch':
        return Birch(**params)
    elif method == 'OPTICS':
        return OPTICS(**params, n_jobs=-1)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_metrics(labels_true, labels_pred):
    h, c, v = homogeneity_completeness_v_measure(labels_true, labels_pred)
    return {
        'RI': rand_score(labels_true, labels_pred),
        'ARI': adjusted_rand_score(labels_true, labels_pred),
        'MI': mutual_info_score(labels_true, labels_pred),
        'NMI': normalized_mutual_info_score(labels_true, labels_pred),
        'AMI': adjusted_mutual_info_score(labels_true, labels_pred),
        'homogeneity': h,
        'completeness': c,
        'v_measure': v
    }


def save_results(experiment_results, model_name, method, num_data, cat_data, param_dict, metrics_values,
                 additional_metrics, predictions_df, pca_params=None):
    experiment_results.loc[len(experiment_results)] = {
        'model_name': model_name,
        'method': method,
        'data': f'{num_data}_{cat_data}',
        'pca_params': str(pca_params),
        'parameters': str(param_dict),
        **metrics_values,
        **additional_metrics
    }
    predictions_df.to_csv(f'experiment_results/{model_name}.csv', index=False)


def update_results(experiment_results, index, metrics):
    for metric, value in metrics.items():
        experiment_results.at[index, metric] = value


def run_experiment_without_pca(num_data, cat_data, labels_true, methods_params, experiment_results_file):
    data = load_data(num_data, cat_data)
    experiment_results = load_experiment_results(experiment_results_file)
    model_counter = len(experiment_results)

    for method, param_grid in methods_params.items():
        for params in product(*param_grid.values()):
            param_dict = dict(zip(param_grid.keys(), params))
            model_name = f'model_{model_counter}'

            # Check if model with same parameters already exists in experiment results
            existing_model = experiment_results[
                (experiment_results['method'] == method) & (
                        experiment_results['parameters'] == str(param_dict)) & (
                        experiment_results['data'] == f'{num_data}_{cat_data}')]

            if existing_model.empty:
                try:
                    # Train model
                    print(f"'{method}' with params {param_dict} on data {num_data} {cat_data} not found. Training model...")
                    model = get_model_instance(method, param_dict)
                    model.fit(data)
                    print('Training completed.')

                    # Calculate metrics
                    labels_pred = model.labels_
                    metrics_values = calculate_metrics(labels_true, labels_pred)
                    additional_metrics = calculate_additional_metrics(data, labels_pred)

                    print(metrics_values)
                    print(additional_metrics)

                    if 2 <= additional_metrics['n_clusters'] <= 10:
                        # Save results
                        predictions_df = data.copy()
                        predictions_df.insert(loc=1, column='labels_pred', value=labels_pred)
                        predictions_df.insert(loc=2, column='labels_true', value=labels_true)
                        save_results(experiment_results, model_name, method, num_data, cat_data, param_dict,
                                     metrics_values, additional_metrics, predictions_df)

                        # Increment model counter
                        model_counter += 1

                except Exception as e:
                    # print(f"Error in '{method}' with params {param_dict}")
                    pass

            else:
                print(f"'{method}' with params {param_dict} already measured. Retrieving metrics...")
                try:
                    # Load predictions from file
                    predictions_file = f'experiment_results/{existing_model.iloc[0]["model_name"]}.csv'
                    predictions_df = pd.read_csv(predictions_file)
                    labels_pred = predictions_df['labels_pred'].values

                    # Calculate silhouette score
                    additional_metrics = calculate_additional_metrics(data, labels_pred)

                    # Update results
                    update_results(experiment_results, existing_model.index[0], additional_metrics)

                except Exception as e:
                    print(f"Error retrieving metrics for '{method}' with params {param_dict}: {e}")

            # Save experiment results to file
            experiment_results.to_csv(experiment_results_file, sep='\t', index=False)


def run_experiment(num_data, cat_data, labels_true, methods_params, experiment_results_file, pca_params_grid=None):
    data = load_data(num_data, cat_data)
    experiment_results = load_experiment_results(experiment_results_file)
    model_counter = len(experiment_results)

    if pca_params_grid:
        for pca_params in product(*pca_params_grid.values()):
            try:
                # Apply PCA
                pca = PCA(**dict(zip(pca_params_grid.keys(), pca_params)))
                transformed_data = pca.fit_transform(data.drop(columns=['Id']))
            except Exception as e:
                print(f"Error in PCA with params {pca_params} for data {num_data}, {cat_data}")
                continue
            for method, param_grid in methods_params.items():
                for params in product(*param_grid.values()):
                    param_dict = dict(zip(param_grid.keys(), params))
                    model_name = f'model_{model_counter}'

                    # Check if model with same parameters already exists in experiment results
                    # existing_model = experiment_results[
                    #     (experiment_results['method'] == method) & (
                    #             experiment_results['parameters'] == str(param_dict)) & (
                    #             experiment_results['data'] == f'{num_data}_{cat_data}') & (
                    #             experiment_results['pca_params'] == str(pca_params))]
                    # if not existing_model.empty:
                    if True:
                        try:
                            # Train model
                            print(
                                f"'{method}' with params {param_dict} on data {num_data} {cat_data} not found. Training model...")
                            model = get_model_instance(method, param_dict)
                            model.fit(transformed_data)
                            print('Training completed.')

                            # Calculate metrics
                            labels_pred = model.labels_
                            metrics_values = calculate_metrics(labels_true, labels_pred)
                            additional_metrics = calculate_additional_metrics(transformed_data, labels_pred)

                            print(metrics_values)
                            print(additional_metrics)

                            if 2 <= additional_metrics['n_clusters'] <= 10:
                                # Save results
                                predictions_df = transformed_data.copy()
                                predictions_df.insert(loc=1, column='labels_pred', value=labels_pred)
                                predictions_df.insert(loc=2, column='labels_true', value=labels_true)
                                save_results(experiment_results, model_name, method, num_data, cat_data, param_dict,
                                             metrics_values, additional_metrics, predictions_df, pca_params)

                                # Increment model counter
                                model_counter += 1
                        except Exception as e:
                            # print(f"Error in '{method}' with params {param_dict}")
                            pass

                    else:
                        print(f"'{method}' with params {param_dict} already measured. Retrieving metrics...")
                        try:
                            # Load predictions from file
                            predictions_file = f'experiment_results_pca/{existing_model.iloc[0]["model_name"]}.csv'
                            predictions_df = pd.read_csv(predictions_file)
                            labels_pred = predictions_df['labels_pred'].values

                            # Calculate silhouette score
                            additional_metrics = calculate_additional_metrics(data, labels_pred)

                            # Update results
                            update_results(experiment_results, existing_model.index[0], additional_metrics)

                        except Exception as e:
                            print(f"Error retrieving metrics for '{method}' with params {param_dict}: {e}")

                    # Save experiment results to file
                    experiment_results.to_csv(experiment_results_file, sep='\t', index=False)


y_with5 = pd.read_csv('preprocessed/y1.csv')
y_without5 = pd.read_csv('preprocessed/y2.csv')
data_without5 = product(['X2_num.csv', 'X4_num.csv', 'X6_num.csv', 'X8_num.csv'],
                        ['X4_cat.csv', 'X7_cat.csv', 'X8_cat.csv', 'X11_cat.csv'])
data_with5 = product(['X1_num.csv', 'X3_num.csv', 'X5_num.csv', 'X7_num.csv'],
                     ['X3_cat.csv', 'X5_cat.csv', 'X6_cat.csv', 'X9_cat.csv'])
methods_params = {
    'KMeans': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'random_state': [42]},
    'DBSCAN': {'eps': [0.1, 0.5, 10, 50, 100, 200, 500, 1000], 'min_samples': [5, 10, 15, 20, 50]},
    'HDBSCAN': {'min_cluster_size': [10, 20, 50, 100],
                'cluster_selection_epsilon': {0.5, 1.0, 10, 50, 100, 200, 500, 1000},
                'min_samples': [1, 2, 5, 10, 15, 20, 50, 100],
                'metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan', 'hamming']},
    'AffinityPropagation': {'damping': [0.5, 0.7, 0.9], 'random_state': [42]},
    'MeanShift': {'bandwidth': [None, 0.5, 1.0], 'bin_seeding': [True, False]},
    'SpectralClustering': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'random_state': [42], 'eigen_solver': ['arpack', 'lobpcg', 'amg'],
                           'assign_labels': ['kmeans', 'discretize', 'cluster_qr']},
    'AgglomerativeClustering': {'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10], 'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
                                'linkage': ['ward', 'complete', 'average', 'single']},
    'Birch': {'threshold': [0.1, 0.5, 1.0, 1.5, 5, 10], 'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]},
    'OPTICS': {'min_samples': [0.5, 2, 10, 20, 50, 100],
               'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']}
}
pca_params_grid = {'n_components': [0.95, 2, 5, 10, 20, 50, 100, 'mle'], 'random_state': [42]}
experiment_results_file = 'experiment_results.csv'

for comb in data_without5:
    # run_experiment(num_data=f'preprocessed/{comb[0]}', cat_data=f'preprocessed/{comb[1]}',
    #                labels_true=y_without5.iloc[:, 1].values, methods_params=methods_params,
    #                experiment_results_file=experiment_results_file, pca_params_grid=pca_params_grid)
    run_experiment_without_pca(num_data=f'preprocessed/{comb[0]}', cat_data=f'preprocessed/{comb[1]}',
                               labels_true=y_without5.iloc[:, 1].values, methods_params=methods_params,
                               experiment_results_file=experiment_results_file)

for comb in data_with5:
    # run_experiment(num_data=f'preprocessed/{comb[0]}', cat_data=f'preprocessed/{comb[1]}',
    #                labels_true=y_with5.iloc[:, 1].values, methods_params=methods_params,
    #                experiment_results_file=experiment_results_file, pca_params_grid=pca_params_grid)
    run_experiment_without_pca(num_data=f'preprocessed/{comb[0]}', cat_data=f'preprocessed/{comb[1]}',
                               labels_true=y_with5.iloc[:, 1].values, methods_params=methods_params,
                               experiment_results_file=experiment_results_file)

print('Done!')
