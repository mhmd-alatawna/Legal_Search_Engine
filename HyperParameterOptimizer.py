import shutil

from ResultsMeasurer import multi_thread_measure_recall_k
from Searchers import *
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    """
    The objective function that Optuna will try to MAXIMIZE.
    We define the intelligent boundaries for our 4 parameters here.
    """
    # micro_K = trial.suggest_float("micro_K", 0.5, 2.0)
    # micro_B = trial.suggest_float("micro_B", 0.2, 0.8)
    # macro_B = trial.suggest_float("macro_B", 0.7, 1.0)
    # threshold = trial.suggest_float("threshold", 0.3, 0.9)
    segment_multiplier = trial.suggest_int("segment_multiplier", 1, 20)
    segment_size = trial.suggest_int("segment_size", 1, 20)
    overlap_size = trial.suggest_int("overlap_size", 0, segment_size - 1)

    # recall = multi_thread_measure_recall_k(BM25_searcher(thresh=threshold,macro_B=macro_B,micro_B=micro_B,micro_K=micro_K,segment_multiplier=segment_multiplier) , query_dict , 10)
    InvertedIndexTantivy.generate_index(load_gzipped_json("Data/filtered_data_small.json.gz"),segment_size=segment_size,overlap_size=overlap_size)
    recall = multi_thread_measure_recall_k(BM25_searcher(segment_multiplier=segment_multiplier) , query_dict , 10)

    return recall

def tune_search_pipeline(n_trials=60):
    """
    Executes the Bayesian Optimization study.
    60 trials * 20 seconds = roughly 20 minutes of execution time.
    """
    print(f"Starting Bayesian Optimization for {n_trials} trials...")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Extract the winning combination
    best_params = study.best_params
    best_recall = study.best_value

    print("\n" + "=" * 40)
    print(f"OPTIMIZATION COMPLETE")
    print("=" * 40)
    print(f"Best Recall Achieved: {best_recall*100:.2f}%")
    print("Winning Hyperparameters:")
    # print(f"  Micro-K (Pyserini k1): {best_params['micro_K']:.4f}")
    # print(f"  Micro-B (Pyserini b):  {best_params['micro_B']:.4f}")
    # print(f"  Macro-B (Length Pen.): {best_params['macro_B']:.4f}")
    # print(f"  Threshold (Cut-off):   {best_params['threshold']:.4f}")
    print(f"  segment_multiplier:   {best_params['segment_multiplier']:.4f}")
    print(f"  segment_size:   {best_params['segment_size']:.4f}")
    print(f"  overlap_size:   {best_params['overlap_size']:.4f}")
    print("=" * 40)

    return best_params

# InvertedIndexPyserini.generate_index(load_gzipped_json("Data/filtered_data_small.json.gz"))

query_dict = load_gzipped_json("Data/query_dict.json.gz")
best_configuration = tune_search_pipeline(n_trials=10)