import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # from tqdm.autonotebook import tqdm
    from tqdm import tqdm

__all__ = ["tqdm"]
