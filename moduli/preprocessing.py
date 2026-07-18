
"""
Shared preprocessing for pixel-level and superpixel-level clustering.

Core idea:
    Every clustering algorithm receives the same standardized node representation:

        X_model      : features for model nodes
        coords_model : coordinates for model nodes
        weights      : area/size weight per node
        neighbors    : kNN graph over model nodes
        node_col     : dataframe column mapping each original pixel to a model node

For mode="pixels":
    one model node = one original pixel

For mode="superpixels":
    one model node = one spatial superpixel
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


DEFAULT_EXCLUDE_COLS = ("row", "col", "easting", "northing", "superpixel", "superpixel_compact", "pixel_id")


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Iterable[str] = DEFAULT_EXCLUDE_COLS,
) -> list[str]:
    exclude = set(exclude_cols)
    return [c for c in df.columns if c not in exclude]

def build_temp_rain_pca_features(
    frame,
    temp_cols,
    rain_cols,
    n_temp=2,
    n_rain=2,
    scale_method="minmax",
    final_scale=True,
):
    """
    Apply PCA separately to temperature and rainfall variables.

    Can be used:
        - before superpixelization (frame=df)
        - after superpixelization (frame=X_super_df)

    Returns:
        X_pca
        feature_names
        info
    """

    T = frame[temp_cols].to_numpy(dtype=np.float64)
    P = frame[rain_cols].to_numpy(dtype=np.float64)

    if scale_method == "minmax":
        temp_scaler = MinMaxScaler()
        rain_scaler = MinMaxScaler()
    elif scale_method == "standard":
        temp_scaler = StandardScaler()
        rain_scaler = StandardScaler()
    else:
        raise ValueError(
            "scale_method must be 'minmax' or 'standard'"
        )

    T_scaled = temp_scaler.fit_transform(T)
    P_scaled = rain_scaler.fit_transform(P)

    pca_temp = PCA()
    temp_pcs_all = pca_temp.fit_transform(T_scaled)

    pca_rain = PCA()
    rain_pcs_all = pca_rain.fit_transform(P_scaled)

    temp_pc = temp_pcs_all[:, :n_temp]
    rain_pc = rain_pcs_all[:, :n_rain]

    X_pca = np.hstack([temp_pc, rain_pc])

    if final_scale:
        X_pca = MinMaxScaler().fit_transform(X_pca)

    feature_names = (
        [f"temp_pc{i+1}" for i in range(n_temp)]
        + [f"rain_pc{i+1}" for i in range(n_rain)]
    )

    info = {
        "pca_temp": pca_temp,
        "pca_rain": pca_rain,
        "temp_explained_variance": pca_temp.explained_variance_ratio_,
        "rain_explained_variance": pca_rain.explained_variance_ratio_,
    }

    return X_pca, feature_names, info


def build_raw_features(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    exclude_cols: Iterable[str] = DEFAULT_EXCLUDE_COLS,
) -> tuple[np.ndarray, list[str]]:
    if feature_cols is None:
        feature_cols = get_feature_columns(df, exclude_cols=exclude_cols)

    X = df[feature_cols].to_numpy(dtype=np.float64)
    return X, list(feature_cols)


def find_temperature_rainfall_columns(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    temp_keyword: str = "temp",
    rain_keyword: str = "padavine",
    exclude_cols: Iterable[str] = DEFAULT_EXCLUDE_COLS,
) -> tuple[list[str], list[str]]:
    if feature_cols is None:
        feature_cols = get_feature_columns(df, exclude_cols=exclude_cols)

    temp_cols = [c for c in feature_cols if temp_keyword.lower() in c.lower()]
    rain_cols = [c for c in feature_cols if rain_keyword.lower() in c.lower()]

    return temp_cols, rain_cols


def build_koppen_features(
    df: pd.DataFrame,
    temp_cols: Optional[list[str]] = None,
    rain_cols: Optional[list[str]] = None,
    temp_keyword: str = "temp",
    rain_keyword: str = "padavine",
    include_std: bool = False,
    include_dryness_index: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Build simple Koppen-inspired features at pixel level.

    Important:
        Build these before superpixel aggregation.
    """
    if temp_cols is None or rain_cols is None:
        feature_cols = get_feature_columns(df)
        detected_temp_cols, detected_rain_cols = find_temperature_rainfall_columns(
            df,
            feature_cols=feature_cols,
            temp_keyword=temp_keyword,
            rain_keyword=rain_keyword,
        )
        if temp_cols is None:
            temp_cols = detected_temp_cols
        if rain_cols is None:
            rain_cols = detected_rain_cols

    if not temp_cols:
        raise ValueError("No temperature columns found. Pass temp_cols explicitly.")
    if not rain_cols:
        raise ValueError("No rainfall columns found. Pass rain_cols explicitly.")

    T = df[temp_cols].to_numpy(dtype=np.float64)
    P = df[rain_cols].to_numpy(dtype=np.float64)

    temp_mean = T.mean(axis=1)
    temp_min = T.min(axis=1)
    temp_max = T.max(axis=1)
    temp_range = temp_max - temp_min

    precip_mean = P.mean(axis=1)
    precip_min = P.min(axis=1)
    precip_max = P.max(axis=1)
    precip_total = P.sum(axis=1)
    precip_range = precip_max - precip_min

    arrays = [
        temp_mean,
        temp_min,
        temp_max,
        temp_range,
        precip_mean,
        precip_min,
        precip_max,
        precip_total,
        precip_range,
    ]

    names = [
        "temp_mean",
        "temp_min",
        "temp_max",
        "temp_range",
        "precip_mean",
        "precip_min",
        "precip_max",
        "precip_total",
        "precip_range",
    ]

    if include_std:
        arrays.extend([T.std(axis=1), P.std(axis=1)])
        names.extend(["temp_std", "precip_std"])

    if include_dryness_index:
        dryness_index = precip_total / (temp_mean + 10.0 + 1e-12)
        arrays.append(dryness_index)
        names.append("dryness_index")

    return np.column_stack(arrays), names


def create_superpixels(
    df: pd.DataFrame,
    target_size: int = 100,
    row_col: tuple[str, str] = ("row", "col"),
    batch_size: int = 4096,
    max_iter: int = 30,
    n_init: int = 1,
    random_state: int = 0,
) -> tuple[np.ndarray, MiniBatchKMeans]:
    coords = df[list(row_col)].to_numpy(dtype=np.float64)

    n_pixels = len(df)
    n_superpixels = max(1, n_pixels // target_size)

    coords_scaled = StandardScaler().fit_transform(coords)

    model = MiniBatchKMeans(
        n_clusters=n_superpixels,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=n_init,
        random_state=random_state,
    )

    labels = model.fit_predict(coords_scaled)
    return labels, model


def aggregate_to_superpixels(
    df: pd.DataFrame,
    X_features: np.ndarray,
    superpixel_labels: np.ndarray,
    row_col: tuple[str, str] = ("row", "col"),
) -> dict:
    X_features = np.asarray(X_features, dtype=np.float64)
    labels = np.asarray(superpixel_labels, dtype=np.int64)
    coords = df[list(row_col)].to_numpy(dtype=np.float64)

    if len(X_features) != len(df):
        raise ValueError("X_features must have the same number of rows as df.")
    if len(labels) != len(df):
        raise ValueError("superpixel_labels must have the same length as df.")

    n_sp = int(labels.max()) + 1
    n_feat = X_features.shape[1]

    X_super_raw = np.zeros((n_sp, n_feat), dtype=np.float64)
    coord_super_raw = np.zeros((n_sp, 2), dtype=np.float64)
    counts = np.zeros(n_sp, dtype=np.float64)

    for i in range(len(df)):
        k = labels[i]
        X_super_raw[k] += X_features[i]
        coord_super_raw[k] += coords[i]
        counts[k] += 1

    mask = counts > 0
    valid_superpixel_ids = np.flatnonzero(mask)

    X_super = X_super_raw[mask] / counts[mask, None]
    coord_super = coord_super_raw[mask] / counts[mask, None]
    sp_sizes = counts[mask]

    old_to_new = {int(old): int(new) for new, old in enumerate(valid_superpixel_ids)}
    compact_labels = np.array([old_to_new[int(k)] for k in labels], dtype=np.int64)

    return {
        "X_model": X_super,
        "coords_model": coord_super,
        "weights": sp_sizes,
        "valid_superpixel_ids": valid_superpixel_ids,
        "old_to_new_superpixel": old_to_new,
        "node_ids_for_pixels": compact_labels,
    }


def build_neighbors(
    coords_model: np.ndarray,
    k_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    coords_model = np.asarray(coords_model, dtype=np.float64)

    if len(coords_model) < 2:
        raise ValueError("Need at least 2 nodes to build neighbors.")

    k = min(k_neighbors + 1, len(coords_model))

    nbrs = NearestNeighbors(n_neighbors=k).fit(coords_model)
    distances, indices = nbrs.kneighbors(coords_model)

    return indices[:, 1:], distances[:, 1:]


def prepare_model_data(
    df: pd.DataFrame,
    X_features: np.ndarray,
    feature_names: Optional[list[str]] = None,
    mode: str = "superpixels",
    target_size: int = 100,
    k_neighbors: int = 8,
    scale_final: bool = True,
    scaler: Optional[object] = None,
    row_col: tuple[str, str] = ("row", "col"),
    random_state: int = 0,
) -> dict:
    """
    Prepare model inputs for both ACO variants.

    mode:
        "pixels"      -> one node per original pixel
        "superpixels" -> one node per spatial superpixel
    """
    X_features = np.asarray(X_features, dtype=np.float64)

    if len(X_features) != len(df):
        raise ValueError("X_features must have the same number of rows as df.")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_features.shape[1])]

    df_out = df.copy()

    if mode == "pixels":
        coords_model = df_out[list(row_col)].to_numpy(dtype=np.float64)
        X_model = X_features.copy()
        weights = np.ones(len(df_out), dtype=np.float64)

        df_out["pixel_id"] = np.arange(len(df_out), dtype=np.int64)
        node_col = "pixel_id"

        superpixel_model = None
        superpixel_labels = None

    elif mode == "superpixels":
        superpixel_labels, superpixel_model = create_superpixels(
            df_out,
            target_size=target_size,
            row_col=row_col,
            random_state=random_state,
        )

        aggregation = aggregate_to_superpixels(
            df_out,
            X_features=X_features,
            superpixel_labels=superpixel_labels,
            row_col=row_col,
        )

        X_model = aggregation["X_model"]
        coords_model = aggregation["coords_model"]
        weights = aggregation["weights"]

        df_out["superpixel"] = superpixel_labels
        df_out["superpixel_compact"] = aggregation["node_ids_for_pixels"]
        node_col = "superpixel_compact"

    else:
        raise ValueError("mode must be either 'pixels' or 'superpixels'.")

    if scale_final:
        if scaler is None:
            scaler = StandardScaler()
        X_final = scaler.fit_transform(X_model)
    else:
        X_final = X_model
        scaler = None

    neighbors, neighbor_dists = build_neighbors(coords_model, k_neighbors=k_neighbors)

    return {
        "mode": mode,
        "df": df_out,
        "feature_names": feature_names,
        "X_model_unscaled": X_model,
        "X_model": X_final,
        "coords_model": coords_model,
        "weights": weights,
        "neighbors": neighbors,
        "neighbor_dists": neighbor_dists,
        "node_col": node_col,
        "scaler": scaler,
        "superpixel_model": superpixel_model,
        "superpixel_labels": superpixel_labels,
    }
