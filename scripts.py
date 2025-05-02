import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV


def get_window(df, start_time, end_time, key_col):
    """
    Extract a time- or sequence-based window from the DataFrame.

    Parameters:
        df (pd.DataFrame): The full DataFrame with order book data.
        start_time (str): Start timestamp or sequence ID.
        end_time (str): End timestamp or sequence ID.
        key_col (str): Column to use for slicing (either 'ts_event' or 'sequence').

    Returns:
        pd.DataFrame: A slice of df between start_time and end_time (inclusive).

    Raises:
        ValueError: If timestamps/sequence IDs are missing or incorrectly ordered.
    """
    start_idx = df.index[df[key_col] == start_time]
    end_idx = df.index[df[key_col] == end_time]

    if len(start_idx) == 0 or len(end_idx) == 0:
        raise ValueError("Start or end timestamp not found in the DataFrame.")

    start_idx, end_idx = start_idx[0], end_idx[0]
    if start_idx >= end_idx:
        raise ValueError("Start time must be before end time.")

    return df.iloc[start_idx:end_idx + 1]


def compute_level_ofi(curr, prev, level):
    """
    Compute the Order Flow Imbalance (OFI) for a specific LOB level.

    Parameters:
        curr (pd.Series): Current row of order book data.
        prev (pd.Series): Previous row of order book data.
        level (int): LOB level to compute OFI for (0-indexed).

    Returns:
        tuple:
            - ofi (float or None): OFI value at the given level (None if data is missing).
            - depth (float or None): Average depth at the level (None if data is missing).
    """
    try:
        bid_px = curr[f'bid_px_{level:02d}']
        ask_px = curr[f'ask_px_{level:02d}']
        bid_sz = curr[f'bid_sz_{level:02d}']
        ask_sz = curr[f'ask_sz_{level:02d}']

        bid_px_prev = prev[f'bid_px_{level:02d}']
        ask_px_prev = prev[f'ask_px_{level:02d}']
        bid_sz_prev = prev[f'bid_sz_{level:02d}']
        ask_sz_prev = prev[f'ask_sz_{level:02d}']

        delta_bid = bid_sz - bid_sz_prev
        delta_ask = ask_sz - ask_sz_prev

        bid_up = bid_px > bid_px_prev
        ask_down = ask_px < ask_px_prev

        return (delta_bid if bid_up else 0) - (delta_ask if ask_down else 0), 0.5 * (bid_sz + ask_sz)
    except KeyError:
        return None, None


def best_level_ofi(df, start_time, end_time, use_timestamp=True):
    """
    Compute the best-level (level 0) OFI over a time window.

    Parameters:
        df (pd.DataFrame): DataFrame containing order book data.
        start_time (str): Start timestamp or sequence ID.
        end_time (str): End timestamp or sequence ID.
        use_timestamp (bool): Whether to slice by 'ts_event' (True) or 'sequence' (False).

    Returns:
        float: Average best-level OFI across the window.
    """
    key_col = 'ts_event' if use_timestamp else 'sequence'
    window = get_window(df, start_time, end_time, key_col)
    if len(window) < 2:
        return 0

    ofi_sum = 0
    for i in range(1, len(window)):
        val, _ = compute_level_ofi(window.iloc[i], window.iloc[i - 1], 0)
        ofi_sum += val if val is not None else 0

    return ofi_sum / (len(window) - 1)


def multi_level_ofi(df, start_time, end_time, num_levels=10, use_timestamp=True):
    """
    Compute the normalized OFI across multiple LOB levels.

    Parameters:
        df (pd.DataFrame): DataFrame with order book data.
        start_time (str): Start timestamp or sequence ID.
        end_time (str): End timestamp or sequence ID.
        num_levels (int): Number of LOB levels to include.
        use_timestamp (bool): Whether to use 'ts_event' or 'sequence'.

    Returns:
        float: Depth-normalized multi-level OFI.
    """
    key_col = 'ts_event' if use_timestamp else 'sequence'
    window = get_window(df, start_time, end_time, key_col)
    if len(window) < 2:
        return 0

    total_ofi, total_depth, count = 0, 0, 0
    for i in range(1, len(window)):
        for level in range(num_levels):
            val, depth = compute_level_ofi(window.iloc[i], window.iloc[i - 1], level)
            if val is None:
                break
            total_ofi += val
            total_depth += depth
            count += 1

    return total_ofi / (total_depth / count) if count > 0 and total_depth != 0 else 0


def integrated_ofi(df, start_time, end_time, num_levels=10, use_timestamp=True):
    """
    Compute PCA-integrated OFI by aggregating multi-level OFIs via the first principal component.

    Parameters:
        df (pd.DataFrame): Order book DataFrame.
        start_time (str): Start timestamp or sequence ID.
        end_time (str): End timestamp or sequence ID.
        num_levels (int): Number of LOB levels to include in PCA.
        use_timestamp (bool): Whether to slice by 'ts_event' or 'sequence'.

    Returns:
        float: Average projected OFI along the first PCA component.
    """
    key_col = 'ts_event' if use_timestamp else 'sequence'
    window = get_window(df, start_time, end_time, key_col)
    if len(window) < 2:
        return 0

    ofi_vectors = []
    for i in range(1, len(window)):
        level_ofis = []
        for level in range(num_levels):
            val, _ = compute_level_ofi(window.iloc[i], window.iloc[i - 1], level)
            if val is None:
                break
            level_ofis.append(val)
        if len(level_ofis) == num_levels:
            ofi_vectors.append(level_ofis)

    if len(ofi_vectors) < 2:
        return 0

    ofi_matrix = np.array(ofi_vectors)
    pca = PCA(n_components=1)
    pca.fit(ofi_matrix)
    w1 = pca.components_[0]
    w1 /= np.sum(np.abs(w1)) or 1  # L1 norm

    integrated = ofi_matrix @ w1
    return np.mean(integrated)


def cross_asset_ofi(df, start_time, end_time, use_timestamp=True, method='integrated', instrument_ids=None, num_levels=10, lasso_weights=None):
    """
    Compute cross-asset OFI using specified OFI method, optionally using LASSO-learned weights.

    Parameters:
        df (pd.DataFrame): Order book data.
        start_time (str): Start time of window.
        end_time (str): End time of window.
        use_timestamp (bool): Whether to use timestamps or sequence IDs.
        method (str): OFI method ('best', 'multi', or 'integrated').
        instrument_ids (list): Instruments to include. Defaults to all.
        num_levels (int): Number of LOB levels for multi/integrated methods.
        lasso_weights (np.array): Optional weights from LASSO model.

    Returns:
        float: Weighted or average cross-asset OFI.
    """
    key_col = 'ts_event' if use_timestamp else 'sequence'
    if instrument_ids is None:
        instrument_ids = df['instrument_id'].unique()

    ofi_methods = {
        'best': best_level_ofi,
        'multi': multi_level_ofi,
        'integrated': integrated_ofi,
    }

    if method not in ofi_methods:
        raise ValueError(f"Unknown OFI method: {method}")

    results = []
    for instrument in instrument_ids:
        asset_df = df[df['instrument_id'] == instrument]
        if len(asset_df) < 2:
            results.append(0)
            continue
        try:
            if method == 'best':
                val = ofi_methods[method](asset_df, start_time, end_time, use_timestamp=use_timestamp)
            else:
                val = ofi_methods[method](asset_df, start_time, end_time, use_timestamp=use_timestamp, num_levels=num_levels)
            results.append(val)
        except Exception as e:
            print(f"Error processing {instrument} {method}: {e}")
            results.append(0)

    results = np.array(results)
    if lasso_weights is not None and len(lasso_weights) == len(results):
        return np.dot(results, lasso_weights)
    else:
        return np.mean(results)


def report_ofi_metrics(df, start_time, end_time, use_timestamp=True, num_levels=10, lasso_weights=None):
    """
    Computes and prints all OFI metrics (best, multi-level, integrated, and cross-asset variants).

    Parameters:
        df (pd.DataFrame): Full order book DataFrame.
        start_time (str): Start of window.
        end_time (str): End of window.
        use_timestamp (bool): Whether to use timestamps.
        num_levels (int): Number of LOB levels.
        lasso_weights (np.array): Optional LASSO weights for cross-asset OFI.
    """
    print(f"\nOFI Metrics Report")
    print(f"Start Time: {start_time}")
    print(f"End Time:   {end_time}")
    print(f"Using {'timestamp' if use_timestamp else 'sequence'} as key\n")

    try:
        best = best_level_ofi(df, start_time, end_time, use_timestamp)
        print(f"Best-Level OFI:       {best:.6f}")
    except Exception as e:
        print(f"Best-Level OFI:       Error - {e}")

    try:
        multi = multi_level_ofi(df, start_time, end_time, num_levels, use_timestamp)
        print(f"Multi-Level OFI:      {multi:.6f}")
    except Exception as e:
        print(f"Multi-Level OFI:      Error - {e}")

    try:
        integrated = integrated_ofi(df, start_time, end_time, num_levels, use_timestamp)
        print(f"Integrated OFI (PCA): {integrated:.6f}")
    except Exception as e:
        print(f"Integrated OFI:       Error - {e}")

    print("\nCross-Asset Averages:")
    for method in ['best', 'multi', 'integrated']:
        try:
            cross = cross_asset_ofi(df, start_time, end_time, use_timestamp, method, num_levels=num_levels, lasso_weights=lasso_weights)
            print(f"- {method.title()}-Level OFI:   {cross:.6f}")
        except Exception as e:
            print(f"- {method.title()}-Level OFI:   Error - {e}")


if __name__ == '__main__':
    df = pd.read_csv('first_25000_rows.csv')
    start_time = '2024-10-21T11:54:29.221064336Z'
    end_time = '2024-10-21T11:55:30.509413271Z'

    # Placeholder weights (you should get these from a trained LASSO model)
    instrument_ids = df['instrument_id'].unique()
    dummy_weights = np.ones(len(instrument_ids)) / len(instrument_ids)

    report_ofi_metrics(df, start_time, end_time, use_timestamp=True, num_levels=10, lasso_weights=dummy_weights)
