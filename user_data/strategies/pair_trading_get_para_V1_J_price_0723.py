# --- Import Freqtrade Libraries ---
import itertools
import json
import logging
import os
from datetime import timedelta

import numpy as np
import pandas as pd
import statsmodels.api as sm
from freqtrade.strategy.interface import IStrategy
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


# discord该策略实时推送，欢迎来玩，https://discord.gg/3EABfUPxbQ
class pair_trading_get_para_V1_J_price_0723(IStrategy):
    can_short = True
    timeframe = "5m"
    startup_candle_count = 0
    process_only_new_candles = False
    # 交易参数
    minimal_roi = {"0": 1}
    stoploss = -0.99999
    use_custom_stoploss = False  # 启用动态止损

    def __init__(self, config):
        super().__init__(config)
        # 交易对白名单
        self.whitelist = config.get("exchange", {}).get("pair_whitelist", [])

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        current_pair = metadata["pair"]
        dataframe["zero"] = 0

        # Chỉ chạy logic tính toán một lần khi candle của cặp cuối cùng trong whitelist đến
        if current_pair != self.whitelist[-1]:
            return dataframe

        all_new_params = {}
        log_close_prices = {}

        # Tạo một DataFrame duy nhất chứa tất cả các giá đóng cửa cần thiết.
        close_series_list = []
        for pair in self.whitelist:
            # Lấy toàn bộ dataframe nhưng chỉ giữ lại cột 'close'
            # và chuyển đổi kiểu dữ liệu để tiết kiệm bộ nhớ
            close_data = self.dp.get_pair_dataframe(pair, self.timeframe)[
                "close"
            ].astype(np.float64)
            close_data.name = pair  # Đặt tên cho Series để sau này join
            close_series_list.append(close_data)

        # Gộp tất cả các Series giá đóng cửa vào một DataFrame duy nhất
        all_closes_df = pd.concat(close_series_list, axis=1)

        # Lấy giá đóng cửa cuối cùng cho log_close_prices
        for pair in self.whitelist:
            if not all_closes_df[pair].empty:
                log_close_prices[f"{pair}_log_close"] = all_closes_df[pair].iloc[-1]

        # --- TỐI ƯU 2: Tái cấu trúc vòng lặp để không tạo DataFrame mới liên tục ---
        from tqdm import tqdm

        for pair_y, pair_x in tqdm(list(itertools.permutations(self.whitelist, 2))):
            try:
                # Chọn 2 cột cần thiết từ DataFrame đã gộp, không cần tạo df_merged mới
                df_pair = all_closes_df[[pair_y, pair_x]].copy()
                df_pair.columns = ["y", "x"]  # Đổi tên cột để tương thích với fn_ecm
                df_pair.dropna(inplace=True)

                if len(df_pair) < 1000:
                    continue

                # Các hàm tính toán giữ nguyên
                (
                    c,
                    gamma,
                    z,
                    z_mean,
                    z_std,
                    z_cross_zero_count,
                    consistency_score,
                    half_life,
                    corr,
                ) = self.fn_ecm(df_pair, "y", "x")

                pvalue, adfstat = self.adf_test_on_residuals(z)

                pair_key = f"{pair_y}_{pair_x}"

                all_new_params[pair_key] = {
                    "pvalue": pvalue,
                    "adfstat": adfstat,
                    "gamma": gamma,
                    "c": c,
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "z_cross_zero_count": z_cross_zero_count,
                    "consistency_score": consistency_score,
                    "half_life": half_life,
                    "corr": corr,
                    "regression_y": pair_y,
                    "regression_x": pair_x,
                }
            except Exception as e:
                logger.error(f"分析配对 {pair_y}/{pair_x} 时出错: {e}")

        self.overwrite_run_config_params(all_new_params, log_close_prices)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        return dataframe

    def fn_ecm(self, df: pd.DataFrame, y_col: str, x_col: str):
        y = df[y_col]
        x = df[x_col]

        assert isinstance(y, pd.Series), "y 必须是 pd.Series 类型"
        assert isinstance(x, pd.Series), "x 必须是 pd.Series 类型"
        assert y.index.equals(x.index), "y 和 x 必须有相同的时间索引"

        # 皮尔森相关系数
        corr = y.corr(x)

        # 协整回归
        long_run_ols = sm.OLS(y, sm.add_constant(x))
        long_run_fit = long_run_ols.fit()
        c, gamma = long_run_fit.params

        z = long_run_fit.resid

        z_std = z.std()
        z_mean = z.mean()
        z_score = (z - z_mean) / z_std

        z_nonan = z.dropna()
        if len(z_nonan) > 1:
            z_lag = z_nonan.shift(1).dropna()
            z_ret = z_nonan.diff().dropna()
            z_lag = z_lag.loc[z_ret.index]
            X = sm.add_constant(z_lag)
            model = sm.OLS(z_ret, X)
            res = model.fit()
            if len(res.params) > 1 and res.params.iloc[1] != 0:
                half_life = -np.log(2) / res.params.iloc[1]
            else:
                half_life = np.nan
        else:
            half_life = np.nan

        crossings = (np.sign(z).shift(1) * np.sign(z)) < 0
        crossing_indices = crossings[crossings].index

        consistency_score = 0.0
        if len(crossing_indices) > 2:
            intervals = pd.Series(crossing_indices).diff().dropna()
            interval_mean = intervals.mean()
            interval_std = intervals.std()
            if interval_mean > 0 and interval_std > 0:
                consistency_score = 1000 / (interval_mean * interval_std)

        z_score_lag = z_score.shift(1)
        crossing_1 = ((z_score_lag < 1) & (z_score >= 1)).sum() + (
            (z_score_lag > 1) & (z_score <= 1)
        ).sum()
        crossing_minus1 = ((z_score_lag > -1) & (z_score <= -1)).sum() + (
            (z_score_lag < -1) & (z_score >= -1)
        ).sum()
        z_cross_zero_count = crossing_1 + crossing_minus1

        # if half_life < 1440 and half_life > 0:
        #     logger.info(f"Half-life for {y_col} vs {x_col}: {half_life}")
        #     logger.info(f"corr {y_col} vs {x_col}: {corr}")

        return (
            c,
            gamma,
            z_score,
            z_mean,
            z_std,
            z_cross_zero_count,
            consistency_score,
            half_life,
            corr,
        )

    def convert_numpy_types(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    def adf_test_on_residuals(self, z: pd.Series) -> tuple[float, float]:
        z_cleaned = z.dropna()
        if len(z_cleaned) < 20:
            return 1.0, 0.0
        adfstat, pvalue, _, _, _ = adfuller(z_cleaned, maxlag=1, autolag=None)
        return pvalue, adfstat

    def overwrite_run_config_params(self, new_params: dict, log_close_prices: dict):
        run_config_path = "configs/future_pair_pairing_run_0723.json"
        logger.info(f"准备将 {len(new_params)} 个配对的新参数写入到: {run_config_path}")

        try:
            # Tạo các dict từ new_params, không cần thay đổi logic này
            param_dicts = {
                f"{key}_dict": {}
                for key in [
                    "pvalue",
                    "adfstat",
                    "gamma",
                    "c",
                    "z_mean",
                    "z_std",
                    "z_cross_zero_count",
                    "consistency_score",
                    "half_life",
                    "corr",
                ]
            }

            for key, params in new_params.items():
                y = params["regression_y"]
                x = params["regression_x"]
                prefix = f"{y}_{x}"
                for param_name in param_dicts:
                    dict_key = param_name.replace("_dict", "")
                    param_dicts[param_name][f"{prefix}_{dict_key}"] = params[dict_key]

            run_config_data = {}
            if os.path.exists(run_config_path):
                with open(run_config_path, "r", encoding="utf-8") as f:
                    run_config_data = json.load(f)
            else:
                logger.warning(
                    f"目标配置文件 {run_config_path} 不存在，将创建一个新的。"
                )

            # Cập nhật run_config_data với các dict đã tạo
            run_config_data.update(param_dicts)
            run_config_data["log_close_dict"] = log_close_prices

            with open(run_config_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.convert_numpy_types(run_config_data),
                    f,
                    ensure_ascii=False,
                    indent=4,
                )
            logger.info("成功覆写 run 策略的配置文件！")

        except Exception as e:
            logger.error(f"覆写配置文件时出错: {e}")
