#scripts/evaluate.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型評估腳本 - 電池老化預測系統
獨立的模型評估工具，支持詳細分析、跨溫度比較、充放電分析和增強可視化。
"""

import sys
import time
import json
import argparse
import datetime
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(project_root))

# 導入項目模組
from config.base_config import config
from core.logging import setup_logger, LoggingTimer, log_memory_usage
from core.memory import memory_manager, memory_cleanup
from data.data_provider import DataLoader
from utils.visualization import VisualizationManager

# 嘗試導入多溫度訓練器的充放電回調
try:
    from trainers.multitemp_trainer import ChargeAwareCallback
    HAS_CHARGE_AWARE = True
except ImportError:
    HAS_CHARGE_AWARE = False
    print("警告: 無法導入充放電感知模組，將不支持充放電分析功能")

# 設置日誌
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logger = setup_logger("evaluate", log_file=f"evaluate_{timestamp}.log")

# ANSI顏色代碼
class Colors:
    HEADER = '\033[95m\033[1m'
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m\033[1m'
    ENDC = '\033[0m'
    
    @staticmethod
    def colored(text, color_code):
        return f"{color_code}{text}{Colors.ENDC}"
    
    @staticmethod
    def header(text):  return Colors.colored(text, Colors.HEADER)
    @staticmethod
    def info(text):    return Colors.colored(text, Colors.INFO)
    @staticmethod
    def success(text): return Colors.colored(text, Colors.SUCCESS)
    @staticmethod
    def warning(text): return Colors.colored(text, Colors.WARNING)
    @staticmethod
    def error(text):   return Colors.colored(text, Colors.ERROR)

def log_step(step_name: str):
    """記錄處理步驟的裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"執行 {step_name}...")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{step_name} 完成，耗時 {elapsed:.2f} 秒")
                return result
            except Exception as e:
                logger.error(f"{step_name} 失敗: {e}")
                raise
        return wrapper
    return decorator

class ModelFactory:
    """模型工廠，用於加載不同類型的模型"""
    
    @staticmethod
    def load_model(experiment_name: str, model_type: str, temp: str, checkpoint_dir: str):
        """加載訓練好的模型"""
        # 檢查點路徑
        checkpoint_path = Path(checkpoint_dir) / f"{experiment_name}_{temp}_best.h5"
        
        if not checkpoint_path.exists():
            logger.error(f"找不到檢查點: {checkpoint_path}")
            raise FileNotFoundError(f"找不到檢查點: {checkpoint_path}")
        
        # 根據模型類型加載
        model = None
        model_creators = {
            "baseline": "create_baseline_model",
            "cnn_gru": "create_cnn_gru_model",
            "pinn": "create_pinn_model",
            "gan": "create_gan_model"
        }
        
        # 嘗試從對應模組加載模型創建函數
        try:
            if model_type in ["baseline", "cnn_gru"]:
                from models.baseline import create_baseline_model, create_cnn_gru_model
                model = create_baseline_model() if model_type == "baseline" else create_cnn_gru_model()
            elif model_type == "pinn":
                from models.pinn import create_pinn_model
                model = create_pinn_model()
            elif model_type == "gan":
                from models.gan import create_gan_model
                model = create_gan_model()
        except ImportError:
            logger.warning(f"無法導入{model_type}模型模組，嘗試通用加載方式")
        
        # 加載模型或權重
        try:
            if model is None:
                # 通用加載方式
                model_path = str(checkpoint_path).replace("_best.h5", "")
                logger.info(f"使用通用方式加載模型: {model_path}")
                model = tf.keras.models.load_model(model_path)
            else:
                # 編譯模型並加載權重
                model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
                logger.info(f"加載模型權重: {checkpoint_path}")
                model.load_weights(str(checkpoint_path))
                
            return model
        except Exception as e:
            logger.error(f"加載模型失敗: {e}")
            raise

class ModelEvaluator:
    """模型評估器，提供詳細評估功能"""
    
    def __init__(self, config_override: Dict[str, Any]):
        """初始化模型評估器"""
        self.config_override = config_override
        
        # 初始化數據加載器
        self.data_loader = DataLoader({
            'batch_size': config_override.get('batch_size', 32),
            'data_dir': config_override.get('system.data_dir', config.get('system.data_dir')),
            'tfrecord_dir': config_override.get('system.tfrecord_dir', config.get('system.tfrecord_dir'))
        })
        
        # 初始化視覺化管理器
        self.viz_manager = VisualizationManager(
            save_dir=config_override.get('figures_dir', config.get('system.figures_dir')),
            dpi=config_override.get('dpi', 300)
        )
        
        # 創建線程池 (如果啟用了并行處理)
        max_workers = min(16, (os.cpu_count() or 4) * 2) if config_override.get('parallel', True) else 1
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    @log_step("評估模型性能")
    def evaluate_model(self, model, dataset, scaler_params=None, metrics=None):
        """評估模型性能"""
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'r2']
        
        results = {}
        
        try:
            # 標準 Keras 評估
            eval_results = model.evaluate(dataset, verbose=1)
            
            # 獲取度量名稱
            if isinstance(eval_results, list):
                metric_names = model.metrics_names
                for name, value in zip(metric_names, eval_results):
                    results[name] = float(value)
            else:
                results["loss"] = float(eval_results)
            
            # 收集預測和真實值
            def process_batch(batch):
                x_batch, y_batch = batch
                y_pred = model.predict(x_batch, verbose=0)
                return y_batch.numpy(), y_pred
            
            # 並行處理批次
            batch_results = []
            with self.executor as executor:
                futures = [executor.submit(process_batch, batch) for batch in dataset]
                batch_results = [future.result() for future in as_completed(futures)]
            
            # 合併結果
            y_true_list, y_pred_list = zip(*batch_results)
            y_true = np.vstack(y_true_list)
            y_pred = np.vstack(y_pred_list)
            
            # 反轉標準化
            if scaler_params is not None and len(y_true.shape) == 3 and y_true.shape[2] >= 2:
                # 提取標準化參數
                fdcr_mean = scaler_params.get("y_fdcr_mean", 0)
                fdcr_scale = scaler_params.get("y_fdcr_scale", 1)
                rsoc_mean = scaler_params.get("y_rsoc_mean", 0)
                rsoc_scale = scaler_params.get("y_rsoc_scale", 1)
                
                # 反轉標準化
                y_true[:, :, 0] = y_true[:, :, 0] * fdcr_scale + fdcr_mean
                y_pred[:, :, 0] = y_pred[:, :, 0] * fdcr_scale + fdcr_mean
                
                if y_true.shape[2] > 1:
                    y_true[:, :, 1] = y_true[:, :, 1] * rsoc_scale + rsoc_mean
                    y_pred[:, :, 1] = y_pred[:, :, 1] * rsoc_scale + rsoc_mean
            
            # 計算詳細指標
            feature_names = ["FDCR", "RSOC"]
            for i in range(min(y_true.shape[2], len(feature_names))):
                feature_name = feature_names[i]
                y_true_feature = y_true[:, :, i].flatten()
                y_pred_feature = y_pred[:, :, i].flatten()
                
                # 計算指標
                results[f"{feature_name}_MAE"] = float(mean_absolute_error(y_true_feature, y_pred_feature))
                results[f"{feature_name}_MSE"] = float(mean_squared_error(y_true_feature, y_pred_feature))
                results[f"{feature_name}_RMSE"] = float(np.sqrt(mean_squared_error(y_true_feature, y_pred_feature)))
                results[f"{feature_name}_R2"] = float(r2_score(y_true_feature, y_pred_feature))
            
            # 保存樣本
            results["samples"] = {
                "y_true": y_true[:5].tolist(),  # 前5個樣本
                "y_pred": y_pred[:5].tolist()
            }
            
            logger.info(f"評估完成，結果: {results}")
            return results, (y_true, y_pred)
        
        except Exception as e:
            logger.error(f"評估過程中錯誤: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}, (None, None)
    
    @log_step("評估特定溫度模型")
    def evaluate_temp(self, experiment_name, model_type, temp, checkpoint_dir, data_prefix="source"):
        """評估特定溫度的模型"""
        try:
            # 加載模型
            model = ModelFactory.load_model(experiment_name, model_type, temp, checkpoint_dir)
            
            # 加載測試數據集
            test_dataset = self.data_loader.get_dataset(f"{data_prefix}_{temp}", "test")
            
            # 嘗試加載標準化參數
            try:
                scaler_params = self.data_loader.load_scaler_params(f"{data_prefix}_{temp}")
            except Exception:
                logger.warning(f"無法加載 {temp} 的標準化參數，使用未標準化評估")
                scaler_params = None
            
            # 評估模型
            return self.evaluate_model(model, test_dataset, scaler_params)
            
        except FileNotFoundError as e:
            logger.warning(f"找不到 {temp} 溫度的模型或數據: {e}")
            return {'error': f'file_not_found: {str(e)}'}, (None, None)
        except Exception as e:
            logger.error(f"評估 {temp} 溫度模型時出錯: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': f'evaluation_failed: {str(e)}'}, (None, None)
    
    @log_step("評估充放電性能")
    def evaluate_charge_discharge(self, model, test_dataset, temp, 
                                 current_feature_idx=2, threshold=0.0):
        """評估模型在充電和放電數據上的性能"""
        if not HAS_CHARGE_AWARE:
            logger.warning("充放電感知模組不可用，無法進行充放電分析")
            return {"error": "charge_aware_module_not_available"}
        
        try:
            # 分離充電和放電數據
            x_all = []
            y_all = []
            charge_indices = []
            discharge_indices = []
            
            sample_idx = 0
            for x_batch, y_batch in test_dataset:
                # 將批次數據添加到列表
                x_batch_np = x_batch.numpy()
                y_batch_np = y_batch.numpy()
                
                x_all.append(x_batch_np)
                y_all.append(y_batch_np)
                
                # 提取電流值
                current_values = x_batch_np[:, :, current_feature_idx]
                
                # 識別充電和放電樣本
                batch_size = x_batch_np.shape[0]
                for i in range(batch_size):
                    mean_current = np.mean(current_values[i])
                    if mean_current > threshold:
                        charge_indices.append(sample_idx)
                    else:
                        discharge_indices.append(sample_idx)
                    sample_idx += 1
            
            # 合併所有批次數據
            if x_all and y_all:
                x_all = np.vstack(x_all)
                y_all = np.vstack(y_all)
            else:
                logger.warning("評估數據集為空")
                return {"error": "empty_dataset"}
            
            # 初始化評估結果
            eval_results = {
                'overall': {},
                'charge': {'count': len(charge_indices)},
                'discharge': {'count': len(discharge_indices)},
                'comparative': {}
            }
            
            # 評估整體性能
            logger.info(f"評估整體性能 (樣本數: {len(x_all)})")
            overall_metrics = model.evaluate(x_all, y_all, verbose=0)
            for i, name in enumerate(model.metrics_names):
                eval_results['overall'][name] = float(overall_metrics[i])
            
            # 評估充電數據性能
            if charge_indices:
                logger.info(f"評估充電性能 (樣本數: {len(charge_indices)})")
                x_charge = x_all[charge_indices]
                y_charge = y_all[charge_indices]
                charge_metrics = model.evaluate(x_charge, y_charge, verbose=0)
                for i, name in enumerate(model.metrics_names):
                    eval_results['charge'][name] = float(charge_metrics[i])
            
            # 評估放電數據性能
            if discharge_indices:
                logger.info(f"評估放電性能 (樣本數: {len(discharge_indices)})")
                x_discharge = x_all[discharge_indices]
                y_discharge = y_all[discharge_indices]
                discharge_metrics = model.evaluate(x_discharge, y_discharge, verbose=0)
                for i, name in enumerate(model.metrics_names):
                    eval_results['discharge'][name] = float(discharge_metrics[i])
            
            # 計算比較指標
            if charge_indices and discharge_indices:
                for name in eval_results['charge']:
                    if (name in eval_results['discharge'] and name != 'count' and 
                        isinstance(eval_results['charge'][name], (int, float)) and
                        isinstance(eval_results['discharge'][name], (int, float))):
                        ratio = eval_results['discharge'][name] / eval_results['charge'][name]
                        eval_results['comparative'][f'{name}_ratio'] = float(ratio)
            
            # 創建充放電可視化
            self._create_charge_discharge_visualization(eval_results, temp)
            
            return eval_results
            
        except Exception as e:
            logger.error(f"充放電評估出錯: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _create_charge_discharge_visualization(self, eval_results, temp):
        """創建充放電評估可視化"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 創建保存路徑
            figures_dir = self.config_override.get('figures_dir', config.get('system.figures_dir'))
            Path(figures_dir).mkdir(parents=True, exist_ok=True)
            
            # 獲取指標
            metrics = ['loss', 'mae', 'rmse']
            available_metrics = [m for m in metrics if 
                               m in eval_results['charge'] and 
                               m in eval_results['discharge']]
            
            if not available_metrics:
                logger.warning("沒有可用的充放電指標進行可視化")
                return
            
            # 創建二維條形圖
            plt.figure(figsize=(12, 6))
            
            # 準備數據
            x = np.arange(len(available_metrics))
            charge_values = [eval_results['charge'][m] for m in available_metrics]
            discharge_values = [eval_results['discharge'][m] for m in available_metrics]
            
            # 繪製條形圖
            width = 0.35
            plt.bar(x - width/2, charge_values, width, label='充電', color='#66b3ff')
            plt.bar(x + width/2, discharge_values, width, label='放電', color='#ff9999')
            
            # 添加標籤和圖例
            plt.xlabel('評估指標')
            plt.ylabel('數值')
            plt.title(f'{temp} 溫度下的充放電性能比較')
            plt.xticks(x, available_metrics)
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
            
            # 添加數值標籤
            for i, (c, d) in enumerate(zip(charge_values, discharge_values)):
                plt.text(i - width/2, c, f'{c:.4f}', ha='center', va='bottom', fontsize=8)
                plt.text(i + width/2, d, f'{d:.4f}', ha='center', va='bottom', fontsize=8)
                
                # 添加比率標籤
                metric = available_metrics[i]
                if f'{metric}_ratio' in eval_results['comparative']:
                    ratio = eval_results['comparative'][f'{metric}_ratio']
                    plt.text(i, max(c, d) * 1.05, f'比率: {ratio:.2f}', ha='center', fontsize=9)
            
            # 保存圖表
            experiment_name = self.config_override.get('experiment_name', 'unknown')
            cd_path = Path(figures_dir) / f"{experiment_name}_{temp}_cd_comparison.png"
            plt.savefig(cd_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"充放電比較圖已保存: {cd_path}")
            
            # 創建餅圖顯示樣本分佈
            plt.figure(figsize=(8, 8))
            
            charge_count = eval_results['charge']['count']
            discharge_count = eval_results['discharge']['count']
            total = charge_count + discharge_count
            
            labels = [f'充電 ({charge_count}, {charge_count/total*100:.1f}%)', 
                     f'放電 ({discharge_count}, {discharge_count/total*100:.1f}%)']
            sizes = [charge_count, discharge_count]
            colors = ['#66b3ff', '#ff9999']
            explode = (0.1, 0)
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                   shadow=True, startangle=90)
            plt.axis('equal')
            plt.title(f'{temp} 溫度下的樣本分佈')
            
            # 保存餅圖
            pie_path = Path(figures_dir) / f"{experiment_name}_{temp}_sample_dist.png"
            plt.savefig(pie_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"樣本分佈圖已保存: {pie_path}")
            
        except Exception as e:
            logger.warning(f"創建充放電可視化失敗: {e}")
    
    @log_step("評估所有溫度模型")
    def evaluate_all_temps(self, experiment_name, model_type, temps, checkpoint_dir, data_prefix="source"):
        """評估所有溫度的模型"""
        results = {}
        cd_results = {}
        prediction_samples = {}
        
        # 並行評估所有溫度
        def evaluate_temp_parallel(temp):
            logger.info(f"評估 {temp} 溫度模型...")
            
            # 加載模型
            try:
                model = ModelFactory.load_model(experiment_name, model_type, temp, checkpoint_dir)
                
                # 標準評估
                metrics, data = self.evaluate_temp(
                    experiment_name, model_type, temp, checkpoint_dir, data_prefix
                )
                
                # 充放電評估 (如果啟用)
                charge_discharge_metrics = None
                if self.config_override.get('charge_discharge', False) and HAS_CHARGE_AWARE:
                    # 加載測試數據集
                    test_dataset = self.data_loader.get_dataset(f"{data_prefix}_{temp}", "test")
                    
                    # 評估充放電性能
                    charge_discharge_metrics = self.evaluate_charge_discharge(
                        model, 
                        test_dataset, 
                        temp,
                        threshold=self.config_override.get('charge_threshold', 0.0)
                    )
                
                return temp, metrics, data, charge_discharge_metrics
                
            except Exception as e:
                logger.error(f"評估 {temp} 溫度模型失敗: {e}")
                return temp, {'error': str(e)}, (None, None), None
        
        # 使用線程池並行評估
        with self.executor as executor:
            futures = [executor.submit(evaluate_temp_parallel, temp) for temp in temps]
            
            for future in as_completed(futures):
                try:
                    temp, metrics, (y_true, y_pred), cd_metrics = future.result()
                    results[temp] = metrics
                    prediction_samples[temp] = (y_true, y_pred)
                    
                    if cd_metrics:
                        cd_results[temp] = cd_metrics
                    
                    # 清理內存
                    memory_cleanup()
                except Exception as e:
                    logger.error(f"處理評估結果時出錯: {e}")
        
        # 保存評估結果
        self._save_results(experiment_name, results, "evaluation")
        
        # 保存充放電評估結果 (如果有)
        if cd_results:
            self._save_results(experiment_name, cd_results, "charge_discharge_evaluation")
        
        return results, prediction_samples, cd_results
    
    def _save_results(self, experiment_name, results, suffix="evaluation"):
        """保存評估結果到文件"""
        output_dir = self.config_override.get('output_dir', config.get('system.output_dir'))
        output_path = Path(output_dir) / f"{experiment_name}_{suffix}.json"
        
        # 確保目錄存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # 移除不可序列化的樣本數據
            serializable_results = self._prepare_serializable_results(results)
                
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"評估結果已保存至: {output_path}")
        except Exception as e:
            logger.error(f"保存評估結果失敗: {e}")
    
    def _prepare_serializable_results(self, results):
        """準備可序列化的結果"""
        if isinstance(results, dict):
            if "samples" in results:
                # 直接移除樣本，避免嵌套遍歷
                return {k: v for k, v in results.items() if k != 'samples'}
            
            # 遞歸處理嵌套字典
            return {k: self._prepare_serializable_results(v) if isinstance(v, dict) else v 
                   for k, v in results.items()}
        return results
    
    @log_step("交叉溫度評估")
    def cross_temp_evaluation(self, experiment_name, model_type, temps, checkpoint_dir, data_prefix="source"):
        """交叉溫度評估"""
        cross_temp_results = {}
        
        # 對每個模型溫度進行評估
        for model_temp in temps:
            logger.info(f"加載 {model_temp} 溫度模型進行交叉評估")
            
            try:
                # 加載模型
                model = ModelFactory.load_model(experiment_name, model_type, model_temp, checkpoint_dir)
                
                # 初始化該模型溫度的結果
                cross_temp_results[model_temp] = {}
                
                # 在每個溫度數據上評估
                for data_temp in temps:
                    logger.info(f"  在 {data_temp} 溫度數據上評估")
                    
                    try:
                        # 加載測試數據集
                        test_dataset = self.data_loader.get_dataset(f"{data_prefix}_{data_temp}", "test")
                        
                        # 嘗試加載標準化參數
                        try:
                            scaler_params = self.data_loader.load_scaler_params(f"{data_prefix}_{data_temp}")
                        except Exception:
                            logger.warning(f"無法加載 {data_temp} 的標準化參數，使用未標準化評估")
                            scaler_params = None
                        
                        # 評估模型
                        metrics, _ = self.evaluate_model(model, test_dataset, scaler_params)
                        
                        # 保存結果
                        cross_temp_results[model_temp][data_temp] = metrics
                        
                    except Exception as e:
                        logger.error(f"評估 {model_temp} 模型在 {data_temp} 數據上失敗: {e}")
                        cross_temp_results[model_temp][data_temp] = {'error': str(e)}
                
                # 清理內存
                memory_cleanup()
                
            except Exception as e:
                logger.error(f"加載 {model_temp} 溫度模型失敗: {e}")
                cross_temp_results[model_temp] = {'error': str(e)}
        
        # 保存交叉評估結果
        self._save_results(experiment_name, cross_temp_results, "cross_evaluation")
        
        # 繪製交叉溫度熱力圖
        for metric in ["FDCR_MAE", "FDCR_RMSE", "FDCR_R2", "RSOC_MAE", "RSOC_RMSE", "RSOC_R2", "loss"]:
            self.plot_cross_temp_heatmap(cross_temp_results, experiment_name, metric)
        
        return cross_temp_results
    
    @log_step("繪製交叉溫度熱力圖")
    def plot_cross_temp_heatmap(self, cross_results, experiment_name, metric="FDCR_MAE"):
        """繪製交叉溫度評估熱力圖"""
        figures_dir = self.config_override.get('figures_dir', config.get('system.figures_dir'))
        output_path = Path(figures_dir) / f"{experiment_name}_cross_{metric}.png"
        
        # 提取所有溫度
        model_temps = list(cross_results.keys())
        data_temps = []
        for model_temp in model_temps:
            data_temps.extend(list(cross_results[model_temp].keys()))
        data_temps = sorted(list(set(data_temps)))
        
        # 檢查是否有足夠的數據
        if not model_temps or not data_temps:
            logger.warning(f"沒有足夠的數據繪製交叉溫度熱力圖")
            return None
        
        # 創建熱力圖數據
        heatmap_data = np.zeros((len(model_temps), len(data_temps)))
        
        for i, model_temp in enumerate(model_temps):
            for j, data_temp in enumerate(data_temps):
                if (data_temp in cross_results[model_temp] and 
                    isinstance(cross_results[model_temp][data_temp], dict) and
                    metric in cross_results[model_temp][data_temp]):
                    heatmap_data[i, j] = cross_results[model_temp][data_temp][metric]
        
        # 創建圖形
        plt.figure(figsize=(8, 6))
        im = plt.imshow(heatmap_data, cmap='YlOrRd')
        
        # 添加色彩條
        cbar = plt.colorbar(im)
        cbar.set_label(metric)
        
        # 設置刻度標籤
        plt.xticks(np.arange(len(data_temps)), data_temps, rotation=45)
        plt.yticks(np.arange(len(model_temps)), model_temps)
        
        # 添加標題和軸標籤
        plt.title(f'交叉溫度評估 - {metric}')
        plt.xlabel('數據溫度')
        plt.ylabel('模型溫度')
        
        # 在每個方格中添加數值
        for i in range(len(model_temps)):
            for j in range(len(data_temps)):
                plt.text(j, i, f'{heatmap_data[i, j]:.4f}',
                        ha="center", va="center", color="black" if heatmap_data[i, j] < 0.5 else "white",
                        fontsize=8)
        
        plt.tight_layout()
        
        # 保存圖形
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"交叉溫度熱力圖已保存至: {output_path}")
        
        return str(output_path)
    
    @log_step("繪製預測對比圖")
    def plot_predictions(self, y_true, y_pred, output_path, temp, n_samples=5, target_names=None):
        """繪製預測值與真實值對比圖"""
        if target_names is None:
            target_names = ['FDCR', 'RSOC']
        
        output_path = Path(output_path)
        
        if y_true is None or y_pred is None:
            logger.warning(f"無法繪製預測圖，數據為空")
            return None
        
        if len(y_true.shape) != 3 or len(y_pred.shape) != 3:
            logger.warning(f"無法繪製預測圖，形狀不匹配: y_true {y_true.shape}, y_pred {y_pred.shape}")
            return None
        
        n_features = min(y_true.shape[2], len(target_names))
        n_samples = min(n_samples, y_true.shape[0])
        
        # 創建圖形
        fig, axes = plt.subplots(n_samples, n_features, figsize=(4*n_features, 3*n_samples))
        
        # 處理單樣本或單特徵的情況
        if n_samples == 1 and n_features == 1:
            axes = np.array([[axes]])
        elif n_samples == 1:
            axes = np.array([axes])
        elif n_features == 1:
            axes = axes.reshape(-1, 1)
        
        # 繪製每個樣本的每個特徵
        for i in range(n_samples):
            for j in range(n_features):
                ax = axes[i, j]
                time_steps = np.arange(y_true.shape[1])
                
                # 繪製真實值和預測值
                ax.plot(time_steps, y_true[i, :, j], 'b-', label='真實值')
                ax.plot(time_steps, y_pred[i, :, j], 'r--', label='預測值')
                
                # 計算誤差指標
                mae = mean_absolute_error(y_true[i, :, j], y_pred[i, :, j])
                rmse = np.sqrt(mean_squared_error(y_true[i, :, j], y_pred[i, :, j]))
                
                # 設置標題和標籤
                if i == 0:
                    ax.set_title(target_names[j])
                ax.set_ylabel(f'樣本 {i+1}')
                if i == n_samples - 1:
                    ax.set_xlabel('時間步')
                
                # 添加誤差信息
                ax.text(0.05, 0.05, f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}', 
                       transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
                
                # 添加圖例
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 調整子圖之間的間距
        plt.tight_layout()
        
        # 在頂部添加大標題
        fig.suptitle(f'{temp} 溫度下的時間序列預測', fontsize=16, y=1.02)
        
        # 保存圖形
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"預測可視化已保存至: {output_path}")
        
        return str(output_path)
    
    @log_step("繪製指標對比圖")
    def plot_comparison_metrics(self, results, output_path, metric_names=None):
        """繪製不同溫度或實驗之間的指標對比圖"""
        if metric_names is None:
            metric_names = ['FDCR_MAE', 'FDCR_RMSE', 'FDCR_R2', 'RSOC_MAE', 'RSOC_RMSE', 'RSOC_R2']
        
        output_path = Path(output_path)
        
        # 過濾出可用的指標名稱
        all_metrics = set()
        for temp_results in results.values():
            all_metrics.update(temp_results.keys())
        
        available_metrics = [m for m in metric_names if m in all_metrics]
        if not available_metrics:
            logger.warning(f"沒有可用的指標進行比較")
            return None
        
        # 提取指標數據
        temps = list(results.keys())
        metrics_data = {metric: [results[temp].get(metric, 0) for temp in temps] 
                      for metric in available_metrics}
        
        # 創建圖形布局
        n_metrics = len(available_metrics)
        n_rows = (n_metrics + 1) // 2  # 每行最多2個圖
        n_cols = min(2, n_metrics)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # 處理單一指標的情況
        if n_metrics == 1:
            axes = np.array([axes])
        elif n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        
        # 繪製每個指標
        for i, metric in enumerate(available_metrics):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col] if n_rows > 1 and n_cols > 1 else axes[i]
            
            # 條形圖
            x = np.arange(len(temps))
            bars = ax.bar(x, metrics_data[metric], color='skyblue', width=0.6)
            
            # 設置標題和標籤
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(temps, rotation=45)
            ax.grid(True, axis='y', alpha=0.3)
            
            # 在條形上方顯示數值
            for bar, value in zip(bars, metrics_data[metric]):
                height = bar.get_height()
                max_val = max(metrics_data[metric]) if metrics_data[metric] else 0
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02*max_val,
                       f'{value:.4f}', ha='center', va='bottom', fontsize=8)
        
        # 隱藏可能的空子圖
        for i in range(n_metrics, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            if n_rows > 1 and n_cols > 1:
                axes[row, col].axis('off')
            elif n_metrics < n_rows * n_cols:
                axes[i].axis('off')
        
        # 調整子圖之間的間距
        plt.tight_layout()
        
        # 保存圖形
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"指標對比圖已保存至: {output_path}")
        
        return str(output_path)
    
    @log_step("比較多個實驗")
    def compare_experiments(self, experiment_names, output_dir, metrics=None, figures_dir=None):
        """比較多個實驗的評估結果"""
        if metrics is None:
            metrics = ["FDCR_MAE", "FDCR_RMSE", "FDCR_R2", "RSOC_MAE", "RSOC_RMSE", "RSOC_R2"]
        
        output_dir = Path(output_dir)
        
        # 初始化結果字典
        compare_results = {}
        
        # 加載每個實驗的評估結果
        for exp_name in experiment_names:
            exp_results_path = output_dir / f"{exp_name}_evaluation.json"
            if exp_results_path.exists():
                try:
                    with open(exp_results_path, 'r') as f:
                        exp_results = json.load(f)
                    
                    # 計算各個指標的平均值
                    metrics_avg = {}
                    for metric in metrics:
                        values = []
                        for temp_results in exp_results.values():
                            if metric in temp_results:
                                values.append(temp_results[metric])
                        
                        if values:
                            metrics_avg[metric] = sum(values) / len(values)
                    
                    compare_results[exp_name] = metrics_avg
                    
                except Exception as e:
                    logger.error(f"加載實驗 {exp_name} 的評估結果失敗: {e}")
            else:
                logger.warning(f"找不到實驗 {exp_name} 的評估結果")
        
        # 保存比較結果
        if compare_results:
            output_path = output_dir / "experiments_comparison.json"
            
            try:
                with open(output_path, 'w') as f:
                    json.dump(compare_results, f, indent=2)
                logger.info(f"實驗比較結果已保存至: {output_path}")
                
                # 繪製比較圖表
                if figures_dir:
                    figure_path = Path(figures_dir) / "experiments_comparison.png"
                    self.plot_comparison_metrics(compare_results, figure_path)
                
            except Exception as e:
                logger.error(f"保存實驗比較結果失敗: {e}")
        
        return compare_results

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="電池老化預測系統評估腳本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本參數
    parser.add_argument("--experiment-name", "-e", type=str, required=True,
                      help="實驗名稱，用於加載模型")
    parser.add_argument("--model-type", "-m", type=str, default=config.get("model.type"),
                      choices=["baseline", "cnn_gru", "pinn", "gan"],
                      help="模型類型")
    parser.add_argument("--base-temp", "-bt", type=str, default=config.get("training.base_temp"),
                      help="基準溫度")
    parser.add_argument("--temps", "-t", type=str, nargs='+', 
                      default=None,
                      help="要評估的溫度列表")
    
    # 數據參數
    parser.add_argument("--data-dir", "-d", type=str, default=config.get("system.data_dir"),
                      help="數據目錄")
    parser.add_argument("--tfrecord-dir", "-tf", type=str, default=config.get("system.tfrecord_dir"),
                      help="TFRecord目錄")
    parser.add_argument("--data-prefix", "-dp", type=str, default="source",
                      help="數據文件前綴")
    parser.add_argument("--batch-size", "-bs", type=int, default=32,
                      help="評估批次大小")
    
    # 評估選項
    parser.add_argument("--checkpoint-dir", "-c", type=str, default=config.get("system.checkpoint_dir"),
                      help="檢查點目錄")
    parser.add_argument("--output-dir", "-o", type=str, default=config.get("system.output_dir"),
                      help="評估結果輸出目錄")
    parser.add_argument("--figures-dir", "-f", type=str, default=config.get("system.figures_dir"),
                      help="圖表輸出目錄")
    parser.add_argument("--cross-temp", action="store_true", default=True,
                      help="進行交叉溫度評估")
    parser.add_argument("--no-cross-temp", action="store_false", dest="cross_temp",
                      help="不進行交叉溫度評估")
    parser.add_argument("--detailed", action="store_true", default=False,
                      help="生成詳細評估報告")
    parser.add_argument("--compare", type=str, nargs='+', default=None,
                      help="比較多個實驗 (提供實驗名稱列表)")
    parser.add_argument("--time-series", action="store_true", default=False,
                        help="生成時間序列預測對比圖")
    parser.add_argument("--samples", type=int, default=5,
                        help="時間序列預測顯示的樣本數量")
    parser.add_argument("--parallel", action="store_true", default=True,
                        help="使用並行處理加速評估")
    
    # 新增充放電分析參數
    parser.add_argument("--charge-discharge", action="store_true", default=False,
                      help="進行充放電分析")
    parser.add_argument("--charge-threshold", type=float, default=0.0,
                      help="充放電判定閾值，高於此值視為充電")
    
    # 可視化參數
    parser.add_argument("--dpi", type=int, default=300,
                      help="圖表DPI")
    parser.add_argument("--plot-style", type=str, default="seaborn-v0_8-whitegrid",
                      help="圖表風格")
    parser.add_argument("--interactive", action="store_true", default=False,
                      help="生成互動式圖表")
    
    # 控制選項
    parser.add_argument("--debug", action="store_true", default=config.get("system.debug"),
                      help="啟用調試模式")
    parser.add_argument("--config", type=str, default=None,
                      help="使用自定義配置文件")
    
    args = parser.parse_args()
    
    # 如果沒有提供溫度列表，使用默認值
    if args.temps is None:
        args.temps = [args.base_temp] + config.get("training.transfer_temps")
    
    return args

def setup_config_from_args(args):
    """根據命令行參數更新配置"""
    # 從自定義配置文件加載（如果有）
    if args.config:
        config.load_from_file(args.config)
    
    # 基於命令行參數創建配置覆蓋
    config_override = {
        "model.type": args.model_type,
        "training.base_temp": args.base_temp,
        "system.debug": args.debug,
        "system.data_dir": args.data_dir,
        "system.tfrecord_dir": args.tfrecord_dir,
        "system.checkpoint_dir": args.checkpoint_dir,
        "system.output_dir": args.output_dir,
        "system.figures_dir": args.figures_dir,
        "batch_size": args.batch_size,
        "data_prefix": args.data_prefix,
        "detailed": args.detailed,
        "cross_temp": args.cross_temp,
        "time_series": args.time_series,
        "samples": args.samples,
        "parallel": args.parallel,
        "charge_discharge": args.charge_discharge,
        "charge_threshold": args.charge_threshold,
        "dpi": args.dpi,
        "plot_style": args.plot_style,
        "interactive": args.interactive,
        "experiment_name": args.experiment_name
    }
    
    # 更新全局配置
    config.update(config_override)
    
    return config_override

def display_welcome_message(args):
    """顯示歡迎訊息和配置摘要"""
    print(Colors.header("\n" + "=" * 60))
    print(Colors.header("     電池老化預測系統評估腳本     "))
    print(Colors.header("=" * 60))
    
    # 基本配置
    print(Colors.info(f"\n[基本配置]"))
    print(f"實驗名稱:      {Colors.info(args.experiment_name)}")
    print(f"模型類型:      {Colors.info(args.model_type)}")
    print(f"評估溫度:      {Colors.info(', '.join(args.temps))}")
    
    # 評估選項
    print(Colors.info(f"\n[評估選項]"))
    print(f"交叉溫度評估:  {Colors.success('啟用') if args.cross_temp else Colors.warning('停用')}")
    print(f"時間序列預測:  {Colors.success('啟用') if args.time_series else Colors.warning('停用')}")
    print(f"充放電分析:    {Colors.success('啟用') if args.charge_discharge else Colors.warning('停用')}")
    if args.charge_discharge:
        print(f"充放電閾值:    {Colors.info(str(args.charge_threshold))}")
    
    # 可視化選項
    print(Colors.info(f"\n[可視化選項]"))
    print(f"圖表DPI:       {Colors.info(str(args.dpi))}")
    print(f"圖表風格:      {Colors.info(args.plot_style)}")
    print(f"互動式圖表:    {Colors.success('啟用') if args.interactive else Colors.warning('停用')}")
    
    # 控制選項
    print(Colors.info(f"\n[控制選項]"))
    print(f"並行處理:      {Colors.success('啟用') if args.parallel else Colors.warning('停用')}")
    print(f"調試模式:      {Colors.success('啟用') if args.debug else Colors.warning('停用')}")
    
    print("\n" + "=" * 60 + "\n")

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_arguments()
    
    # 更新配置
    config_override = setup_config_from_args(args)
    
    # 確保目錄存在
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    
    # 輸出歡迎訊息
    display_welcome_message(args)
    
    try:
        # 創建評估器
        evaluator = ModelEvaluator(config_override)
        
        # 執行評估流程
        results = {}
        
        # 1. 評估所有溫度模型
        logger.info("執行評估階段...")
        evaluation_results, prediction_samples, cd_results = evaluator.evaluate_all_temps(
            experiment_name=args.experiment_name,
            model_type=args.model_type,
            temps=args.temps,
            checkpoint_dir=args.checkpoint_dir,
            data_prefix=args.data_prefix
        )
        results['evaluation'] = evaluation_results
        
        # 2. 交叉溫度評估（如果啟用）
        if args.cross_temp:
            logger.info("執行交叉溫度評估階段...")
            cross_temp_results = evaluator.cross_temp_evaluation(
                experiment_name=args.experiment_name,
                model_type=args.model_type,
                temps=args.temps,
                checkpoint_dir=args.checkpoint_dir,
                data_prefix=args.data_prefix
            )
            results['cross_evaluation'] = cross_temp_results
        
        # 3. 繪製預測對比圖（如果啟用）
        if args.time_series:
            logger.info("繪製預測對比圖階段...")
            prediction_plot_paths = []
            
            for temp, (y_true, y_pred) in prediction_samples.items():
                if y_true is not None and y_pred is not None:
                    output_path = Path(args.figures_dir) / f"{args.experiment_name}_{temp}_predictions.png"
                    plot_path = evaluator.plot_predictions(
                        y_true, y_pred, output_path, temp, n_samples=args.samples
                    )
                    if plot_path:
                        prediction_plot_paths.append(plot_path)
            
            results['prediction_plots'] = prediction_plot_paths
        
        # 4. 繪製指標對比圖
        logger.info("繪製指標對比圖階段...")
        metrics_plot_path = evaluator.plot_comparison_metrics(
            evaluation_results,
            Path(args.figures_dir) / f"{args.experiment_name}_metrics_comparison.png"
        )
        results['metrics_plot'] = metrics_plot_path
        
        # 5. 比較多個實驗（如果提供了比較列表）
        if args.compare:
            logger.info("比較多個實驗階段...")
            compare_results = evaluator.compare_experiments(
                experiment_names=[args.experiment_name] + args.compare,
                output_dir=args.output_dir,
                figures_dir=args.figures_dir
            )
            results['compare_results'] = compare_results
        
        # 打印結果摘要
        print(Colors.header("\n===== 評估結果摘要 ====="))
        
        # 標準評估結果
        for temp, temp_results in evaluation_results.items():
            print(Colors.info(f"\n{temp} 溫度:"))
            for metric in ["FDCR_MAE", "FDCR_RMSE", "FDCR_R2", "RSOC_MAE", "RSOC_RMSE", "RSOC_R2"]:
                if metric in temp_results:
                    print(f"  {metric}: {temp_results[metric]:.6f}")
        
        # 充放電評估結果（如果有）
        if cd_results:
            print(Colors.header("\n===== 充放電分析摘要 ====="))
            for temp, temp_cd in cd_results.items():
                print(Colors.info(f"\n{temp} 溫度:"))
                
                if 'overall' in temp_cd and 'loss' in temp_cd['overall']:
                    print(f"  整體損失: {temp_cd['overall']['loss']:.6f}")
                
                if 'charge' in temp_cd and 'loss' in temp_cd['charge']:
                    print(f"  充電損失: {temp_cd['charge']['loss']:.6f} (樣本數: {temp_cd['charge']['count']})")
                
                if 'discharge' in temp_cd and 'loss' in temp_cd['discharge']:
                    print(f"  放電損失: {temp_cd['discharge']['loss']:.6f} (樣本數: {temp_cd['discharge']['count']})")
                
                if 'comparative' in temp_cd and 'loss_ratio' in temp_cd['comparative']:
                    ratio = temp_cd['comparative']['loss_ratio']
                    ratio_str = f"{ratio:.3f}"
                    
                    if ratio > 1.5:
                        ratio_str = Colors.warning(ratio_str + " (!)")
                        print(f"  放電/充電比: {ratio_str} - 放電表現顯著差於充電")
                    elif ratio < 0.5:
                        ratio_str = Colors.warning(ratio_str + " (!)")
                        print(f"  放電/充電比: {ratio_str} - 充電表現顯著差於放電")
                    else:
                        print(f"  放電/充電比: {ratio_str} - 表現平衡")
                        
        # 顯示輸出位置
        print(Colors.success(f"\n評估結果已保存至: {args.output_dir}"))
        print(Colors.success(f"圖表已保存至: {args.figures_dir}"))
        
        logger.info("評估完成")
        return 0
        
    except Exception as e:
        import traceback
        logger.error(f"評估過程中出錯: {e}")
        logger.error(traceback.format_exc())
        print(Colors.error(f"\n評估失敗: {e}"))
        return 1
    
    finally:
        # 清理資源
        logger.info("清理資源...")
        memory_cleanup()
        
        # 關閉日誌系統

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
