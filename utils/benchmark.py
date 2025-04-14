#utils/benchmark.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""性能基準測試工具 - 電池老化預測系統 (優化版)"""

import os
import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, TypeVar
from functools import wraps
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import gc
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt

# 嘗試導入 TensorFlow 和 CUDA
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

# 設置記錄器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('benchmark')

# 類型定義
T = TypeVar('T')
PathLike = Union[str, Path]


@dataclass
class SystemResources:
    """系統資源數據類"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used: List[float] = None
    gpu_utilization: List[float] = None


class MonitoringPipeline:
    """監控管道，用於處理和分析系統資源數據"""
    
    def __init__(self, data: List[SystemResources]):
        """初始化監控管道"""
        self.data = data
        self.df = pd.DataFrame([asdict(point) for point in data])
        self.transformations = []
    
    def add_transformation(self, func: Callable[[pd.DataFrame], pd.DataFrame]) -> 'MonitoringPipeline':
        """添加數據轉換"""
        self.transformations.append(func)
        return self
    
    def process(self) -> pd.DataFrame:
        """處理數據"""
        result = self.df.copy()
        for transform in self.transformations:
            result = transform(result)
        return result
    
    def resample(self, rule: str) -> 'MonitoringPipeline':
        """重新採樣數據"""
        def transform(df):
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('datetime', inplace=True)
            return df.resample(rule).mean().ffill()
        return self.add_transformation(transform)
    
    def smooth(self, window: int = 3) -> 'MonitoringPipeline':
        """平滑數據"""
        return self.add_transformation(lambda df: df.rolling(window=window, min_periods=1).mean())
    
    def calculate_stats(self) -> Dict[str, Dict[str, float]]:
        """計算統計數據"""
        result = self.process()
        stats = {}
        
        # 對每一列計算統計量
        for column in result.columns:
            if column not in ['timestamp', 'datetime']:
                try:
                    series = result[column].dropna()
                    if not series.empty:
                        stats[column] = {
                            'mean': series.mean(),
                            'std': series.std(),
                            'min': series.min(),
                            'max': series.max(),
                            'median': series.median()
                        }
                except Exception as e:
                    logger.warning(f"計算 {column} 統計數據時出錯: {e}")
        
        return stats
    
    def plot(self, filename: Optional[PathLike] = None, 
            title: str = "System Performance Monitoring", 
            figsize: Tuple[int, int] = (12, 10)) -> Optional[str]:
        """繪製監控數據圖表"""
        result = self.process()
        
        # 計算子圖數量
        n_plots = 2  # CPU + 記憶體
        if 'gpu0_memory_gb' in result.columns:
            n_plots += 1  # GPU 記憶體
        if 'gpu0_utilization' in result.columns:
            n_plots += 1  # GPU 使用率
        
        # 創建圖表
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        if n_plots == 1:
            axes = [axes]
        
        # 繪製 CPU 使用率
        axes[0].plot(result.index, result['cpu_percent'], 'b-')
        axes[0].set_ylabel('CPU 使用率 (%)')
        axes[0].set_title('CPU 使用率')
        axes[0].grid(True)
        
        # 繪製記憶體使用率
        axes[1].plot(result.index, result['memory_percent'], 'r-')
        axes[1].set_ylabel('記憶體使用率 (%)')
        axes[1].set_title('系統記憶體使用率')
        axes[1].grid(True)
        
        # 繪製 GPU 相關數據（如果有）
        plot_idx = 2
        
        # GPU 記憶體
        gpu_memory_columns = [col for col in result.columns if col.startswith('gpu') and col.endswith('_memory_gb')]
        if gpu_memory_columns and plot_idx < len(axes):
            ax = axes[plot_idx]
            for col in gpu_memory_columns:
                gpu_id = col.split('_')[0][3:]
                ax.plot(result.index, result[col], label=f'GPU {gpu_id}')
            ax.set_ylabel('GPU 記憶體 (GB)')
            ax.set_title('GPU 記憶體使用')
            ax.legend()
            ax.grid(True)
            plot_idx += 1
        
        # GPU 使用率
        gpu_util_columns = [col for col in result.columns if col.startswith('gpu') and col.endswith('_utilization')]
        if gpu_util_columns and plot_idx < len(axes):
            ax = axes[plot_idx]
            for col in gpu_util_columns:
                gpu_id = col.split('_')[0][3:]
                ax.plot(result.index, result[col], label=f'GPU {gpu_id}')
            ax.set_ylabel('GPU 使用率 (%)')
            ax.set_title('GPU 使用率')
            ax.legend()
            ax.grid(True)
        
        # 設置 x 軸標籤
        if hasattr(result.index, 'strftime'):
            # 時間索引格式化
            date_format = '%H:%M:%S' if result.index[-1] - result.index[0] < pd.Timedelta(days=1) else '%Y-%m-%d %H:%M:%S'
            axes[-1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
            fig.autofmt_xdate()
        
        axes[-1].set_xlabel('時間')
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        # 保存或顯示圖表
        if filename:
            save_path = Path(filename)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"已保存監視圖表到 {save_path}")
            plt.close(fig)
            return str(save_path)
        else:
            plt.show()
            return None


class PerformanceMonitor:
    """性能監視器，用於監視 CPU、記憶體和 GPU 使用情況"""
    
    def __init__(self, interval=0.5, max_data_points=1000):
        """初始化監視器"""
        self.interval = interval
        self.max_data_points = max_data_points
        self.monitoring = False
        self.resources = []
        self.thread = None
        self.start_time = None
        self._lock = threading.Lock()
        
        # 檢測 GPU 數量
        self.gpu_count = 0
        if HAS_TF:
            self.gpu_count = len(tf.config.list_physical_devices('GPU'))
        elif HAS_CUDA:
            self.gpu_count = cuda.Device.count()
    
    def _monitor_loop(self):
        """監視循環，在單獨線程中運行"""
        while self.monitoring:
            try:
                # 記錄時間戳
                current_time = time.time() - self.start_time
                
                # 基本系統資源
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024 * 1024 * 1024)  # GB
                
                # 監視 GPU 資源
                gpu_memory_used = []
                gpu_utilization = []
                
                if HAS_TF:
                    try:
                        for i in range(self.gpu_count):
                            mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                            gpu_memory_used.append(mem_info['current'] / (1024 * 1024 * 1024))  # GB
                    except Exception:
                        gpu_memory_used = [0] * self.gpu_count
                elif HAS_CUDA:
                    try:
                        for i in range(self.gpu_count):
                            handle = cuda.Device(i)
                            info = cuda.DeviceAttribute()
                            free, total = handle.get_memory_info()
                            used = (total - free) / (1024 * 1024 * 1024)  # GB
                            gpu_memory_used.append(used)
                            gpu_utilization.append(info.gpu_utilization)
                    except Exception:
                        gpu_memory_used = [0] * self.gpu_count
                        gpu_utilization = [0] * self.gpu_count
                
                # 創建並添加資源數據點
                resource = SystemResources(
                    timestamp=current_time,
                    cpu_percent=cpu_percent,
                    memory_percent=memory_percent,
                    memory_used_gb=memory_used_gb,
                    gpu_memory_used=gpu_memory_used,
                    gpu_utilization=gpu_utilization
                )
                
                with self._lock:
                    self.resources.append(resource)
                    # 限制數據點數量
                    if len(self.resources) > self.max_data_points:
                        self.resources = self.resources[-self.max_data_points:]
            except Exception as e:
                logger.error(f"監視循環出錯: {e}")
            
            time.sleep(self.interval)
    
    def start(self):
        """開始監視"""
        if self.monitoring:
            return
        
        # 重置數據並啟動監視線程
        self.resources = []
        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        logger.info("性能監視已啟動")
    
    def stop(self) -> List[SystemResources]:
        """停止監視並返回結果"""
        if not self.monitoring:
            return []
        
        # 停止監視線程
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        logger.info("性能監視已停止")
        return self.resources
    
    def create_pipeline(self) -> MonitoringPipeline:
        """創建監控數據處理管道"""
        return MonitoringPipeline(self.resources)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """獲取當前系統統計數據"""
        stats = {}
        
        # CPU 信息
        stats['cpu'] = {
            'percent': psutil.cpu_percent(),
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False)
        }
        
        # 記憶體信息
        mem = psutil.virtual_memory()
        stats['memory'] = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent
        }
        
        # 磁盤信息
        disk = psutil.disk_usage('/')
        stats['disk'] = {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'used_gb': disk.used / (1024**3),
            'percent': disk.percent
        }
        
        # GPU 信息
        stats['gpu'] = {'count': self.gpu_count}
        if self.gpu_count > 0:
            if HAS_TF:
                for i in range(self.gpu_count):
                    stats['gpu'][f'gpu{i}'] = {}
                    try:
                        mem_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                        stats['gpu'][f'gpu{i}']['memory_gb'] = mem_info['current'] / (1024**3)
                    except:
                        pass
            elif HAS_CUDA:
                for i in range(self.gpu_count):
                    stats['gpu'][f'gpu{i}'] = {}
                    try:
                        handle = cuda.Device(i)
                        free, total = handle.get_memory_info()
                        stats['gpu'][f'gpu{i}']['memory_total_gb'] = total / (1024**3)
                        stats['gpu'][f'gpu{i}']['memory_free_gb'] = free / (1024**3)
                        stats['gpu'][f'gpu{i}']['memory_used_gb'] = (total - free) / (1024**3)
                        stats['gpu'][f'gpu{i}']['utilization'] = handle.compute_capability()
                    except:
                        pass
        
        return stats
    
    def save_results(self, filename: PathLike, format: str = 'csv'):
        """保存監視結果到文件"""
        # 創建並處理數據
        result = self.create_pipeline().process()
        
        # 保存數據
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'csv':
            result.to_csv(path)
        elif format.lower() == 'json':
            result.to_json(path, orient='records', indent=2)
        elif format.lower() == 'pickle':
            result.to_pickle(path)
        else:
            raise ValueError(f"不支持的格式: {format}")
        
        logger.info(f"已保存監視結果到 {path}")
    
    def plot_results(self, filename: Optional[PathLike] = None):
        """繪製監視結果"""
        return self.create_pipeline().plot(filename)


def with_monitoring(func=None, interval=0.5, save_path=None, plot=False, format='csv'):
    """性能監控裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 創建並啟動監視器
            monitor = PerformanceMonitor(interval=interval)
            monitor.start()
            
            try:
                # 執行函數
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
            finally:
                # 停止監視
                monitor.stop()
                
                # 保存結果
                if save_path:
                    try:
                        path = Path(save_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        monitor.save_results(path, format=format)
                        
                        # 繪製圖表
                        if plot:
                            monitor.plot_results(path.with_suffix('.png'))
                    
                    except Exception as e:
                        logger.error(f"保存監視結果時出錯: {e}")
            
            return result
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def benchmark(func=None, label=None, n_runs=1, warmup=1, gc_collect=True):
    """函數性能基準測試裝飾器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = label or func.__name__
            
            # 預熱運行
            if warmup > 0:
                logger.info(f"執行 {warmup} 次預熱運行...")
                for _ in range(warmup):
                    _ = func(*args, **kwargs)
            
            # 執行基準測試
            logger.info(f"開始對 {func_name} 執行 {n_runs} 次基準測試...")
            times = []
            
            for i in range(n_runs):
                # 垃圾回收（可選）
                if gc_collect:
                    gc.collect()
                
                # 計時執行
                start = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                
                times.append(elapsed)
                logger.info(f"  運行 {i+1}/{n_runs}: {elapsed:.6f} 秒")
            
            # 計算並輸出統計數據
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            max_time = np.max(times)
            
            logger.info(f"基準測試結果 ({func_name}):")
            logger.info(f"  平均時間: {avg_time:.6f} 秒")
            logger.info(f"  標準差: {std_time:.6f} 秒")
            logger.info(f"  最小時間: {min_time:.6f} 秒")
            logger.info(f"  最大時間: {max_time:.6f} 秒")
            
            return result
        
        return wrapper
    
    # 允許裝飾器有或沒有參數
    if func is None:
        return decorator
    else:
        return decorator(func)


class BenchmarkSuite:
    """基準測試套件，用於比較多個函數的性能"""
    
    def __init__(self, results_dir: PathLike = 'benchmark_results'):
        """初始化基準測試套件"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tests = []
    
    def add_test(self, func: Callable, name: Optional[str] = None, **kwargs):
        """添加測試函數"""
        self.tests.append((func, name or func.__name__, kwargs))
        return self
    
    def run(self, n_runs: int = 3, save_results: bool = True) -> List[Dict[str, Any]]:
        """執行所有測試"""
        results = []
        
        for func, name, kwargs in self.tests:
            # 執行測試
            times = []
            for _ in range(n_runs):
                start = time.time()
                try:
                    _ = func(**kwargs)
                    elapsed = time.time() - start
                    times.append(elapsed)
                except Exception as e:
                    logger.error(f"測試 {name} 時出錯: {e}")
                    logger.error(traceback.format_exc())
                    times.append(float('inf'))
            
            # 計算統計數據
            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)
            
            # 添加結果
            results.append({
                'name': name,
                'avg_time': avg_time,
                'std_time': std_time,
                'min_time': min_time,
                'times': times
            })
        
        # 排序結果並輸出
        results = sorted(results, key=lambda x: x['avg_time'])
        
        logger.info("速度測試結果:")
        for i, res in enumerate(results):
            relative = f" (基準)" if i == 0 else f" ({res['avg_time'] / results[0]['avg_time']:.2f}x)"
            logger.info(f"  {i+1}. {res['name']}: {res['avg_time']:.6f} 秒 ±{res['std_time']:.6f}{relative}")
        
        # 保存結果
        if save_results:
            self._save_results(results)
        
        return results
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """保存測試結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存詳細結果
        results_file = self.results_dir / f"speed_test_{timestamp}.json"
        with results_file.open('w') as f:
            json.dump([{k: v for k, v in r.items() if k != 'times'} for r in results], f, indent=2)
        
        # 繪製結果圖表
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            [r['name'] for r in results],
            [r['avg_time'] for r in results],
            yerr=[r['std_time'] for r in results],
            capsize=5
        )
        
        # 添加數據標籤
        for i, bar in enumerate(bars):
            height = bar.get_height()
            relative = "" if i == 0 else f" ({results[i]['avg_time'] / results[0]['avg_time']:.2f}x)"
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + results[i]['std_time'] + 0.01,
                f"{results[i]['avg_time']:.4f}s{relative}",
                ha='center', va='bottom', rotation=0, fontsize=9
            )
        
        plt.ylabel('執行時間 (秒)')
        plt.title('函數執行時間比較')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # 保存圖表
        plot_file = self.results_dir / f"speed_test_{timestamp}.png"
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"已保存結果到 {results_file} 和 {plot_file}")


def run_speed_test(test_funcs, names=None, n_runs=3, save_results=True, results_dir='benchmark_results'):
    """執行速度測試並比較多個函數"""
    if names is None:
        names = [f.__name__ for f in test_funcs]
    
    # 創建並運行測試套件
    suite = BenchmarkSuite(results_dir=results_dir)
    for func, name in zip(test_funcs, names):
        suite.add_test(func, name)
    
    return suite.run(n_runs=n_runs, save_results=save_results)


def profile_memory(func):
    """記憶體使用分析裝飾器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 初始狀態
        gc.collect()
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # 記錄 GPU 記憶體（如果可用）
        gpu_before = {}
        if HAS_TF:
            for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
                try:
                    info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    gpu_before[i] = info['current'] / (1024 * 1024)  # MB
                except:
                    gpu_before[i] = None
        
        # 運行函數
        result = func(*args, **kwargs)
        
        # 最終狀態
        gc.collect()
        after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # 記錄 GPU 記憶體（如果可用）
        gpu_after = {}
        gpu_diff = {}
        if HAS_TF:
            for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
                try:
                    info = tf.config.experimental.get_memory_info(f'GPU:{i}')
                    gpu_after[i] = info['current'] / (1024 * 1024)  # MB
                    gpu_diff[i] = gpu_after[i] - (gpu_before[i] or 0)
                except:
                    gpu_after[i] = None
                    gpu_diff[i] = None
        
        # 輸出結果
        diff = after - before
        logger.info(f"記憶體分析 ({func.__name__}):")
        logger.info(f"  CPU 記憶體 (MB): {before:.1f} -> {after:.1f} (Δ{diff:.1f})")
        
        for i in gpu_before:
            if gpu_before[i] is not None and gpu_after[i] is not None:
                logger.info(f"  GPU:{i} 記憶體 (MB): {gpu_before[i]:.1f} -> {gpu_after[i]:.1f} (Δ{gpu_diff[i]:.1f})")
        
        return result
    
    return wrapper


def profile_tensorflow_model(model, input_data, logdir='tf_profile'):
    """使用 TensorFlow Profiler 分析模型性能"""
    if not HAS_TF:
        logger.error("未安裝 TensorFlow，無法進行模型分析")
        return
    
    from tensorflow.python.profiler import profiler_v2 as profiler
    
    # 確保目錄存在
    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    
    # 預熱模型
    logger.info("預熱模型...")
    _ = model(input_data)
    
    # 開始分析
    logger.info(f"開始分析模型性能，結果將保存到 {logdir}")
    profiler.start(logdir=str(logdir))
    
    # 運行模型
    with tf.profiler.experimental.Trace('inference'):
        _ = model(input_data)
    
    # 停止分析
    profiler.stop()
    
    logger.info(f"模型性能分析完成，使用以下命令查看結果: tensorboard --logdir={logdir}")
    
    return logdir


@with_monitoring
def cpu_stress_test(seconds=10, workers=None):
    """CPU 壓力測試"""
    workers = workers or psutil.cpu_count(logical=True)
    
    # 壓力測試函數
    def worker_func():
        end_time = time.time() + seconds
        while time.time() < end_time:
            # 執行密集計算
            _ = [i ** 2 for i in range(10000)]
    
    # 創建並運行線程
    threads = []
    for _ in range(workers):
        t = threading.Thread(target=worker_func)
        threads.append(t)
        t.start()
    
    # 等待線程完成
    for t in threads:
        t.join()
    
    # 獲取結果
    cpu_percent = psutil.cpu_percent()
    logger.info(f"CPU 壓力測試結果 ({workers} 執行緒, {seconds} 秒): CPU 使用率: {cpu_percent:.2f}%")
    
    return cpu_percent


@with_monitoring
def memory_stress_test(size_gb=1, seconds=5):
    """記憶體壓力測試"""
    # 分配記憶體
    logger.info(f"分配 {size_gb} GB 記憶體...")
    
    # 每塊 100MB，避免一次性分配過大
    block_size = 100 * 1024 * 1024  # 100MB
    num_blocks = int(size_gb * 1024 * 1024 * 1024 / block_size)
    
    # 分配並保持記憶體
    memory_blocks = []
    try:
        for _ in range(num_blocks):
            block = bytearray(block_size)
            for i in range(0, block_size, 4096):
                block[i] = 1  # 防止優化
            memory_blocks.append(block)
    except MemoryError:
        logger.warning(f"無法分配請求的記憶體大小 ({size_gb} GB)，已分配 {len(memory_blocks) * block_size / (1024**3):.2f} GB")
    
    # 保持記憶體分配
    logger.info(f"保持記憶體分配 {seconds} 秒...")
    time.sleep(seconds)
    
    # 獲取結果
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"記憶體壓力測試結果: 記憶體使用率: {memory_percent:.2f}%")
    
    # 釋放記憶體
    memory_blocks = None
    gc.collect()
    
    return memory_percent


@benchmark
def disk_benchmark(file_size_mb=100, chunk_size_kb=64, delete_file=True):
    """磁碟性能基準測試"""
    # 生成測試數據和路徑
    data_chunk = os.urandom(chunk_size_kb * 1024)
    num_chunks = file_size_mb * 1024 // chunk_size_kb
    test_file = Path.cwd() / f"disk_test_{int(time.time())}.bin"
    
    # 寫入測試
    logger.info(f"開始磁碟寫入測試 ({file_size_mb} MB)...")
    write_start = time.time()
    
    with test_file.open('wb') as f:
        for _ in range(num_chunks):
            f.write(data_chunk)
    
    write_time = time.time() - write_start
    write_speed = file_size_mb / write_time
    
    # 清空緩存
    if hasattr(os, 'sync'):
        os.sync()
    
    # 讀取測試
    logger.info(f"開始磁碟讀取測試 ({file_size_mb} MB)...")
    read_start = time.time()
    
    with test_file.open('rb') as f:
        while f.read(chunk_size_kb * 1024):
            pass
    
    read_time = time.time() - read_start
    read_speed = file_size_mb / read_time
    
    # 刪除測試文件
    if delete_file and test_file.exists():
        test_file.unlink()
    
    # 輸出結果
    logger.info(f"磁碟性能測試結果: 寫入速度: {write_speed:.2f} MB/s, 讀取速度: {read_speed:.2f} MB/s")
    
    return write_speed, read_speed


def network_benchmark(url="https://www.google.com", size_mb=10, num_requests=5, timeout=30):
    """網絡性能基準測試"""
    try:
        import requests
    except ImportError:
        logger.error("未安裝 requests 庫，無法進行網絡測試")
        return None
    
    # 驗證 URL
    test_url = url if "://" in url else f"https://{url}"
    
    # 嘗試訪問網站
    try:
        response = requests.head(test_url, timeout=timeout)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"無法連接到 {test_url}: {e}")
        return None
    
    # 查找適合的測試文件
    test_files = [
        "https://speed.cloudflare.com/10mb.bin",
        "https://proof.ovh.net/files/10Mb.dat",
        "https://sabnzbd.org/tests/internetspeed/10MB.bin"
    ]
    
    working_url = None
    for url in test_files:
        try:
            requests.head(url, timeout=timeout)
            working_url = url
            break
        except:
            continue
    
    if not working_url:
        logger.error("無法找到適合的測試文件")
        return None
    
    # 執行下載測試
    logger.info(f"開始網絡下載測試 ({num_requests} 次請求)...")
    download_speeds = []
    
    for i in range(num_requests):
        logger.info(f"  下載測試 {i+1}/{num_requests}")
        start_time = time.time()
        response = requests.get(working_url, stream=True, timeout=timeout)
        
        # 讀取數據
        downloaded = 0
        for chunk in response.iter_content(chunk_size=1024*1024):
            if chunk:
                downloaded += len(chunk)
        
        # 計算速度
        elapsed = time.time() - start_time
        speed_mbps = (downloaded / (1024 * 1024)) / elapsed
        download_speeds.append(speed_mbps)
        logger.info(f"    下載速度: {speed_mbps:.2f} MB/s")
    
    # 計算平均速度
    avg_speed = np.mean(download_speeds)
    logger.info(f"網絡性能測試結果: 平均下載速度: {avg_speed:.2f} MB/s")
    
    return avg_speed


class SystemBenchmark:
    """系統基準測試套件"""
    
    def __init__(self, results_dir='benchmark_results'):
        """初始化系統基準測試套件"""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
    
    def run_all(self, save_results=True):
        """運行所有基準測試"""
        logger.info("==== 系統性能測試工具 ====")
        
        # 獲取系統信息
        self.results['system_info'] = self._get_system_info()
        
        # 運行測試
        logger.info("\n==== 執行基準測試 ====")
        
        # CPU 測試
        logger.info("\n--- CPU 壓力測試 ---")
        self.results['cpu_test'] = cpu_stress_test(seconds=2)
        
        # 記憶體測試
        logger.info("\n--- 記憶體測試 ---")
        memory_gb = psutil.virtual_memory().total / (1024**3)
        test_size = min(0.5, memory_gb * 0.2)  # 使用 20% 的系統記憶體，最大 0.5GB
        self.results['memory_test'] = memory_stress_test(size_gb=test_size, seconds=1)
        
        # 磁碟測試
        logger.info("\n--- 磁碟性能測試 ---")
        write_speed, read_speed = disk_benchmark(file_size_mb=50)
        self.results['disk_test'] = {
            'write_speed': write_speed,
            'read_speed': read_speed
        }
        
        # 網絡測試（可選）
        logger.info("\n--- 網絡性能測試 ---")
        self.results['network_test'] = network_benchmark(num_requests=2)
        
        logger.info("\n所有測試完成")
        
        # 保存結果
        if save_results:
            self._save_results()
        
        return self.results
    
    def _get_system_info(self):
        """獲取系統信息"""
        info = {}
        
        # CPU 信息
        info['cpu'] = {
            'count_logical': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'name': self._get_cpu_name()
        }
        
        # 記憶體信息
        memory = psutil.virtual_memory()
        info['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3)
        }
        
        # 磁碟信息
        info['disk'] = {}
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                info['disk'][part.mountpoint] = {
                    'total_gb': usage.total / (1024**3),
                    'free_gb': usage.free / (1024**3)
                }
            except:
                pass
        
        # GPU 信息
        info['gpu'] = {}
        if HAS_TF:
            gpus = tf.config.list_physical_devices('GPU')
            info['gpu']['count'] = len(gpus)
            for i, gpu in enumerate(gpus):
                info['gpu'][f'gpu{i}'] = {'name': gpu.name}
        
        return info
    
    def _get_cpu_name(self):
        """獲取 CPU 名稱"""
        try:
            if os.name == 'nt':  # Windows
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                return winreg.QueryValueEx(key, "ProcessorNameString")[0]
            else:  # Linux/Unix
                with open('/proc/cpuinfo') as f:
                    for line in f:
                        if line.startswith('model name'):
                            return line.split(':')[1].strip()
        except:
            pass
        return "Unknown CPU"
    
    def _save_results(self):
        """保存測試結果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"system_benchmark_{timestamp}.json"
        
        with results_file.open('w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"已保存測試結果到 {results_file}")


# 命令行界面
if __name__ == "__main__":
    import argparse
    
    # 設置命令行界面
    parser = argparse.ArgumentParser(description="性能基準測試工具")
    parser.add_argument("--test", choices=['all', 'cpu', 'memory', 'disk', 'network'],
                       default='all', help="要運行的測試")
    parser.add_argument("--output", type=str, default='benchmark_results',
                       help="結果保存目錄")
    parser.add_argument("--monitor", action="store_true",
                       help="是否顯示性能監視器")
    
    args = parser.parse_args()
    
    # 創建結果目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建並啟動監視器
    monitor = None
    if args.monitor:
        monitor = PerformanceMonitor(interval=0.5)
        monitor.start()
    
    try:
        # 根據選項運行測試
        if args.test == 'all':
            benchmark = SystemBenchmark(results_dir=args.output)
            benchmark.run_all()
        elif args.test == 'cpu':
            cpu_stress_test(seconds=5)
        elif args.test == 'memory':
            memory_stress_test(size_gb=0.5, seconds=2)
        elif args.test == 'disk':
            disk_benchmark(file_size_mb=100)
        elif args.test == 'network':
            network_benchmark(num_requests=3)
    
    finally:
        # 如果啟動了監視器，停止並保存結果
        if monitor:
            monitor.stop()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            monitor.save_results(output_dir / f"monitor_{timestamp}.csv")
            monitor.plot_results(output_dir / f"monitor_{timestamp}.png")
    
    logger.info("測試完成")