#utils/filesystem.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文件系統工具模組 - 電池老化預測系統 (優化版)"""

import time
import hashlib
import threading
import json
import pickle
import gzip
import logging
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar, Generator
from pathlib import Path
from functools import wraps, lru_cache
from contextlib import contextmanager

import numpy as np
import pandas as pd
import h5py
from concurrent.futures import ThreadPoolExecutor, as_completed

# 有條件導入
try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# 設置日誌
logger = logging.getLogger(__name__)

# 類型與默認值
T = TypeVar('T')
PathLike = Union[str, Path]
DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1 MB
DEFAULT_CACHE_DIR = Path("cache")
DEFAULT_CACHE_EXPIRY = 24 * 3600  # 24小時
DEFAULT_COMPRESSION_LEVEL = 5
DEFAULT_MAX_WORKERS = 4

def ensure_path(path: PathLike) -> Path:
    """確保輸入是Path物件"""
    return Path(path) if not isinstance(path, Path) else path


class FileProcessingPipeline:
    """文件處理管道，用於鏈式處理文件操作"""
    
    def __init__(self, input_path: PathLike):
        self.input_path = ensure_path(input_path)
        self.operations = []
        self.result = None
        
    def add_operation(self, operation: Callable, *args, **kwargs) -> 'FileProcessingPipeline':
        """添加操作到管道"""
        self.operations.append((operation, args, kwargs))
        return self
    
    def process(self) -> Any:
        """執行管道中的所有操作"""
        current = self.input_path
        for op, args, kwargs in self.operations:
            current = op(current, *args, **kwargs)
        self.result = current
        return self.result
    
    def cache(self, cache_dir: PathLike = DEFAULT_CACHE_DIR, 
             expiry: int = DEFAULT_CACHE_EXPIRY) -> 'FileProcessingPipeline':
        """將結果緩存到文件系統"""
        def cache_operation(data):
            cache_path = ensure_path(cache_dir) / f"{self.input_path.stem}_cache"
            file_manager.save_to_cache(data, cache_path)
            return data
        return self.add_operation(cache_operation)
    
    def transform(self, func: Callable[[Any], T]) -> 'FileProcessingPipeline':
        """應用轉換函數"""
        return self.add_operation(func)
    
    def save(self, output_path: PathLike, **kwargs) -> 'FileProcessingPipeline':
        """保存處理結果"""
        def save_operation(data):
            output = ensure_path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            
            # 根據擴展名決定保存方式
            ext = output.suffix.lower()
            if ext == '.csv':
                pd.DataFrame(data).to_csv(output, **kwargs)
            elif ext == '.json':
                with output.open('w', encoding='utf-8') as f:
                    json.dump(data, f, **kwargs)
            elif ext == '.pkl':
                with output.open('wb') as f:
                    pickle.dump(data, f, **kwargs)
            elif ext == '.npy':
                np.save(output, data)
            elif ext == '.h5':
                with h5py.File(output, 'w') as f:
                    if isinstance(data, dict):
                        for k, v in data.items():
                            f.create_dataset(k, data=v)
                    else:
                        f.create_dataset('data', data=data)
            else:
                with output.open('wb' if 'b' in kwargs.get('mode', 'w') else 'w') as f:
                    f.write(data)
            
            return data
        
        return self.add_operation(save_operation)


class FileManager:
    """文件管理器，提供文件操作和緩存管理功能"""
    
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, cache_expiry=DEFAULT_CACHE_EXPIRY,
                 max_cache_size_gb=20.0, max_workers=DEFAULT_MAX_WORKERS):
        """初始化文件管理器"""
        self.cache_dir = ensure_path(cache_dir)
        self.cache_expiry = cache_expiry
        self.max_cache_size_gb = max_cache_size_gb
        self.max_workers = max_workers
        
        # 初始化管理對象
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 自動清理過期緩存
        self.cleanup_cache()
    
    def pipeline(self, input_path: PathLike) -> FileProcessingPipeline:
        """創建一個新的文件處理管道"""
        return FileProcessingPipeline(input_path)
    
    @lru_cache(maxsize=128)
    def get_file_hash(self, file_path: PathLike, algorithm: str = "md5") -> str:
        """計算文件的雜湊值（帶緩存）"""
        file_path = ensure_path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 選擇雜湊算法
        hasher = {
            "md5": hashlib.md5(),
            "sha1": hashlib.sha1(),
            "sha256": hashlib.sha256()
        }.get(algorithm)
        
        if not hasher:
            raise ValueError(f"不支援的雜湊算法: {algorithm}")
        
        # 分塊計算雜湊值
        with file_path.open('rb') as f:
            for chunk in iter(lambda: f.read(DEFAULT_CHUNK_SIZE), b''):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def get_cache_path(self, original_path: PathLike, prefix: str = "", suffix: str = "") -> Path:
        """獲取緩存文件路徑"""
        original_path = ensure_path(original_path)
        file_hash = self.get_file_hash(original_path)
        stem, ext = original_path.stem, original_path.suffix
        
        # 構建緩存文件名
        cache_name = f"{prefix}{stem}_{file_hash[:8]}{suffix}{ext}"
        return self.cache_dir / cache_name
    
    def file_is_cached(self, cache_path: PathLike, original_path: Optional[PathLike] = None) -> bool:
        """檢查文件是否已緩存"""
        cache_path = ensure_path(cache_path)
        
        # 基本檢查
        if not cache_path.exists():
            return False
        
        # 檢查是否過期
        cache_mtime = cache_path.stat().st_mtime
        if time.time() - cache_mtime > self.cache_expiry:
            return False
        
        # 檢查原始文件是否更新
        if original_path and ensure_path(original_path).exists():
            original_mtime = ensure_path(original_path).stat().st_mtime
            if original_mtime > cache_mtime:
                return False
        
        return True
    
    def cleanup_cache(self, force: bool = False, min_age_hours: Optional[int] = None) -> int:
        """清理過期的緩存文件"""
        # 使用鎖確保多線程安全
        with self._cleanup_lock:
            try:
                deleted_count = 0
                current_time = time.time()
                
                # 確定最小文件年齡
                min_age = 0
                if min_age_hours is not None:
                    min_age = min_age_hours * 3600
                elif not force:
                    min_age = self.cache_expiry
                
                # 獲取所有緩存文件信息
                cache_files = []
                total_size = 0
                
                for file_path in self.cache_dir.iterdir():
                    if file_path.is_dir():
                        continue
                    
                    try:
                        file_stat = file_path.stat()
                        cache_files.append((file_path, current_time - file_stat.st_mtime, file_stat.st_size))
                        total_size += file_stat.st_size
                    except OSError:
                        continue
                
                # 轉換為 GB
                total_size_gb = total_size / (1024**3)
                
                # 按年齡排序（最舊的在前）
                cache_files.sort(key=lambda x: x[1], reverse=True)
                
                # 刪除過期文件
                for file_path, file_age, file_size in cache_files:
                    if file_age > min_age or total_size_gb > self.max_cache_size_gb:
                        try:
                            file_path.unlink()
                            deleted_count += 1
                            total_size_gb -= file_size / (1024**3)
                            logger.debug(f"已刪除緩存文件: {file_path.name}")
                        except OSError as e:
                            logger.warning(f"無法刪除緩存文件 {file_path}: {e}")
                
                logger.info(f"緩存清理完成，刪除了 {deleted_count} 個文件")
                return deleted_count
                
            except Exception as e:
                logger.error(f"清理緩存時出錯: {e}")
                return 0

    @contextmanager
    def cached_operation(self, original_path: PathLike, operation_name: str, 
                        format: str = "pickle", compression: bool = False):
        """緩存操作的上下文管理器"""
        original_path = ensure_path(original_path)
        cache_path = self.get_cache_path(original_path, prefix=f"{operation_name}_")
        
        cache_hit = self.file_is_cached(cache_path, original_path)
        
        try:
            yield cache_hit, cache_path
        finally:
            pass
    
    def save_to_cache(self, data: Any, cache_path: PathLike, 
                      compression: bool = False, format: str = "pickle") -> bool:
        """保存數據到緩存"""
        cache_path = ensure_path(cache_path)
        try:
            # 確保緩存目錄存在
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根據格式保存數據
            if format == "pickle":
                with (gzip.open(cache_path, 'wb', compresslevel=DEFAULT_COMPRESSION_LEVEL) if compression 
                      else cache_path.open('wb')) as f:
                    pickle.dump(data, f)
            
            elif format == "json":
                with (gzip.open(cache_path, 'wt', compresslevel=DEFAULT_COMPRESSION_LEVEL) if compression 
                      else cache_path.open('w', encoding='utf-8')) as f:
                    json.dump(data, f, ensure_ascii=False)
            
            elif format == "numpy":
                if compression:
                    np.savez_compressed(cache_path, data=data)
                else:
                    np.save(cache_path, data)
            
            else:
                raise ValueError(f"不支援的數據格式: {format}")
            
            logger.debug(f"數據已保存到緩存: {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存數據到緩存時出錯: {e}")
            # 清理部分文件
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except:
                    pass
            return False
    
    def load_from_cache(self, cache_path: PathLike, 
                       compression: bool = False, format: str = "pickle") -> Any:
        """從緩存載入數據"""
        cache_path = ensure_path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(f"緩存文件不存在: {cache_path}")
        
        try:
            # 根據格式載入數據
            if format == "pickle":
                with (gzip.open(cache_path, 'rb') if compression else cache_path.open('rb')) as f:
                    return pickle.load(f)
            
            elif format == "json":
                with (gzip.open(cache_path, 'rt') if compression else cache_path.open('r', encoding='utf-8')) as f:
                    return json.load(f)
            
            elif format == "numpy":
                return np.load(cache_path)['data'] if compression else np.load(cache_path)
            
            else:
                raise ValueError(f"不支援的數據格式: {format}")
            
        except Exception as e:
            logger.error(f"從緩存載入數據時出錯: {e}")
            raise
    
    def read_in_chunks(self, file_path: PathLike, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Generator[bytes, None, None]:
        """分塊讀取大文件，使用生成器避免一次載入整個文件"""
        with ensure_path(file_path).open('rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk
    
    def parallel_file_operation(self, file_paths: List[PathLike], operation: Callable[[Path], T]) -> List[T]:
        """並行執行文件操作"""
        futures = []
        results = []
        
        # 提交任務並收集結果
        for path in file_paths:
            futures.append(self._executor.submit(operation, ensure_path(path)))
        
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"並行操作失敗: {e}")
                results.append(None)
        
        return results
    
    def convert_file_format(self, input_path: PathLike, output_path: PathLike, 
                           input_format: Optional[str] = None, 
                           output_format: Optional[str] = None) -> bool:
        """轉換文件格式"""
        input_path = ensure_path(input_path)
        output_path = ensure_path(output_path)
        
        try:
            # 從文件擴展名推斷格式
            if input_format is None:
                input_format = input_path.suffix.lstrip('.').lower()
            
            if output_format is None:
                output_format = output_path.suffix.lstrip('.').lower()
            
            # 確保輸出目錄存在
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 使用策略模式處理不同格式轉換
            conversion_map = {
                ('csv', 'parquet'): self._convert_csv_to_parquet,
                ('parquet', 'csv'): self._convert_parquet_to_csv,
                ('parquet', 'tfrecord'): self._convert_parquet_to_tfrecord,
                ('h5', 'parquet'): self._convert_h5_to_parquet,
                ('parquet', 'h5'): self._convert_parquet_to_h5,
            }
            
            # 執行轉換
            key = (input_format, output_format)
            if key in conversion_map:
                return conversion_map[key](input_path, output_path)
            else:
                logger.warning(f"不支援的格式轉換: {input_format} -> {output_format}")
                return False
                
        except Exception as e:
            logger.error(f"文件格式轉換失敗: {e}")
            return False
    
    def _convert_csv_to_parquet(self, input_path: Path, output_path: Path) -> bool:
        """將 CSV 文件轉換為 Parquet 格式"""
        try:
            pd.read_csv(input_path).to_parquet(output_path)
            return True
        except Exception as e:
            logger.error(f"CSV 到 Parquet 轉換失敗: {e}")
            return False
    
    def _convert_parquet_to_csv(self, input_path: Path, output_path: Path) -> bool:
        """將 Parquet 文件轉換為 CSV 格式"""
        if not HAS_PYARROW:
            raise ImportError("需要安裝 pyarrow 庫以支援 Parquet 轉換")
        
        try:
            pd.read_parquet(input_path).to_csv(output_path, index=False)
            return True
        except Exception as e:
            logger.error(f"Parquet 到 CSV 轉換失敗: {e}")
            return False
    
    def _convert_h5_to_parquet(self, input_path: Path, output_path: Path) -> bool:
        """將 HDF5 文件轉換為 Parquet 格式"""
        try:
            with h5py.File(input_path, 'r') as h5f:
                # 收集並處理數據集
                datasets = {}
                h5f.visititems(lambda name, obj: 
                               setattr(datasets, name, obj[:]) if isinstance(obj, h5py.Dataset) else None)
                
                # 轉換為 DataFrame
                df = pd.DataFrame({name: data for name, data in datasets.items() 
                                  if len(data.shape) == 1})
                df.to_parquet(output_path)
                return True
        except Exception as e:
            logger.error(f"HDF5 到 Parquet 轉換失敗: {e}")
            return False
    
    def _convert_parquet_to_h5(self, input_path: Path, output_path: Path) -> bool:
        """將 Parquet 文件轉換為 HDF5 格式"""
        if not HAS_PYARROW:
            raise ImportError("需要安裝 pyarrow 庫以支援 Parquet 轉換")
        
        try:
            df = pd.read_parquet(input_path)
            with h5py.File(output_path, 'w') as h5f:
                for column in df.columns:
                    h5f.create_dataset(column, data=df[column].values)
            return True
        except Exception as e:
            logger.error(f"Parquet 到 HDF5 轉換失敗: {e}")
            return False
    
    def _convert_parquet_to_tfrecord(self, input_path: Path, output_path: Path) -> bool:
        """將 Parquet 文件轉換為 TFRecord 格式"""
        if not HAS_TF or not HAS_PYARROW:
            raise ImportError("需要安裝 tensorflow 和 pyarrow 庫以支援此轉換")
        
        try:
            # 讀取 Parquet 文件
            parquet_file = pq.ParquetFile(input_path)
            
            # 創建 TFRecord 寫入器
            with tf.io.TFRecordWriter(str(output_path)) as writer:
                # 分批讀取並處理
                for batch in parquet_file.iter_batches():
                    df_batch = batch.to_pandas()
                    
                    for _, row in df_batch.iterrows():
                        # 創建特徵字典
                        feature = {}
                        
                        for col, value in row.items():
                            # 根據數據類型創建不同的特徵
                            if pd.api.types.is_numeric_dtype(type(value)):
                                if pd.api.types.is_integer_dtype(type(value)):
                                    feature[col] = tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=[value])
                                    )
                                else:
                                    feature[col] = tf.train.Feature(
                                        float_list=tf.train.FloatList(value=[value])
                                    )
                            elif pd.api.types.is_string_dtype(type(value)):
                                feature[col] = tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[str(value).encode('utf-8')])
                                )
                            elif isinstance(value, (list, np.ndarray)):
                                if value.dtype.kind in 'iub':  # 整數類型
                                    feature[col] = tf.train.Feature(
                                        int64_list=tf.train.Int64List(value=value)
                                    )
                                else:  # 浮點類型
                                    feature[col] = tf.train.Feature(
                                        float_list=tf.train.FloatList(value=value)
                                    )
                        
                        # 創建示例並寫入
                        writer.write(tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString())
            
            logger.info(f"已將 {input_path} 轉換為 TFRecord 格式: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Parquet 到 TFRecord 轉換失敗: {e}")
            return False


# 創建全局文件管理器實例
file_manager = FileManager()


# 便捷函數
def create_pipeline(input_path: PathLike) -> FileProcessingPipeline:
    """創建文件處理管道"""
    return file_manager.pipeline(input_path)


def cached(operation_name: str, format: str = "pickle", compression: bool = False):
    """緩存裝飾器，用於緩存函數結果"""
    def decorator(func):
        @wraps(func)
        def wrapper(file_path, *args, **kwargs):
            file_path = ensure_path(file_path)
            cache_path = file_manager.get_cache_path(file_path, prefix=f"{operation_name}_")
            
            # 檢查緩存並執行操作
            if file_manager.file_is_cached(cache_path, file_path):
                logger.debug(f"從緩存載入 {operation_name} 結果")
                return file_manager.load_from_cache(cache_path, compression, format)
            
            result = func(file_path, *args, **kwargs)
            file_manager.save_to_cache(result, cache_path, compression, format)
            
            return result
        return wrapper
    return decorator


def save_to_cache(data: Any, cache_name: str, prefix: str = "", 
                compression: bool = False, format: str = "pickle") -> Path:
    """保存數據到緩存的便捷函數"""
    cache_path = file_manager.cache_dir / f"{prefix}{cache_name}"
    file_manager.save_to_cache(data, cache_path, compression, format)
    return cache_path


def load_from_cache(cache_name: str, prefix: str = "",
                   compression: bool = False, format: str = "pickle") -> Any:
    """從緩存載入數據的便捷函數"""
    cache_path = file_manager.cache_dir / f"{prefix}{cache_name}"
    return file_manager.load_from_cache(cache_path, compression, format)


def cleanup_cache(force: bool = False, min_age_hours: Optional[int] = None) -> int:
    """清理緩存的便捷函數"""
    return file_manager.cleanup_cache(force, min_age_hours)


def convert_file_format(input_path: PathLike, output_path: PathLike, 
                       input_format: Optional[str] = None, 
                       output_format: Optional[str] = None) -> bool:
    """轉換文件格式的便捷函數"""
    return file_manager.convert_file_format(input_path, output_path, input_format, output_format)


def is_file_newer(file_path: PathLike, reference_path: PathLike) -> bool:
    """檢查文件是否比參考文件更新"""
    file_path, reference_path = ensure_path(file_path), ensure_path(reference_path)
    
    if not file_path.exists() or not reference_path.exists():
        return False
    
    return file_path.stat().st_mtime > reference_path.stat().st_mtime


def parallel_process(file_list: List[PathLike], process_func: Callable[[Path], Any], 
                    max_workers: int = DEFAULT_MAX_WORKERS) -> List[Any]:
    """並行處理多個文件"""
    return file_manager.parallel_file_operation(file_list, process_func)


# 命令行界面
if __name__ == "__main__":
    import argparse
    
    # 設置命令行界面
    parser = argparse.ArgumentParser(description="文件系統工具")
    parser.add_argument("--cleanup", action="store_true", help="清理緩存")
    parser.add_argument("--max-age", type=int, default=24, help="清理時的最大文件年齡（小時）")
    parser.add_argument("--convert", action="store_true", help="轉換文件格式")
    parser.add_argument("--input", type=str, help="輸入文件")
    parser.add_argument("--output", type=str, help="輸出文件")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="緩存目錄")
    
    args = parser.parse_args()
    
    # 配置日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 創建文件管理器
    manager = FileManager(cache_dir=args.cache_dir)
    
    # 處理命令
    if args.cleanup:
        deleted = manager.cleanup_cache(min_age_hours=args.max_age)
        print(f"清理完成，刪除了 {deleted} 個緩存文件")
    
    elif args.convert:
        if not args.input or not args.output:
            parser.error("轉換模式需要指定 --input 和 --output 參數")
        
        success = manager.convert_file_format(args.input, args.output)
        if success:
            print(f"成功將 {args.input} 轉換為 {args.output}")
        else:
            print(f"轉換失敗")
    
    else:
        parser.print_help()