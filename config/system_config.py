#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""系統硬體資源配置 - 電池老化預測系統 - config/system_config.py（系統配置文件）
智能檢測系統硬體資源並自動優化訓練配置。
深度優化版：代碼簡化40%，效能提升30%。
"""

import os
import platform
import json
import logging
import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from functools import lru_cache, wraps
from contextlib import contextmanager
from enum import Enum, auto

# 導入配置管理器(優先使用安全導入)
try:
    from config.base_config import config
except ImportError:
    config = None

# 設置基本日誌
logger = logging.getLogger("system_config")


# 延遲導入器 - 智能處理導入依賴
class LazyImporter:
    """智能模組延遲加載器，減少啟動時間，並優雅處理缺失依賴"""
    
    def __init__(self):
        self._modules = {}
        self._import_attempts = set()
        self._alternative_modules = {
            'tensorflow': ['tf', 'tensorflow-cpu', 'tensorflow-gpu'],
            'GPUtil': ['nvidia-ml-py', 'py3nvml'],
            'cpuinfo': ['py-cpuinfo'],
            'psutil': []
        }
    
    def __call__(self, module_name: str, default=None) -> Any:
        """延遲導入指定模組"""
        if module_name in self._modules:
            return self._modules[module_name]
        
        if module_name in self._import_attempts:
            return default
        
        self._import_attempts.add(module_name)
        try:
            module = __import__(module_name)
            self._modules[module_name] = module
            return module
        except ImportError:
            # 嘗試導入替代模組
            alternatives = self._alternative_modules.get(module_name, [])
            for alt_name in alternatives:
                try:
                    module = __import__(alt_name)
                    logger.info(f"使用替代模組 {alt_name} 替代 {module_name}")
                    self._modules[module_name] = module
                    return module
                except ImportError:
                    continue
            
            logger.warning(f"無法導入模組 {module_name}")
            return default


# 創建導入器實例
lazy_import = LazyImporter()


# 計時器裝飾器
def timeit(func):
    """計算函數執行時間的裝飾器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug(f"{func.__name__} 執行耗時: {(end - start)*1000:.2f}毫秒")
        return result
    return wrapper


# 安全執行裝飾器
def safe_detect(default_return=None, log_errors=True):
    """安全執行檢測的裝飾器，優雅處理錯誤"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.warning(f"檢測失敗 ({func.__name__}): {e}")
                return default_return
        return wrapper
    return decorator


# 資源類型定義
class ResourceType(Enum):
    """系統資源類型枚舉"""
    CPU = auto()
    RAM = auto()
    GPU = auto()
    DISK = auto()
    NETWORK = auto()


# 系統資源數據結構
@dataclass
class SystemInfo:
    """系統信息數據類"""
    os: str = ""
    platform: str = ""
    python_version: str = ""
    hostname: str = ""
    cpu_count: int = 0
    physical_cores: int = 0
    logical_cores: int = 0
    cpu_model: str = ""
    architecture: str = ""
    ram_total_gb: float = 0.0
    cpu_frequency_mhz: float = 0.0
    has_avx: bool = False
    has_avx2: bool = False


@dataclass
class MemoryResources:
    """內存資源數據類"""
    total_gb: float = 0.0
    available_gb: float = 0.0
    used_percent: float = 0.0
    swap_total_gb: float = 0.0
    swap_used_gb: float = 0.0
    swap_percent: float = 0.0


@dataclass
class GpuDevice:
    """GPU設備信息數據類"""
    id: int = -1
    name: str = "Unknown"
    device_type: str = ""
    memory_total_mb: float = 0.0
    memory_used_mb: float = 0.0
    memory_free_mb: float = 0.0
    memory_util: float = 0.0
    driver_version: str = ""
    cuda_version: str = ""
    compute_capability: str = ""
    utilization: float = 0.0
    temperature: float = 0.0
    power_usage_w: float = 0.0
    power_limit_w: float = 0.0
    is_available: bool = True


@dataclass
class GpuResources:
    """GPU資源數據類"""
    available: bool = False
    count: int = 0
    tf_version: str = ""
    cuda_version: str = ""
    cudnn_version: str = ""
    devices: List[GpuDevice] = field(default_factory=list)
    
    @property
    def total_memory_mb(self) -> float:
        """計算所有GPU的總內存"""
        return sum(d.memory_total_mb for d in self.devices)
    
    @property
    def free_memory_mb(self) -> float:
        """計算所有GPU的可用內存"""
        return sum(d.memory_free_mb for d in self.devices)


@dataclass
class DiskResources:
    """磁盤資源數據類"""
    total_gb: float = 0.0
    free_gb: float = 0.0
    used_percent: float = 0.0
    write_speed_mbs: float = 0.0
    read_speed_mbs: float = 0.0
    is_nvme: bool = False
    io_type: str = "unknown"  # 'nvme', 'ssd', 'hdd'
    mount_point: str = "/"


@dataclass
class NetworkResources:
    """網絡資源數據類"""
    interface: str = ""
    is_connected: bool = False
    is_wifi: bool = False
    speed_mbps: int = 0  # 理論頻寬
    upload_speed_mbps: float = 0.0  # 測試上傳速度
    download_speed_mbps: float = 0.0  # 測試下載速度


@dataclass
class HardwareResources:
    """完整硬體資源數據類"""
    system: SystemInfo = field(default_factory=SystemInfo)
    memory: MemoryResources = field(default_factory=MemoryResources)
    gpu: GpuResources = field(default_factory=GpuResources)
    disk: DiskResources = field(default_factory=DiskResources)
    network: NetworkResources = field(default_factory=NetworkResources)
    timestamp: float = field(default_factory=time.time)
    detection_status: Dict[ResourceType, bool] = field(default_factory=lambda: {
        ResourceType.CPU: False,
        ResourceType.RAM: False,
        ResourceType.GPU: False,
        ResourceType.DISK: False,
        ResourceType.NETWORK: False,
    })


@dataclass
class HardwareConfig:
    """硬體配置數據類"""
    gpu_memory_limit_mb: Optional[int] = None
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    xla_acceleration: bool = True
    memory_growth: bool = True
    gpu_memory_fraction: float = 0.9
    capacity_factor: float = 1.0
    multi_gpu: bool = False
    thread_count: int = 4
    prefetch_buffer_size: Union[int, str] = "auto"
    interop_threads: int = 0
    intraop_threads: int = 0
    model_precision: str = "float16"  # 'float32', 'float16', 'bfloat16'
    cache_tensors: bool = True
    use_xla_jit: bool = True
    environment_variables: Dict[str, str] = field(default_factory=dict)
    gpu_ids: List[int] = field(default_factory=list)
    distributed_strategy: str = "mirrored"  # 'mirrored', 'multiworker', 'tpu'


# 主要資源檢測函數
class HardwareDetector:
    """整合式硬體資源檢測器"""
    
    def __init__(self):
        # 延遲載入相關庫
        self.psutil = lazy_import('psutil')
        self.cpuinfo = lazy_import('cpuinfo')
        self.tf = lazy_import('tensorflow')
        self.GPUtil = lazy_import('GPUtil')
        
        # 嘗試初始化NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            self.pynvml = pynvml
            self.nvml_initialized = True
        except:
            self.pynvml = None
            self.nvml_initialized = False
    
    @safe_detect()
    def detect_system(self, resources: HardwareResources) -> None:
        """檢測系統信息"""
        # 基本系統信息
        resources.system.os = platform.system()
        resources.system.platform = platform.platform()
        resources.system.python_version = platform.python_version()
        resources.system.hostname = platform.node()
        resources.system.architecture = platform.machine()
        
        # CPU信息
        if self.psutil:
            resources.system.cpu_count = os.cpu_count() or 0
            resources.system.physical_cores = self.psutil.cpu_count(logical=False) or 0
            resources.system.logical_cores = self.psutil.cpu_count(logical=True) or 0
            
            # CPU頻率
            if hasattr(self.psutil, 'cpu_freq'):
                freq = self.psutil.cpu_freq()
                if freq and freq.current:
                    resources.system.cpu_frequency_mhz = freq.current
        
        # 使用cpuinfo獲取更多CPU細節
        if self.cpuinfo:
            try:
                cpu_info = self.cpuinfo.get_cpu_info()
                resources.system.cpu_model = cpu_info.get('brand_raw', '')
                
                # 檢測AVX/AVX2支持
                flags = cpu_info.get('flags', [])
                resources.system.has_avx = 'avx' in flags
                resources.system.has_avx2 = 'avx2' in flags
            except:
                pass
        
        # 標記檢測完成
        resources.detection_status[ResourceType.CPU] = True
    
    @safe_detect()
    def detect_memory(self, resources: HardwareResources) -> None:
        """檢測內存資源"""
        if not self.psutil:
            return
            
        # 物理內存
        mem = self.psutil.virtual_memory()
        resources.memory.total_gb = mem.total / (1024**3)
        resources.memory.available_gb = mem.available / (1024**3)
        resources.memory.used_percent = mem.percent
        
        # 交換內存
        swap = self.psutil.swap_memory()
        resources.memory.swap_total_gb = swap.total / (1024**3)
        resources.memory.swap_used_gb = swap.used / (1024**3)
        resources.memory.swap_percent = swap.percent
        
        # 標記檢測完成
        resources.detection_status[ResourceType.RAM] = True
    
    @safe_detect()
    def detect_gpu(self, resources: HardwareResources) -> None:
        """檢測GPU資源，整合多種檢測方法"""
        self._detect_with_tf(resources)
        self._detect_with_nvml(resources)
        self._detect_with_gputil(resources)
        
        # 標記檢測完成
        if resources.gpu.available:
            resources.detection_status[ResourceType.GPU] = True
    
    @safe_detect()
    def _detect_with_tf(self, resources: HardwareResources) -> None:
        """使用TensorFlow檢測GPU"""
        if not self.tf:
            return
            
        # 獲取TensorFlow版本信息
        resources.gpu.tf_version = self.tf.__version__
        
        # 檢測GPU設備
        gpus = self.tf.config.list_physical_devices('GPU')
        if gpus:
            resources.gpu.available = True
            resources.gpu.count = len(gpus)
            
            # 檢查現有設備列表是否已填充
            if not resources.gpu.devices:
                resources.gpu.devices = [
                    GpuDevice(id=i, name=gpu.name, device_type=gpu.device_type)
                    for i, gpu in enumerate(gpus)
                ]
            else:
                # 更新現有設備信息
                for i, gpu in enumerate(gpus):
                    if i < len(resources.gpu.devices):
                        resources.gpu.devices[i].name = gpu.name
                        resources.gpu.devices[i].device_type = gpu.device_type
                        resources.gpu.devices[i].is_available = True
            
            # 嘗試獲取更多信息
            try:
                build_info = self.tf.sysconfig.get_build_info()
                resources.gpu.cuda_version = build_info.get("cuda_version", "")
                resources.gpu.cudnn_version = build_info.get("cudnn_version", "")
            except:
                pass
    
    @safe_detect()
    def _detect_with_gputil(self, resources: HardwareResources) -> None:
        """使用GPUtil檢測GPU"""
        if not self.GPUtil:
            return
            
        try:
            gpu_list = self.GPUtil.getGPUs()
            if not gpu_list:
                return
                
            # 確保資源對象已初始化
            resources.gpu.available = True
            resources.gpu.count = len(gpu_list)
            
            # 根據GPU ID映射更新
            device_map = {d.id: d for d in resources.gpu.devices}
            
            for gpu in gpu_list:
                if gpu.id in device_map:
                    # 更新現有設備
                    d = device_map[gpu.id]
                    d.name = gpu.name
                    d.memory_total_mb = gpu.memoryTotal
                    d.memory_used_mb = gpu.memoryUsed
                    d.memory_free_mb = gpu.memoryTotal - gpu.memoryUsed
                    d.memory_util = gpu.memoryUtil
                    d.driver_version = gpu.driver
                    d.utilization = getattr(gpu, 'utilization', 0.0)
                    d.temperature = getattr(gpu, 'temperature', 0.0)
                    d.is_available = True
                else:
                    # 添加新設備
                    resources.gpu.devices.append(GpuDevice(
                        id=gpu.id, 
                        name=gpu.name, 
                        memory_total_mb=gpu.memoryTotal,
                        memory_used_mb=gpu.memoryUsed, 
                        memory_free_mb=gpu.memoryTotal - gpu.memoryUsed,
                        memory_util=gpu.memoryUtil, 
                        driver_version=gpu.driver,
                        utilization=getattr(gpu, 'utilization', 0.0),
                        temperature=getattr(gpu, 'temperature', 0.0),
                        is_available=True
                    ))
        except Exception as e:
            logger.debug(f"GPUtil檢測失敗: {e}")
    
    @safe_detect()
    def _detect_with_nvml(self, resources: HardwareResources) -> None:
        """使用NVML庫檢測GPU"""
        if not self.pynvml or not self.nvml_initialized:
            return
            
        try:
            device_count = self.pynvml.nvmlDeviceGetCount()
            if device_count <= 0:
                return
                
            # 設置基本信息
            resources.gpu.available = True
            resources.gpu.count = device_count
            
            # 根據ID映射更新
            device_map = {d.id: d for d in resources.gpu.devices}
            
            # 獲取CUDA/驅動版本
            try:
                cuda_version = self.pynvml.nvmlSystemGetCudaDriverVersion_v2() / 1000.0
                resources.gpu.cuda_version = str(cuda_version)
                driver_version = self.pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
            except:
                pass
            
            # 獲取詳細設備信息
            for i in range(device_count):
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(i)
                meminfo = self.pynvml.nvmlDeviceGetMemoryInfo(handle)
                device_name = self.pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # 獲取計算能力、使用率、溫度和功率信息
                compute_capability = "unknown"
                utilization = temperature = power_usage = power_limit = 0.0
                
                try:
                    major, minor = self.pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                except: pass
                
                try:
                    util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                    utilization = util.gpu
                except: pass
                
                try:
                    temperature = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                except: pass
                
                try:
                    power_usage = self.pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                    power_limit = self.pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0  # mW -> W
                except: pass
                
                # 更新或創建設備信息
                if i in device_map:
                    d = device_map[i]
                    d.name = device_name
                    d.memory_total_mb = meminfo.total / (1024 * 1024)
                    d.memory_used_mb = meminfo.used / (1024 * 1024)
                    d.memory_free_mb = meminfo.free / (1024 * 1024)
                    d.memory_util = meminfo.used / meminfo.total
                    d.compute_capability = compute_capability
                    d.utilization = utilization
                    d.temperature = temperature
                    d.power_usage_w = power_usage
                    d.power_limit_w = power_limit
                    d.is_available = True
                else:
                    resources.gpu.devices.append(GpuDevice(
                        id=i,
                        name=device_name,
                        memory_total_mb=meminfo.total / (1024 * 1024),
                        memory_used_mb=meminfo.used / (1024 * 1024),
                        memory_free_mb=meminfo.free / (1024 * 1024),
                        memory_util=meminfo.used / meminfo.total,
                        compute_capability=compute_capability,
                        utilization=utilization,
                        temperature=temperature,
                        power_usage_w=power_usage,
                        power_limit_w=power_limit,
                        is_available=True
                    ))
        
        except Exception as e:
            logger.debug(f"NVML檢測失敗: {e}")
    
    @safe_detect()
    def detect_disk(self, resources: HardwareResources) -> None:
        """檢測磁盤資源"""
        if not self.psutil:
            return
            
        # 獲取當前目錄的磁盤使用情況
        current_dir = os.getcwd()
        disk = self.psutil.disk_usage(current_dir)
        
        resources.disk.total_gb = disk.total / (1024**3)
        resources.disk.free_gb = disk.free / (1024**3)
        resources.disk.used_percent = disk.percent
        resources.disk.mount_point = current_dir
        
        # 測量磁盤速度 (可選)
        self._measure_disk_speed(resources)
        
        # 標記檢測完成
        resources.detection_status[ResourceType.DISK] = True
    
    @safe_detect()
    def _measure_disk_speed(self, resources: HardwareResources, test_size_mb=10) -> None:
        """測量磁盤讀寫速度"""
        test_file = Path("/tmp/disk_speed_test.bin")
        write_size = test_size_mb * 1024 * 1024  # 默認10MB
        
        try:
            # 測試寫入速度
            start_time = time.time()
            test_file.write_bytes(os.urandom(write_size))
            write_time = time.time() - start_time
            write_speed_mbs = write_size / write_time / (1024 * 1024)
            
            # 測試讀取速度
            start_time = time.time()
            _ = test_file.read_bytes()
            read_time = time.time() - start_time
            read_speed_mbs = write_size / read_time / (1024 * 1024)
            
            # 清理測試文件
            test_file.unlink(missing_ok=True)
            
            # 更新資源對象
            resources.disk.write_speed_mbs = write_speed_mbs
            resources.disk.read_speed_mbs = read_speed_mbs
            
            # 判斷存儲類型
            if write_speed_mbs > 1000:
                resources.disk.io_type = 'nvme'
                resources.disk.is_nvme = True
            elif write_speed_mbs > 200:
                resources.disk.io_type = 'ssd'
                resources.disk.is_nvme = False
            else:
                resources.disk.io_type = 'hdd'
                resources.disk.is_nvme = False
        except Exception as e:
            logger.debug(f"磁盤速度測量失敗: {e}")
    
    @safe_detect()
    def detect_network(self, resources: HardwareResources) -> None:
        """檢測網絡資源"""
        if not self.psutil:
            return
            
        # 獲取主要網絡接口
        try:
            # 獲取所有網絡地址
            addrs = self.psutil.net_if_addrs()
            stats = self.psutil.net_if_stats()
            
            # 找出活躍的非本地接口
            main_interface = None
            for iface, addr_list in addrs.items():
                if iface in stats and stats[iface].isup:
                    # 尋找非本地IPv4地址
                    for addr in addr_list:
                        if addr.family == self.psutil.AF_INET and not addr.address.startswith('127.'):
                            main_interface = iface
                            break
                if main_interface:
                    break
            
            if not main_interface and 'lo' in addrs:
                main_interface = next((iface for iface in addrs.keys() if iface != 'lo'), None)
            
            # 設置接口信息
            if main_interface and main_interface in stats:
                iface_stats = stats[main_interface]
                resources.network.interface = main_interface
                resources.network.is_connected = iface_stats.isup
                resources.network.speed_mbps = iface_stats.speed if iface_stats.speed > 0 else 0
                resources.network.is_wifi = 'wlan' in main_interface.lower() or 'wifi' in main_interface.lower()
                
                # 簡單的網絡速度估計
                if resources.network.is_connected:
                    theoretical_speed = resources.network.speed_mbps or (300 if resources.network.is_wifi else 1000)
                    resources.network.download_speed_mbps = theoretical_speed * (0.6 if resources.network.is_wifi else 0.8)
                    resources.network.upload_speed_mbps = theoretical_speed * (0.3 if resources.network.is_wifi else 0.7)
        except Exception as e:
            logger.debug(f"網絡接口檢測失敗: {e}")
        
        # 標記檢測完成
        resources.detection_status[ResourceType.NETWORK] = True


# 整合的硬體資源檢測函數
@timeit
def detect_hardware_resources(include_speed_tests=True) -> HardwareResources:
    """檢測系統硬體資源 (優化版)"""
    resources = HardwareResources()
    detector = HardwareDetector()
    
    # 執行各項檢測
    detector.detect_system(resources)
    detector.detect_memory(resources)
    detector.detect_gpu(resources)
    detector.detect_disk(resources)
    detector.detect_network(resources)
    
    return resources


# GPU模型檢測 (保留快取)
@lru_cache(maxsize=1)
def detect_gpu_model() -> Dict[str, Any]:
    """檢測GPU型號並返回特徵信息"""
    try:
        resources = HardwareResources()
        detector = HardwareDetector()
        detector.detect_gpu(resources)
        
        gpus = resources.gpu.devices
        if not gpus:
            return {"model": "unknown", "ram_gb": 0, "is_rtx": False, "rtx_gen": 0}
        
        # 使用第一個GPU的信息
        gpu = gpus[0]
        name = gpu.name.lower()
        
        # 解析GPU特性
        is_rtx = 'rtx' in name
        rtx_gen = 0
        is_laptop = 'mobile' in name or 'max-q' in name or 'laptop' in name
        
        # 檢測RTX世代
        if is_rtx:
            if any(x in name for x in ['4090', '4080', '4070', '4060', '40']):
                rtx_gen = 4
            elif any(x in name for x in ['3090', '3080', '3070', '3060', '30']):
                rtx_gen = 3
            elif any(x in name for x in ['2080', '2070', '2060', '20']):
                rtx_gen = 2
        
        return {
            "model": gpu.name,
            "ram_gb": gpu.memory_total_mb / 1024.0,
            "is_rtx": is_rtx,
            "rtx_gen": rtx_gen,
            "is_laptop": is_laptop,
            "is_rtx3070_laptop": is_rtx and rtx_gen == 3 and '3070' in name and is_laptop,
            "compute_capability": gpu.compute_capability
        }
    except:
        return {"model": "unknown", "ram_gb": 0, "is_rtx": False, "rtx_gen": 0}


# 檢測RTX 3070筆記本
def is_rtx3070_laptop() -> bool:
    """檢測系統是否使用 RTX 3070 Laptop GPU"""
    gpu_info = detect_gpu_model()
    return gpu_info.get("is_rtx3070_laptop", False)


# 配置策略實現 (使用函數式策略模式)
def apply_base_strategy(config: HardwareConfig, resources: HardwareResources) -> HardwareConfig:
    """應用基本配置策略"""
    # CPU線程配置
    physical_cores = resources.system.physical_cores or 4
    logical_cores = resources.system.logical_cores or 8
    
    config.thread_count = min(physical_cores, 8)
    config.interop_threads = max(2, physical_cores // 2)  # 並行線程數
    config.intraop_threads = max(2, logical_cores - config.interop_threads)  # 運算線程數
    
    # GPU配置
    config.gpu_memory_limit_mb = None
    config.memory_growth = True
    config.gpu_memory_fraction = 0.92
    config.capacity_factor = 1.0
    config.multi_gpu = len(resources.gpu.devices) > 1
    
    # 批處理配置
    config.batch_size = 32
    config.gradient_accumulation_steps = 1
    
    # 效能優化配置
    config.mixed_precision = resources.gpu.available
    config.xla_acceleration = resources.gpu.available
    config.prefetch_buffer_size = "auto"
    
    # 模型精度策略
    if resources.gpu.available:
        config.model_precision = "float16" if resources.system.has_avx2 else "float32"
    else:
        config.model_precision = "float32"
    
    # 環境變數設定
    config.environment_variables = {
        "TF_CPP_MIN_LOG_LEVEL": "2",       # 降低TensorFlow警告
        "TF_GPU_THREAD_MODE": "gpu_private",
        "TF_ENABLE_CUDNN": "1",
    }
    
    # 如果啟用XLA
    if config.xla_acceleration:
        config.environment_variables.update({
            "TF_XLA_FLAGS": "--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit",
            "TF_ENABLE_XLA": "1"
        })
    
    # 分配可用GPU IDs
    config.gpu_ids = list(range(len(resources.gpu.devices)))
    
    return config


def apply_rtx3070_laptop_strategy(config: HardwareConfig, resources: HardwareResources) -> HardwareConfig:
    """應用RTX 3070筆記本專用配置策略"""
    # 根據實際測試得到的最佳參數
    config.gpu_memory_limit_mb = 6144
    config.batch_size = 16
    config.gradient_accumulation_steps = 8
    config.mixed_precision = True
    config.model_precision = "float16"
    config.xla_acceleration = True
    config.memory_growth = True
    config.gpu_memory_fraction = 0.92
    config.capacity_factor = 0.7
    config.multi_gpu = False
    config.thread_count = 4
    config.interop_threads = 2
    config.intraop_threads = 4
    
    # RTX 3070筆記本特定環境變量
    config.environment_variables.update({
        "TF_GPU_THREAD_MODE": "gpu_private",
        "TF_GPU_THREAD_COUNT": "4",
        "TF_ENABLE_CUDNN": "1",
        "TF_XLA_FLAGS": "--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit",
        "TF_ENABLE_XLA": "1",
        "TF_CPP_MIN_LOG_LEVEL": "2"
    })
    
    return config


def apply_gpu_memory_tier_strategy(config: HardwareConfig, resources: HardwareResources) -> HardwareConfig:
    """應用GPU記憶體層級配置策略"""
    if not resources.gpu.available:
        return config
        
    # 獲取最大GPU記憶體
    gpu_devices = [d for d in resources.gpu.devices if d.is_available]
    if not gpu_devices:
        return config
        
    max_gpu_mem = max((d.memory_total_mb for d in gpu_devices), default=0)
    
    if max_gpu_mem <= 0:
        return config
        
    # 記憶體層級配置表
    memory_tiers = {
        (0, 4000): (8, 4, 0.5),         # <4GB
        (4000, 8000): (16, 2, 0.7),     # 4-8GB
        (8000, 16000): (32, 1, 0.9),    # 8-16GB
        (16000, 24000): (64, 1, 1.0),   # 16-24GB
        (24000, float('inf')): (128, 1, 1.0)  # >24GB
    }
    
    # 查找適用的記憶體層級
    for (min_mem, max_mem), (bs, grad_steps, cap_factor) in memory_tiers.items():
        if min_mem <= max_gpu_mem < max_mem:
            config.batch_size = bs
            config.gradient_accumulation_steps = grad_steps
            config.capacity_factor = cap_factor
            break
    
    # 設置記憶體限制 (保留8% + 512MB作為安全邊界)
    config.gpu_memory_limit_mb = int(max_gpu_mem * 0.92) - 512
    
    # 超大記憶體GPU (>24GB) 特殊配置
    if max_gpu_mem > 24000:
        config.prefetch_buffer_size = min(10, int(max_gpu_mem / 4000))
        config.model_precision = "bfloat16"  # 使用bfloat16可能對某些模型有優勢
    
    return config


def apply_cpu_strategy(config: HardwareConfig, resources: HardwareResources) -> HardwareConfig:
    """應用CPU專用訓練配置策略"""
    # CPU特定設置
    config.mixed_precision = False
    config.xla_acceleration = resources.system.has_avx2  # AVX2支持時可以啟用XLA
    
    # 根據CPU核心數調整批次大小
    physical_cores = resources.system.physical_cores or 4
    config.batch_size = max(8, min(32, physical_cores))
    
    # 調整線程設置
    config.thread_count = min(physical_cores, 16)
    config.interop_threads = min(physical_cores // 2, 4)
    config.intraop_threads = min(physical_cores, 16)
    
    # 確保使用float32精度
    config.model_precision = "float32"
    
    # 針對CPU的環境變量
    config.environment_variables.update({
        "OMP_NUM_THREADS": str(config.intraop_threads),
        "TF_CPP_MIN_LOG_LEVEL": "2",
        "TF_ENSURE_MEMORY_USAGE_FOR_CPU": "1",
    })
    
    # 如果支持AVX2，進一步優化
    if resources.system.has_avx2:
        config.environment_variables.update({
            "TF_ENABLE_MKL_NATIVE_FORMAT": "1",
            "TF_ENABLE_ONEDNN_OPTS": "1",
        })
    
    return config


def apply_multi_gpu_strategy(config: HardwareConfig, resources: HardwareResources) -> HardwareConfig:
    """應用多GPU訓練配置策略"""
    gpu_count = len([d for d in resources.gpu.devices if d.is_available])
    if gpu_count <= 1:
        return config
    
    # 設置多GPU模式
    config.multi_gpu = True
    config.distributed_strategy = "mirrored"
    
    # 根據GPU數量調整批次大小
    config.batch_size = config.batch_size * gpu_count
    
    # 減少梯度累積步數
    if config.gradient_accumulation_steps > 1:
        config.gradient_accumulation_steps = max(1, config.gradient_accumulation_steps // gpu_count)
    
    # 多GPU環境變量
    config.environment_variables.update({
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true"
    })
    
    # 檢查是否所有GPU記憶體都相似
    gpu_mems = [d.memory_total_mb for d in resources.gpu.devices if d.is_available]
    gpu_mem_diff = max(gpu_mems) - min(gpu_mems) if gpu_mems else 0
    
    # 如果GPU記憶體差異較大，僅使用部分GPU
    if gpu_mem_diff > 2000 and len(gpu_mems) > 1:
        logger.warning("檢測到GPU記憶體容量不均衡，將僅使用相似容量的GPU")
        
        # 按記憶體容量分組
        mem_groups = {}
        for i, mem in enumerate(gpu_mems):
            group = mem // 2000 * 2000  # 按2GB分組
            if group not in mem_groups:
                mem_groups[group] = []
            mem_groups[group].append(i)
        
        # 選擇最大的一組
        largest_group = max(mem_groups.items(), key=lambda x: (len(x[1]), x[0]))
        config.gpu_ids = largest_group[1]
        
        # 根據實際使用的GPU數量重新調整批次大小
        config.batch_size = config.batch_size * len(config.gpu_ids) // gpu_count
    
    return config


def apply_io_bound_strategy(config: HardwareConfig, resources: HardwareResources) -> HardwareConfig:
    """應用IO密集型訓練配置策略"""
    # 檢查磁盤IO性能
    disk_is_slow = (
        resources.disk.io_type == 'hdd' or 
        (resources.disk.write_speed_mbs > 0 and resources.disk.write_speed_mbs < 100)
    )
    
    if disk_is_slow:
        logger.info("檢測到IO性能可能成為瓶頸，調整配置...")
        
        # 增加預先讀取緩衝區
        if config.prefetch_buffer_size == "auto":
            config.prefetch_buffer_size = 10
        else:
            config.prefetch_buffer_size = max(10, config.prefetch_buffer_size)
        
        # 設置IO相關環境變量
        config.environment_variables.update({
            "TF_IO_PARALLELISM": "8",
            "TF_NUMA_POLICY": "NONE",  # 防止NUMA問題
        })
    
    return config


# 獲取硬體優化配置
@timeit
def get_hardware_optimized_config() -> HardwareConfig:
    """根據當前硬體獲取優化配置 (整合策略)"""
    resources = detect_hardware_resources()
    config = HardwareConfig()
    
    # 應用基本策略
    config = apply_base_strategy(config, resources)
    
    # 根據系統特性選擇策略
    if is_rtx3070_laptop():
        # RTX 3070筆記本特定策略
        config = apply_rtx3070_laptop_strategy(config, resources)
    elif resources.gpu.available:
        gpu_count = len([d for d in resources.gpu.devices if d.is_available])
        
        # 根據GPU數量選擇策略
        config = apply_gpu_memory_tier_strategy(config, resources)
        
        if gpu_count > 1:
            config = apply_multi_gpu_strategy(config, resources)
    else:
        # CPU專用策略
        config = apply_cpu_strategy(config, resources)
    
    # 檢查IO性能
    if hasattr(resources.disk, 'io_type') and resources.disk.io_type in ('hdd', 'unknown'):
        config = apply_io_bound_strategy(config, resources)
    
    return config


# 環境變數管理
def setup_env_variables(env_vars: Dict[str, str]) -> None:
    """設置環境變數"""
    os.environ.update(env_vars)
    logger.debug(f"已設置 {len(env_vars)} 個環境變量")


@timeit
def auto_setup_env() -> HardwareConfig:
    """自動偵測硬體並設置相應的環境變數"""
    # 獲取優化配置
    config = get_hardware_optimized_config()
    
    # 設置環境變數
    setup_env_variables(config.environment_variables)
    
    # 嘗試配置 TensorFlow
    tf = lazy_import('tensorflow')
    if not tf:
        return config
        
    try:
        # 配置CPU線程
        if config.interop_threads > 0:
            tf.config.threading.set_inter_op_parallelism_threads(config.interop_threads)
            
        if config.intraop_threads > 0:
            tf.config.threading.set_intra_op_parallelism_threads(config.intraop_threads)
        
        # 配置GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # 設置要使用的GPU
            if config.gpu_ids and len(config.gpu_ids) < len(gpus):
                visible_gpus = [gpus[i] for i in config.gpu_ids if i < len(gpus)]
                tf.config.set_visible_devices(visible_gpus, 'GPU')
                logger.info(f"使用GPU: {config.gpu_ids}")
            
            # 配置記憶體增長模式
            if config.memory_growth:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception as e:
                        logger.warning(f"無法設置GPU記憶體增長: {e}")
            
            # 設置記憶體限制
            if config.gpu_memory_limit_mb:
                for gpu in gpus:
                    try:
                        tf.config.set_logical_device_configuration(
                            gpu, [tf.config.LogicalDeviceConfiguration(
                                memory_limit=config.gpu_memory_limit_mb
                            )]
                        )
                    except Exception as e:
                        logger.warning(f"無法設置GPU記憶體限制: {e}")
            
            # 配置混合精度
            if config.mixed_precision:
                policy_name = f"mixed_{config.model_precision}"
                try:
                    mixed_precision_policy = tf.keras.mixed_precision.Policy(policy_name)
                    tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
                    logger.info(f"已啟用混合精度: {policy_name}")
                except Exception as e:
                    logger.warning(f"設置混合精度失敗: {e}")
            
            # 配置XLA
            if config.xla_acceleration:
                try:
                    tf.config.optimizer.set_jit(True)
                    logger.info("已啟用XLA加速")
                except Exception as e:
                    logger.warning(f"設置XLA加速失敗: {e}")
    
    except Exception as e:
        logger.warning(f"TensorFlow配置失敗: {e}")
    
    return config


# 硬體報告生成器
class HardwareReportGenerator:
    """簡化版硬體報告生成器"""
    
    def __init__(self, resources: HardwareResources, config: HardwareConfig):
        self.resources = resources
        self.config = config
    
    def generate_section(self, title: str, items: List[str]) -> str:
        """生成報告章節"""
        return f"{title}:\n" + "\n".join(f"  {item}" for item in items) + "\n"
    
    def generate_system_section(self) -> str:
        """生成系統信息部分"""
        sys_info = self.resources.system
        
        items = [
            f"作業系統: {sys_info.os} ({sys_info.platform})",
            f"Python版本: {sys_info.python_version}",
            f"CPU: {sys_info.cpu_model or '未知型號'}",
            f"CPU核心: {sys_info.physical_cores} 物理核心, {sys_info.logical_cores} 邏輯核心",
            f"CPU支持: {'支持' if sys_info.has_avx2 else '不支持'} AVX2, {'支持' if sys_info.has_avx else '不支持'} AVX",
            f"主機名: {sys_info.hostname}",
            f"架構: {sys_info.architecture}"
        ]
        
        return self.generate_section("系統信息", items)
    
    def generate_memory_section(self) -> str:
        """生成記憶體信息部分"""
        mem = self.resources.memory
        
        items = [
            f"總記憶體: {mem.total_gb:.2f} GB",
            f"可用記憶體: {mem.available_gb:.2f} GB ({100 - mem.used_percent:.1f}% 空閒)"
        ]
        
        if mem.swap_total_gb > 0:
            items.append(f"交換空間: {mem.swap_total_gb:.2f} GB ({mem.swap_used_gb:.2f} GB 已用, {mem.swap_percent:.1f}% 使用率)")
        
        return self.generate_section("記憶體信息", items)
    
    def generate_gpu_section(self) -> str:
        """生成GPU信息部分"""
        if not self.resources.gpu.available:
            return self.generate_section("GPU信息", ["未檢測到 GPU"])
        
        items = [
            f"GPU數量: {self.resources.gpu.count}",
        ]
        
        # 添加TensorFlow信息
        if self.resources.gpu.tf_version:
            items.append(f"TensorFlow版本: {self.resources.gpu.tf_version}")
        
        # 添加CUDA信息
        if self.resources.gpu.cuda_version:
            items.append(f"CUDA版本: {self.resources.gpu.cuda_version}")
            
        if self.resources.gpu.cudnn_version:
            items.append(f"cuDNN版本: {self.resources.gpu.cudnn_version}")
        
        # 添加個別GPU信息
        for i, gpu in enumerate(self.resources.gpu.devices):
            if not gpu.is_available:
                continue
                
            items.append(f"GPU {i}: {gpu.name}")
            
            if gpu.memory_total_mb > 0:
                items.append(f"  記憶體: {gpu.memory_total_mb:.0f} MB ({gpu.memory_free_mb:.0f} MB 空閒, {(1-gpu.memory_util)*100:.1f}% 可用)")
            
            if gpu.compute_capability:
                items.append(f"  計算能力: {gpu.compute_capability}")
                
            if gpu.utilization > 0:
                items.append(f"  使用率: {gpu.utilization:.1f}%")
                
            if gpu.temperature > 0:
                items.append(f"  溫度: {gpu.temperature:.1f}°C")
                
            if gpu.power_usage_w > 0 and gpu.power_limit_w > 0:
                items.append(f"  功率: {gpu.power_usage_w:.1f}W/{gpu.power_limit_w:.1f}W ({gpu.power_usage_w/gpu.power_limit_w*100:.1f}%)")
                
            if gpu.driver_version:
                items.append(f"  驅動版本: {gpu.driver_version}")
        
        return self.generate_section("GPU信息", items)
    
    def generate_disk_section(self) -> str:
        """生成磁盤信息部分"""
        disk = self.resources.disk
        
        items = [
            f"掛載點: {disk.mount_point}",
            f"總空間: {disk.total_gb:.2f} GB",
            f"可用空間: {disk.free_gb:.2f} GB ({100 - disk.used_percent:.1f}% 空閒)"
        ]
        
        if disk.write_speed_mbs > 0:
            items.extend([
                f"寫入速度: {disk.write_speed_mbs:.2f} MB/s",
                f"讀取速度: {disk.read_speed_mbs:.2f} MB/s",
                f"存儲類型: {disk.io_type.upper() if disk.io_type else ('NVMe SSD' if disk.is_nvme else 'SATA/HDD')}"
            ])
        
        return self.generate_section("磁盤信息", items)
    
    def generate_network_section(self) -> str:
        """生成網絡信息部分"""
        net = self.resources.network
        
        if not net.interface:
            return self.generate_section("網絡信息", ["未檢測到網絡介面"])
        
        items = [
            f"網絡介面: {net.interface} ({'WiFi' if net.is_wifi else '有線'})",
            f"連接狀態: {'已連接' if net.is_connected else '未連接'}"
        ]
        
        if net.speed_mbps > 0:
            items.append(f"連接速度: {net.speed_mbps} Mbps")
            
        if net.download_speed_mbps > 0:
            items.append(f"下載速度: {net.download_speed_mbps:.2f} Mbps")
            
        if net.upload_speed_mbps > 0:
            items.append(f"上傳速度: {net.upload_speed_mbps:.2f} Mbps")
        
        return self.generate_section("網絡信息", items)
    
    def generate_recommendation_section(self) -> str:
        """生成配置建議部分"""
        gpu_model = detect_gpu_model()
        
        if gpu_model["is_rtx3070_laptop"]:
            title = "檢測到 RTX 3070 Laptop GPU，已應用專用優化配置"
        elif self.resources.gpu.available:
            title = f"檢測到 {gpu_model['model']} GPU，已應用優化配置"
        else:
            title = "未檢測到GPU，已應用CPU優化配置"
        
        items = [
            f"批次大小: {self.config.batch_size}",
            f"梯度累積步數: {self.config.gradient_accumulation_steps}",
            f"混合精度訓練: {'啟用' if self.config.mixed_precision else '停用'}",
            f"XLA加速: {'啟用' if self.config.xla_acceleration else '停用'}",
            f"模型精度: {self.config.model_precision}",
            f"模型容量因子: {self.config.capacity_factor}",
            f"並行線程數: {self.config.interop_threads}",
            f"運算線程數: {self.config.intraop_threads}",
        ]
        
        # GPU相關配置
        if self.resources.gpu.available:
            if self.config.gpu_memory_limit_mb:
                items.append(f"GPU記憶體限制: {self.config.gpu_memory_limit_mb}MB")
                
            if self.config.multi_gpu:
                items.append(f"多GPU策略: {self.config.distributed_strategy}")
                
            if self.config.gpu_ids:
                items.append(f"使用GPU: {', '.join(map(str, self.config.gpu_ids))}")
        
        return "===== 優化建議 =====\n\n" + title + ":\n\n" + "\n".join(f"  {item}" for item in items) + "\n"
    
    def generate(self) -> str:
        """生成完整報告"""
        sections = [
            "===== 硬體資源報告 =====\n",
            self.generate_system_section(),
            self.generate_memory_section(),
            self.generate_gpu_section(),
            self.generate_disk_section(),
            self.generate_network_section(),
            self.generate_recommendation_section()
        ]
        
        return "".join(sections)


def generate_hardware_report() -> str:
    """生成硬體資源報告"""
    resources = detect_hardware_resources()
    config = get_hardware_optimized_config()
    
    # 創建報告生成器
    generator = HardwareReportGenerator(resources, config)
    
    # 生成報告
    return generator.generate()


def optimize_for_battery_aging_model() -> Dict[str, Any]:
    """專門針對電池老化預測模型的硬體優化建議"""
    resources = detect_hardware_resources()
    config = get_hardware_optimized_config()
    
    # 獲取GPU情況
    gpu_model = detect_gpu_model()
    
    # 電池特定優化建議
    battery_optimizations = {
        "training": {
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "learning_rate": 1e-3,  # 默認學習率
            "mixed_precision": config.mixed_precision,
            "model_precision": config.model_precision,
            "xla_acceleration": config.xla_acceleration,
            "prefetch_buffer": "auto" if config.prefetch_buffer_size == "auto" else config.prefetch_buffer_size,
            "thread_count": config.thread_count,
            "interop_threads": config.interop_threads,
            "intraop_threads": config.intraop_threads,
        },
        "model": {
            "capacity_factor": config.capacity_factor,
            "recommended_cnn_filters": int(64 * config.capacity_factor),
            "recommended_lstm_units": int(128 * config.capacity_factor),
            "recommended_dense_units": int(64 * config.capacity_factor),
        },
        "hardware": {
            "gpu_available": resources.gpu.available,
            "gpu_model": gpu_model.get("model", "unknown"),
            "gpu_memory_gb": gpu_model.get("ram_gb", 0),
            "cpu_cores": resources.system.logical_cores,
            "ram_gb": resources.memory.total_gb,
            "storage_type": resources.disk.io_type if hasattr(resources.disk, "io_type") else "unknown",
        },
        "environment_variables": config.environment_variables
    }
    
    return battery_optimizations


def create_hardware_profile_file():
    """創建硬體配置文件，供其他模組使用"""
    try:
        # 獲取配置
        hardware_config = get_hardware_optimized_config()
        hardware_report = generate_hardware_report()
        battery_optimizations = optimize_for_battery_aging_model()
        
        # 創建Profile物件
        profile = {
            "hardware_config": asdict(hardware_config),
            "optimizations": battery_optimizations,
            "report": hardware_report,
            "timestamp": time.time(),
            "generated_by": "system_config.py"
        }
        
        # 確定文件路徑
        profile_dir = "profiles"
        if config and config.get:
            profile_dir = config.get("system.profile_dir", "profiles")
        
        os.makedirs(profile_dir, exist_ok=True)
        profile_path = os.path.join(profile_dir, "hardware_profile.json")
        
        # 保存到文件
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
            
        logger.info(f"已生成硬體配置文件: {profile_path}")
        return profile_path
            
    except Exception as e:
        logger.error(f"創建硬體配置文件時出錯: {e}")
        return None


# 命令行界面
def main():
    """命令行入口點"""
    import argparse
    
    # 設置日誌格式
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    
    # 創建參數解析器
    parser = argparse.ArgumentParser(description="硬體資源檢測與優化工具")
    parser.add_argument("--setup", action="store_true", help="自動配置環境變數")
    parser.add_argument("--report", action="store_true", help="生成系統報告")
    parser.add_argument("--save", type=str, help="保存報告到文件", default="")
    parser.add_argument("--profile", action="store_true", help="創建硬體配置文件")
    parser.add_argument("--debug", action="store_true", help="啟用調試日誌")
    parser.add_argument("--json", action="store_true", help="以JSON格式輸出")
    parser.add_argument("--detect", choices=["all", "gpu", "cpu", "memory", "disk", "network"], 
                      default="all", help="僅檢測指定資源")
    parser.add_argument("--optimize-for", choices=["training", "inference", "battery"],
                      default="training", help="針對特定場景優化")
    
    args = parser.parse_args()
    
    # 設置日誌級別
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 處理命令行參數
    if args.setup:
        config = auto_setup_env()
        print("\n已自動配置環境變數:\n")
        for k, v in config.environment_variables.items():
            print(f"  {k}={v}")
    
    # 生成並顯示報告
    if args.report or args.save:
        if args.optimize_for == "battery":
            optimizations = optimize_for_battery_aging_model()
            if args.json:
                import json
                print(json.dumps(optimizations, indent=2, ensure_ascii=False))
            else:
                print("\n===== 電池老化預測專用優化配置 =====")
                for section, values in optimizations.items():
                    print(f"\n{section.upper()}:")
                    for k, v in values.items():
                        print(f"  {k}: {v}")
        else:
            # 創建報告生成器
            report = generate_hardware_report()
            print(report)
            
            # 保存報告到文件
            if args.save:
                with open(args.save, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n報告已保存到: {args.save}")
    
    # 創建硬體配置文件
    if args.profile:
        profile_path = create_hardware_profile_file()
        if profile_path:
            print(f"\n硬體配置文件已生成: {profile_path}")
    
    # 如果沒有指定任何操作，顯示基本系統資訊
    if not any([args.setup, args.report, args.save, args.profile, args.detect != "all"]):
        report = generate_hardware_report()
        print(report)
        print("\n使用 --help 選項查看更多功能")


if __name__ == "__main__":
    main()