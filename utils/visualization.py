#utils/visualization.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""視覺化工具模組 - 電池老化預測系統 (優化版)"""

import os
from typing import Dict, List, Tuple, Optional, Union, Any, TypeVar, Callable
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# 有條件導入
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 導入自訂模組
try:
    from config.base_config import config
    from core.logging import setup_logger
except ImportError:
    # 配置不可用時提供基本功能
    config = {"system.figures_dir": "figures"}
    
    def setup_logger(name):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# 設置記錄器和全局繪圖風格
logger = setup_logger("utils.visualization")
plt.style.use('ggplot')
sns.set(style="whitegrid")

# 類型定義
T = TypeVar('T')
PathLike = Union[str, Path]
FigureType = Union[plt.Figure, 'go.Figure']


class PlotConfig:
    """圖表配置類，用於存儲和管理圖表設置"""
    
    def __init__(self, title="", figsize=(12, 8), dpi=300, save_dir=None, style='seaborn-v0_8-whitegrid',
                colors=None, filename=None, show=False, 
                tight_layout=True, return_fig=False, interactive=False,
                theme="plotly_white", font_family="Arial", font_size=12, **kwargs):
        """初始化圖表配置
        
        Args:
            title: 圖表標題
            figsize: 圖表大小 (寬, 高)，單位為英寸
            dpi: 圖表分辨率
            save_dir: 保存目錄
            style: Matplotlib樣式
            colors: 顏色列表
            filename: 保存的文件名
            show: 是否顯示圖表
            tight_layout: 是否使用緊湊布局
            return_fig: 是否返回圖形對象
            interactive: 是否創建交互式圖表 (需要 Plotly)
            theme: Plotly主題 (僅交互式圖表)
            font_family: 字體系列
            font_size: 字體大小
            **kwargs: 其他參數
        """
        self.title = title
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir) if save_dir else Path(config.get("system.figures_dir", "figures"))
        self.style = style
        self.colors = colors or config.get("visualization.colors", [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
        ])
        self.filename = filename
        self.show = show
        self.tight_layout = tight_layout
        self.return_fig = return_fig
        self.interactive = interactive
        self.theme = theme
        self.font_family = font_family
        self.font_size = font_size
        self.kwargs = kwargs
    
    def get_default_filename(self, prefix: str) -> str:
        """獲取默認文件名"""
        if self.filename:
            return self.filename
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.interactive and HAS_PLOTLY:
            return f"{prefix}_{timestamp}.html"
        return f"{prefix}_{timestamp}.png"
    
    def get_save_path(self, prefix: str) -> Path:
        """獲取保存路徑"""
        return self.save_dir / self.get_default_filename(prefix)
    
    def apply_style(self):
        """應用樣式到當前圖表"""
        if not self.interactive:
            plt.style.use(self.style)
            plt.rcParams.update({
                'font.family': self.font_family,
                'font.size': self.font_size
            })


class PlotBase(ABC):
    """圖表基類，定義圖表介面"""
    
    def __init__(self, config: PlotConfig):
        """初始化圖表"""
        self.config = config
        self.fig = None
        self.axs = None
        self.plotly_fig = None
    
    @abstractmethod
    def create(self) -> 'PlotBase':
        """創建圖表"""
        pass
    
    def save(self, path: Optional[PathLike] = None) -> str:
        """保存圖表"""
        if self.fig is None and self.plotly_fig is None:
            raise ValueError("圖表尚未創建")
        
        save_path = Path(path) if path else self.config.get_save_path(self.__class__.__name__.lower())
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.config.interactive and HAS_PLOTLY and self.plotly_fig is not None:
            # 保存交互式圖表
            self.plotly_fig.write_html(str(save_path))
        else:
            # 保存靜態圖表
            if self.config.tight_layout and self.fig is not None:
                self.fig.tight_layout()
            self.fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        logger.info(f"圖表已保存到: {save_path}")
        
        if self.config.show:
            if self.config.interactive and HAS_PLOTLY and self.plotly_fig is not None:
                self.plotly_fig.show()
            else:
                plt.show()
        elif not self.config.return_fig:
            plt.close(self.fig)
        
        if self.config.return_fig:
            return self.plotly_fig if self.config.interactive and HAS_PLOTLY and self.plotly_fig is not None else self.fig
        
        return str(save_path)
    
    def with_config(self, **kwargs) -> 'PlotBase':
        """更新配置並返回自身，支持方法鏈接"""
        for key, value in kwargs.items():
            setattr(self.config, key, value)
        return self


class TrainingHistoryPlot(PlotBase):
    """訓練歷史曲線圖"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.history = None
        self.metrics = None
        self.smooth = True
        self.smooth_factor = 0.6
    
    def set_data(self, history: Dict[str, List[float]], metrics=None, smooth=True, smooth_factor=0.6) -> 'TrainingHistoryPlot':
        """設置數據
        
        Args:
            history: 訓練歷史字典 (key: 指標名, value: 指標值列表)
            metrics: 要顯示的指標列表 (None表示全部)
            smooth: 是否平滑曲線
            smooth_factor: 平滑因子 (0-1)，越大越平滑
        """
        self.history = history
        self.metrics = metrics
        self.smooth = smooth
        self.smooth_factor = smooth_factor
        return self
    
    def create(self) -> 'TrainingHistoryPlot':
        """創建訓練歷史曲線圖"""
        if not self.history:
            raise ValueError("歷史數據為空，無法繪製圖表")
        
        # 獲取指標
        available_metrics = [k for k in self.history if not k.startswith('val_')]
        val_metrics = [k for k in self.history if k.startswith('val_')]
        
        # 如果沒有指定指標，使用所有非驗證指標
        if self.metrics is None:
            self.metrics = [m for m in available_metrics if not m.startswith('val_') and not m == 'lr']
        
        # 計算子圖數量
        n_plots = len(self.metrics) + (1 if 'lr' in self.history else 0)
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            self.plotly_fig = make_subplots(
                rows=n_plots, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[m.capitalize() for m in self.metrics] + (['Learning Rate'] if 'lr' in self.history else [])
            )
            
            # 平滑化函數
            def smooth_curve(points, factor=0.6):
                """指數平滑曲線"""
                smoothed_points = []
                for point in points:
                    if smoothed_points:
                        smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
                    else:
                        smoothed_points.append(point)
                return smoothed_points
            
            # 繪製每個指標
            for i, metric in enumerate(self.metrics):
                train_values = self.history[metric]
                epochs = list(range(1, len(train_values) + 1))
                
                # 訓練數據
                if self.smooth and len(train_values) > 3:
                    smoothed_values = smooth_curve(train_values, self.smooth_factor)
                    self.plotly_fig.add_trace(
                        go.Scatter(
                            x=epochs, y=smoothed_values,
                            mode='lines',
                            name=f'Training {metric}',
                            line=dict(color=self.config.colors[i % len(self.config.colors)])
                        ),
                        row=i+1, col=1
                    )
                    # 添加原始點
                    self.plotly_fig.add_trace(
                        go.Scatter(
                            x=epochs, y=train_values,
                            mode='markers',
                            name=f'Training {metric} (raw)',
                            marker=dict(color=self.config.colors[i % len(self.config.colors)], size=4),
                            showlegend=False
                        ),
                        row=i+1, col=1
                    )
                else:
                    self.plotly_fig.add_trace(
                        go.Scatter(
                            x=epochs, y=train_values,
                            mode='lines+markers',
                            name=f'Training {metric}',
                            line=dict(color=self.config.colors[i % len(self.config.colors)])
                        ),
                        row=i+1, col=1
                    )
                
                # 驗證數據
                val_metric = f'val_{metric}'
                if val_metric in self.history:
                    val_values = self.history[val_metric]
                    
                    if self.smooth and len(val_values) > 3:
                        smoothed_val = smooth_curve(val_values, self.smooth_factor)
                        self.plotly_fig.add_trace(
                            go.Scatter(
                                x=epochs, y=smoothed_val,
                                mode='lines',
                                name=f'Validation {metric}',
                                line=dict(color=self.config.colors[(i + 5) % len(self.config.colors)])
                            ),
                            row=i+1, col=1
                        )
                        # 添加原始點
                        self.plotly_fig.add_trace(
                            go.Scatter(
                                x=epochs, y=val_values,
                                mode='markers',
                                name=f'Validation {metric} (raw)',
                                marker=dict(color=self.config.colors[(i + 5) % len(self.config.colors)], size=4),
                                showlegend=False
                            ),
                            row=i+1, col=1
                        )
                    else:
                        self.plotly_fig.add_trace(
                            go.Scatter(
                                x=epochs, y=val_values,
                                mode='lines+markers',
                                name=f'Validation {metric}',
                                line=dict(color=self.config.colors[(i + 5) % len(self.config.colors)])
                            ),
                            row=i+1, col=1
                        )
                    
                    # 找出最佳值
                    if len(val_values) > 0:
                        best_epoch = np.argmin(val_values) if 'loss' in metric else np.argmax(val_values)
                        best_value = val_values[best_epoch]
                        
                        # 添加最佳點標記
                        self.plotly_fig.add_trace(
                            go.Scatter(
                                x=[best_epoch + 1], y=[best_value],
                                mode='markers',
                                marker=dict(color='red', size=10, symbol='star'),
                                name=f'Best {metric}',
                                text=f'Best: {best_value:.4f}',
                                hoverinfo='text+x+y'
                            ),
                            row=i+1, col=1
                        )
                
                # 更新軸標籤
                self.plotly_fig.update_yaxes(title_text=metric.capitalize(), row=i+1, col=1)
                
                # 對於準確率指標，將y軸限制在0-1
                if 'acc' in metric or 'accuracy' in metric:
                    self.plotly_fig.update_yaxes(range=[-0.05, 1.05], row=i+1, col=1)
            
            # 學習率曲線
            if 'lr' in self.history:
                lr_idx = len(self.metrics)
                lr_values = self.history['lr']
                epochs = list(range(1, len(lr_values) + 1))
                
                self.plotly_fig.add_trace(
                    go.Scatter(
                        x=epochs, y=lr_values,
                        mode='lines+markers',
                        name='Learning Rate',
                        line=dict(color='green')
                    ),
                    row=lr_idx+1, col=1
                )
                
                # 更新軸標籤
                self.plotly_fig.update_yaxes(
                    title_text='Learning Rate', 
                    type='log',
                    row=lr_idx+1, col=1
                )
            
            # 更新x軸標籤
            self.plotly_fig.update_xaxes(title_text='Epochs', row=n_plots, col=1)
            
            # 更新圖表標題和佈局
            self.plotly_fig.update_layout(
                title_text=self.config.title or "Training History",
                template=self.config.theme,
                height=250 * n_plots,
                width=900,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                ),
                hovermode='closest'
            )
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建子圖
            self.fig, self.axs = plt.subplots(n_plots, 1, figsize=self.config.figsize, 
                                             dpi=self.config.dpi, constrained_layout=True)
            
            # 確保axs是一個列表
            if n_plots == 1:
                self.axs = [self.axs]
            
            # 平滑化函數
            def smooth_curve(points, factor=0.6):
                """指數平滑曲線"""
                smoothed_points = []
                for point in points:
                    if smoothed_points:
                        smoothed_points.append(smoothed_points[-1] * factor + point * (1 - factor))
                    else:
                        smoothed_points.append(point)
                return smoothed_points
            
            # 繪製每個指標
            for i, metric in enumerate(self.metrics):
                train_values = self.history[metric]
                train_values_plot = smooth_curve(train_values, self.smooth_factor) if self.smooth and len(train_values) > 3 else train_values
                
                # 繪製訓練數據
                color = self.config.colors[i % len(self.config.colors)]
                self.axs[i].plot(train_values, 'o-', alpha=0.3, color=color)
                self.axs[i].plot(train_values_plot, '-', label=f'Training {metric}', color=color)
                
                # 如果有驗證數據，也繪製
                val_metric = f'val_{metric}'
                if val_metric in self.history:
                    val_values = self.history[val_metric]
                    val_values_plot = smooth_curve(val_values, self.smooth_factor) if self.smooth and len(val_values) > 3 else val_values
                    
                    val_color = self.config.colors[(i + 5) % len(self.config.colors)]
                    self.axs[i].plot(val_values, 's-', alpha=0.3, color=val_color)
                    self.axs[i].plot(val_values_plot, '-', label=f'Validation {metric}', color=val_color)
                    
                    # 找出最佳值
                    if len(val_values) > 0:
                        best_epoch = np.argmin(val_values) if 'loss' in metric else np.argmax(val_values)
                        best_value = val_values[best_epoch]
                        
                        self.axs[i].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
                        self.axs[i].text(best_epoch, best_value, f' Best: {best_value:.4f}', verticalalignment='center')
                
                self.axs[i].set_ylabel(metric.capitalize())
                self.axs[i].legend(loc='best')
                self.axs[i].grid(True, alpha=0.3)
                self.axs[i].set_title(f"{metric.capitalize()} vs. Epochs")
                
                # 對於準確率指標，將y軸限制在0-1
                if 'acc' in metric or 'accuracy' in metric:
                    self.axs[i].set_ylim(-0.05, 1.05)
            
            # 如果有學習率，繪製學習率曲線
            if 'lr' in self.history:
                lr_ax = self.axs[-1]
                lr_values = self.history['lr']
                lr_ax.plot(lr_values, '.-', label='Learning Rate', color='green')
                lr_ax.set_ylabel('Learning Rate')
                lr_ax.set_yscale('log')
                lr_ax.legend(loc='best')
                lr_ax.grid(True, alpha=0.3)
                lr_ax.set_title("Learning Rate vs. Epochs")
            
            # 所有子圖共享x軸標籤
            for ax in self.axs[:-1]:
                ax.set_xticklabels([])
            
            self.axs[-1].set_xlabel('Epochs')
            
            # 總標題
            self.fig.suptitle(self.config.title or "Training History", fontsize=16)
        
        return self


class ChargeDischargePlot(PlotBase):
    """充放電性能比較圖表"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.charge_metrics = None
        self.discharge_metrics = None
        self.epochs = None
        self.metric_names = None
        self.temp = None
    
    def set_data(self, charge_metrics: Dict[str, List[float]], discharge_metrics: Dict[str, List[float]], 
                epochs: List[int], metric_names: List[str] = None, temp: str = None) -> 'ChargeDischargePlot':
        """設置數據
        
        Args:
            charge_metrics: 充電指標字典 (key: 指標名, value: 指標值列表)
            discharge_metrics: 放電指標字典 (key: 指標名, value: 指標值列表)
            epochs: 輪次列表
            metric_names: 要顯示的指標名稱列表 (None表示全部)
            temp: 溫度標識
        """
        self.charge_metrics = charge_metrics
        self.discharge_metrics = discharge_metrics
        self.epochs = epochs
        
        if metric_names is None:
            # 使用兩種指標都有的度量
            charge_keys = set(charge_metrics.keys())
            discharge_keys = set(discharge_metrics.keys())
            self.metric_names = list(charge_keys.intersection(discharge_keys))
            # 按優先順序排序
            priority = ['loss', 'mae', 'mse', 'rmse', 'r2_score']
            self.metric_names = sorted(self.metric_names, 
                               key=lambda x: priority.index(x) if x in priority else len(priority))
        else:
            self.metric_names = metric_names
        
        self.temp = temp
        return self
    
    def create(self) -> 'ChargeDischargePlot':
        """創建充放電性能比較圖表"""
        if not self.charge_metrics or not self.discharge_metrics or not self.epochs:
            raise ValueError("數據不完整，無法繪製圖表")
        
        if not self.metric_names:
            raise ValueError("沒有有效的指標可以繪製")
        
        # 過濾有效的指標 (確保兩種類型都有數據)
        valid_metrics = []
        for metric in self.metric_names:
            if (metric in self.charge_metrics and metric in self.discharge_metrics and
                len(self.charge_metrics[metric]) > 0 and len(self.discharge_metrics[metric]) > 0):
                valid_metrics.append(metric)
        
        # 計算比率
        ratios = {}
        for metric in valid_metrics:
            ratios[metric] = []
            charge_values = self.charge_metrics[metric]
            discharge_values = self.discharge_metrics[metric]
            
            for i in range(min(len(charge_values), len(discharge_values))):
                if charge_values[i] and discharge_values[i] and charge_values[i] > 0:
                    ratios[metric].append(discharge_values[i] / charge_values[i])
                else:
                    ratios[metric].append(None)
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            n_metrics = len(valid_metrics)
            self.plotly_fig = make_subplots(
                rows=n_metrics, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=[m.capitalize() for m in valid_metrics]
            )
            
            # 繪製每個指標
            for i, metric in enumerate(valid_metrics):
                charge_values = self.charge_metrics[metric]
                discharge_values = self.discharge_metrics[metric]
                epoch_list = self.epochs[:min(len(charge_values), len(discharge_values))]
                
                # 充電數據
                self.plotly_fig.add_trace(
                    go.Scatter(
                        x=epoch_list, y=charge_values[:len(epoch_list)],
                        mode='lines+markers',
                        name=f'Charge {metric}',
                        line=dict(color='blue')
                    ),
                    row=i+1, col=1
                )
                
                # 放電數據
                self.plotly_fig.add_trace(
                    go.Scatter(
                        x=epoch_list, y=discharge_values[:len(epoch_list)],
                        mode='lines+markers',
                        name=f'Discharge {metric}',
                        line=dict(color='red')
                    ),
                    row=i+1, col=1
                )
                
                # 比率數據
                if metric in ratios:
                    self.plotly_fig.add_trace(
                        go.Scatter(
                            x=epoch_list, y=ratios[metric][:len(epoch_list)],
                            mode='lines+markers',
                            name=f'Ratio {metric}',
                            line=dict(color='green'),
                            yaxis=f'y{i+1}2'
                        ),
                        row=i+1, col=1
                    )
                    
                    # 添加輔助y軸
                    self.plotly_fig.update_layout(**{
                        f'yaxis{i+1}2': dict(
                            title='Discharge/Charge Ratio',
                            titlefont=dict(color='green'),
                            tickfont=dict(color='green'),
                            anchor='x',
                            overlaying=f'y{i+1}',
                            side='right'
                        )
                    })
                
                # 更新軸標籤
                self.plotly_fig.update_yaxes(title_text=metric.capitalize(), row=i+1, col=1)
            
            # 更新x軸標籤
            self.plotly_fig.update_xaxes(title_text='Epochs', row=n_metrics, col=1)
            
            # 更新圖表標題和佈局
            title = self.config.title or f"Charge vs Discharge Performance"
            if self.temp:
                title += f" ({self.temp})"
                
            self.plotly_fig.update_layout(
                title_text=title,
                template=self.config.theme,
                height=250 * n_metrics,
                width=900,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                ),
                hovermode='closest'
            )
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建子圖
            n_metrics = len(valid_metrics)
            self.fig, self.axs = plt.subplots(n_metrics, 1, figsize=self.config.figsize,
                                             dpi=self.config.dpi, constrained_layout=True)
            
            # 確保axs是一個列表
            if n_metrics == 1:
                self.axs = [self.axs]
            
            # 繪製每個指標
            for i, metric in enumerate(valid_metrics):
                ax1 = self.axs[i]
                charge_values = self.charge_metrics[metric]
                discharge_values = self.discharge_metrics[metric]
                
                # 繪製充電和放電數據
                epochs_to_plot = self.epochs[:min(len(charge_values), len(discharge_values))]
                ax1.plot(epochs_to_plot, charge_values[:len(epochs_to_plot)], 'b-o', label=f'Charge {metric}')
                ax1.plot(epochs_to_plot, discharge_values[:len(epochs_to_plot)], 'r-s', label=f'Discharge {metric}')
                
                # 如果有比率數據，添加輔助y軸
                if metric in ratios and any(r is not None for r in ratios[metric]):
                    ax2 = ax1.twinx()
                    ax2.plot(epochs_to_plot, ratios[metric][:len(epochs_to_plot)], 'g-^', label=f'Ratio')
                    ax2.set_ylabel('Discharge/Charge Ratio', color='g')
                    ax2.tick_params(axis='y', labelcolor='g')
                    
                    # 添加水平參考線
                    ax2.axhline(y=1.0, color='g', linestyle='--', alpha=0.3)
                    
                    # 合併圖例
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
                else:
                    ax1.legend(loc='best')
                
                ax1.set_ylabel(metric.capitalize())
                ax1.grid(True, alpha=0.3)
                ax1.set_title(f"{metric.capitalize()} vs. Epochs")
            
            # 所有子圖共享x軸標籤
            for ax in self.axs[:-1]:
                ax.set_xticklabels([])
            
            self.axs[-1].set_xlabel('Epochs')
            
            # 總標題
            title = self.config.title or f"Charge vs Discharge Performance"
            if self.temp:
                title += f" ({self.temp})"
            
            self.fig.suptitle(title, fontsize=16)
        
        return self


class ChargeDischargeSummaryPlot(PlotBase):
    """充放電性能摘要圖表"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.eval_results = None
        self.temp = None
    
    def set_data(self, eval_results: Dict[str, Any], temp: str = None) -> 'ChargeDischargeSummaryPlot':
        """設置數據
        
        Args:
            eval_results: 評估結果字典 (包含charge, discharge和comparative)
            temp: 溫度標識
        """
        self.eval_results = eval_results
        self.temp = temp
        return self
    
    def create(self) -> 'ChargeDischargeSummaryPlot':
        """創建充放電性能摘要圖表"""
        if not self.eval_results:
            raise ValueError("評估結果為空，無法繪製圖表")
        
        # 確保有充電和放電數據
        if 'charge' not in self.eval_results or 'discharge' not in self.eval_results:
            raise ValueError("評估結果缺少充放電數據")
        
        # 獲取樣本數
        charge_count = self.eval_results['charge'].get('count', 0)
        discharge_count = self.eval_results['discharge'].get('count', 0)
        total_count = charge_count + discharge_count
        
        if total_count == 0:
            raise ValueError("沒有樣本數據")
        
        # 提取主要指標
        metrics = ['loss', 'mae', 'rmse', 'r2_score']
        available_metrics = []
        
        for metric in metrics:
            if (metric in self.eval_results['charge'] and 
                metric in self.eval_results['discharge']):
                available_metrics.append(metric)
        
        if not available_metrics:
            raise ValueError("沒有可用的指標")
        
        # 收集數據
        charge_values = []
        discharge_values = []
        ratio_values = []
        
        for metric in available_metrics:
            charge_values.append(self.eval_results['charge'][metric])
            discharge_values.append(self.eval_results['discharge'][metric])
            
            # 計算比率
            ratio_key = f"{metric}_ratio"
            if 'comparative' in self.eval_results and ratio_key in self.eval_results['comparative']:
                ratio_values.append(self.eval_results['comparative'][ratio_key])
            else:
                # 手動計算比率
                if self.eval_results['charge'][metric] > 0:
                    ratio_values.append(self.eval_results['discharge'][metric] / self.eval_results['charge'][metric])
                else:
                    ratio_values.append(None)
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            # 1. 分組條形圖
            self.plotly_fig = go.Figure()
            
            # 添加指標條形圖
            self.plotly_fig.add_trace(go.Bar(
                x=available_metrics,
                y=charge_values,
                name='Charge',
                marker_color='blue'
            ))
            
            self.plotly_fig.add_trace(go.Bar(
                x=available_metrics,
                y=discharge_values,
                name='Discharge',
                marker_color='red'
            ))
            
            # 添加比率標記
            for i, (metric, ratio) in enumerate(zip(available_metrics, ratio_values)):
                if ratio is not None:
                    self.plotly_fig.add_annotation(
                        x=metric,
                        y=max(charge_values[i], discharge_values[i]) * 1.1,
                        text=f"Ratio: {ratio:.2f}",
                        showarrow=False,
                        font=dict(
                            size=10,
                            color="green"
                        )
                    )
            
            # 更新佈局
            title = self.config.title or "Charge vs Discharge Metrics"
            if self.temp:
                title += f" ({self.temp})"
                
            self.plotly_fig.update_layout(
                title_text=title,
                template=self.config.theme,
                xaxis_title="Metrics",
                yaxis_title="Value",
                barmode='group',
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                ),
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        text=f"Samples: {charge_count} charge, {discharge_count} discharge",
                        showarrow=False
                    )
                ]
            )
            
            # 2. 添加樣本分佈餅圖子圖 (使用第二個圖表)
            pie_fig = go.Figure(data=[go.Pie(
                labels=['Charge', 'Discharge'],
                values=[charge_count, discharge_count],
                hole=.3,
                marker_colors=['blue', 'red'],
                textinfo='label+percent',
                texttemplate="%{label}: %{value} (%{percent})",
                insidetextfont=dict(color='white')
            )])
            
            pie_fig.update_layout(
                title_text="Sample Distribution",
                template=self.config.theme,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                )
            )
            
            # 將第二個圖表保存為單獨的文件
            pie_save_path = Path(self.config.save_dir) / f"sample_dist_{self.temp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            pie_fig.write_html(str(pie_save_path))
            logger.info(f"樣本分佈圖已保存到: {pie_save_path}")
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建2x1子圖
            self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize,
                                            dpi=self.config.dpi, constrained_layout=True,
                                            gridspec_kw={'width_ratios': [2, 1]})
            
            # 1. 分組條形圖
            x = np.arange(len(available_metrics))
            width = 0.35
            
            # 繪製充電和放電條形圖
            rects1 = ax1.bar(x - width/2, charge_values, width, label='Charge', color='blue')
            rects2 = ax1.bar(x + width/2, discharge_values, width, label='Discharge', color='red')
            
            # 添加比率標記
            for i, ratio in enumerate(ratio_values):
                if ratio is not None:
                    color = 'green'
                    if ratio > 1.5 or ratio < 0.5:
                        color = 'orange'
                    if ratio > 2.0 or ratio < 0.3:
                        color = 'red'
                        
                    ax1.text(x[i], max(charge_values[i], discharge_values[i]) * 1.1,
                            f'Ratio: {ratio:.2f}', ha='center', color=color)
            
            # 添加軸標籤和標題
            ax1.set_xlabel('Metrics')
            ax1.set_ylabel('Value')
            ax1.set_title('Charge vs Discharge Metrics')
            ax1.set_xticks(x)
            ax1.set_xticklabels(available_metrics)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. 樣本分佈餅圖
            labels = ['Charge', 'Discharge']
            sizes = [charge_count, discharge_count]
            colors = ['blue', 'red']
            explode = (0.1, 0)  # 突出充電部分
            
            ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax2.axis('equal')  # 確保餅圖是圓的
            ax2.set_title('Sample Distribution')
            
            # 總標題
            title = self.config.title or "Charge vs Discharge Performance"
            if self.temp:
                title += f" ({self.temp})"
            
            self.fig.suptitle(title, fontsize=16)
        
        return self


class CrossTempEvaluationPlot(PlotBase):
    """交叉溫度評估圖表"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.cross_eval_data = None
        self.metric = None
    
    def set_data(self, cross_eval_data: Dict[str, Dict[str, Dict[str, float]]], 
                metric: str = 'loss') -> 'CrossTempEvaluationPlot':
        """設置數據
        
        Args:
            cross_eval_data: 交叉評估數據 (model_temp -> data_temp -> metrics)
            metric: 要顯示的指標
        """
        self.cross_eval_data = cross_eval_data
        self.metric = metric
        return self
    
    def create(self) -> 'CrossTempEvaluationPlot':
        """創建交叉溫度評估圖表"""
        if not self.cross_eval_data:
            raise ValueError("交叉評估數據為空，無法繪製圖表")
        
        # 獲取所有溫度
        model_temps = list(self.cross_eval_data.keys())
        data_temps = set()
        
        for model_temp in model_temps:
            data_temps.update(self.cross_eval_data[model_temp].keys())
        
        data_temps = sorted(data_temps)
        
        # 創建評估矩陣
        matrix = np.zeros((len(model_temps), len(data_temps)))
        matrix.fill(np.nan)  # 初始化為 NaN
        
        for i, model_temp in enumerate(model_temps):
            for j, data_temp in enumerate(data_temps):
                if (data_temp in self.cross_eval_data[model_temp] and 
                    isinstance(self.cross_eval_data[model_temp][data_temp], dict) and
                    self.metric in self.cross_eval_data[model_temp][data_temp]):
                    matrix[i, j] = self.cross_eval_data[model_temp][data_temp][self.metric]
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            self.plotly_fig = go.Figure()
            
            # 創建熱力圖
            self.plotly_fig.add_trace(go.Heatmap(
                z=matrix,
                x=data_temps,
                y=model_temps,
                colorscale='YlGnBu',
                colorbar=dict(
                    title=self.metric.capitalize(),
                    titleside='right'
                ),
                hovertemplate='Model: %{y}<br>Data: %{x}<br>Value: %{z:.4f}<extra></extra>'
            ))
            
            # 更新佈局
            title = self.config.title or f"Cross-Temperature Evaluation - {self.metric.capitalize()}"
            
            self.plotly_fig.update_layout(
                title_text=title,
                template=self.config.theme,
                xaxis_title="Data Temperature",
                yaxis_title="Model Temperature",
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                )
            )
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建圖表
            self.fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            
            # 繪製熱力圖
            heatmap = ax.imshow(matrix, cmap='YlGnBu')
            
            # 添加顏色條
            cbar = ax.figure.colorbar(heatmap, ax=ax)
            cbar.ax.set_ylabel(self.metric.capitalize(), rotation=-90, va="bottom")
            
            # 設置刻度標籤
            ax.set_xticks(np.arange(len(data_temps)))
            ax.set_yticks(np.arange(len(model_temps)))
            ax.set_xticklabels(data_temps)
            ax.set_yticklabels(model_temps)
            
            # 旋轉x軸標籤
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # 添加值標籤
            for i in range(len(model_temps)):
                for j in range(len(data_temps)):
                    if not np.isnan(matrix[i, j]):
                        text = ax.text(j, i, f"{matrix[i, j]:.4f}",
                                      ha="center", va="center", color="w" if matrix[i, j] > np.nanmean(matrix) else "black")
            
            # 設置標題和軸標籤
            title = self.config.title or f"Cross-Temperature Evaluation - {self.metric.capitalize()}"
            ax.set_title(title)
            ax.set_xlabel("Data Temperature")
            ax.set_ylabel("Model Temperature")
            
            # 調整佈局
            self.fig.tight_layout()
        
        return self


class TemperatureComparisonPlot(PlotBase):
    """溫度比較圖表"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.metrics_by_temp = None
        self.metric_names = None
    
    def set_data(self, metrics_by_temp: Dict[str, Dict[str, float]], 
                metric_names: List[str] = None) -> 'TemperatureComparisonPlot':
        """設置數據
        
        Args:
            metrics_by_temp: 各溫度的指標字典 (temp -> metric -> value)
            metric_names: 要顯示的指標名稱列表 (None表示全部)
        """
        self.metrics_by_temp = metrics_by_temp
        
        # 確定要顯示的指標
        if metric_names:
            self.metric_names = metric_names
        else:
            # 找出所有溫度都有的指標
            all_metrics = set()
            for temp in metrics_by_temp:
                if isinstance(metrics_by_temp[temp], dict):
                    all_metrics.update(metrics_by_temp[temp].keys())
            
            # 篩選所有溫度都有的指標
            common_metrics = set()
            for metric in all_metrics:
                if all(isinstance(metrics_by_temp[temp], dict) and metric in metrics_by_temp[temp] 
                      for temp in metrics_by_temp):
                    common_metrics.add(metric)
            
            # 按優先順序排序
            priority = ['loss', 'val_loss', 'mae', 'val_mae', 'rmse', 'val_rmse', 'r2_score', 'val_r2_score']
            self.metric_names = sorted(common_metrics, 
                              key=lambda x: priority.index(x) if x in priority else len(priority))
        
        return self
    
    def create(self) -> 'TemperatureComparisonPlot':
        """創建溫度比較圖表"""
        if not self.metrics_by_temp:
            raise ValueError("溫度指標數據為空，無法繪製圖表")
        
        if not self.metric_names:
            raise ValueError("沒有共同的指標可以比較")
        
        # 獲取所有溫度
        temps = list(self.metrics_by_temp.keys())
        
        # 準備數據
        metrics_data = {}
        for metric in self.metric_names:
            metrics_data[metric] = []
            for temp in temps:
                if isinstance(self.metrics_by_temp[temp], dict) and metric in self.metrics_by_temp[temp]:
                    metrics_data[metric].append(self.metrics_by_temp[temp][metric])
                else:
                    metrics_data[metric].append(None)
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            self.plotly_fig = go.Figure()
            
            # 每個指標為單獨的子圖
            self.plotly_fig = make_subplots(
                rows=len(self.metric_names), cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[m.capitalize() for m in self.metric_names]
            )
            
            # 添加每個指標的條形圖
            for i, metric in enumerate(self.metric_names):
                values = metrics_data[metric]
                
                # 添加條形圖
                self.plotly_fig.add_trace(
                    go.Bar(
                        x=temps,
                        y=values,
                        name=metric,
                        marker_color=self.config.colors[i % len(self.config.colors)],
                        text=[f"{v:.4f}" if v is not None else "N/A" for v in values],
                        textposition='auto'
                    ),
                    row=i+1, col=1
                )
                
                # 更新y軸
                self.plotly_fig.update_yaxes(title_text=metric.capitalize(), row=i+1, col=1)
            
            # 更新x軸
            self.plotly_fig.update_xaxes(title_text="Temperature", row=len(self.metric_names), col=1)
            
            # 更新佈局
            title = self.config.title or "Temperature Comparison"
            
            self.plotly_fig.update_layout(
                title_text=title,
                template=self.config.theme,
                height=250 * len(self.metric_names),
                width=800,
                showlegend=False,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                )
            )
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建子圖
            n_metrics = len(self.metric_names)
            self.fig, self.axs = plt.subplots(n_metrics, 1, figsize=self.config.figsize,
                                             dpi=self.config.dpi, constrained_layout=True)
            
            # 確保axs是一個列表
            if n_metrics == 1:
                self.axs = [self.axs]
            
            # 繪製每個指標
            for i, metric in enumerate(self.metric_names):
                values = metrics_data[metric]
                color = self.config.colors[i % len(self.config.colors)]
                
                # 繪製條形圖
                bars = self.axs[i].bar(temps, values, color=color)
                
                # 添加數值標籤
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height is not None:
                        self.axs[i].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{height:.4f}', ha='center', va='bottom', rotation=0)
                
                self.axs[i].set_ylabel(metric.capitalize())
                self.axs[i].grid(True, alpha=0.3)
                self.axs[i].set_title(f"{metric.capitalize()} by Temperature")
            
            # 設置x軸標籤
            self.axs[-1].set_xlabel('Temperature')
            
            # 總標題
            title = self.config.title or "Temperature Comparison"
            self.fig.suptitle(title, fontsize=16)
        
        return self


class TransferCapabilityPlot(PlotBase):
    """轉移能力比較圖表"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.cross_eval_data = None
        self.metric = None
    
    def set_data(self, cross_eval_data: Dict[str, Dict[str, Dict[str, float]]], 
                metric: str = 'loss') -> 'TransferCapabilityPlot':
        """設置數據
        
        Args:
            cross_eval_data: 交叉評估數據 (model_temp -> data_temp -> metrics)
            metric: 要顯示的指標
        """
        self.cross_eval_data = cross_eval_data
        self.metric = metric
        return self
    
    def create(self) -> 'TransferCapabilityPlot':
        """創建轉移能力比較圖表"""
        if not self.cross_eval_data:
            raise ValueError("交叉評估數據為空，無法繪製圖表")
        
        # 獲取所有溫度
        model_temps = list(self.cross_eval_data.keys())
        
        # 計算每個模型的原生性能和轉移性能
        native_perf = []
        transfer_perf = []
        model_labels = []
        
        for model_temp in model_temps:
            model_labels.append(model_temp)
            
            # 原生性能 (模型在其訓練溫度的數據上的表現)
            if (model_temp in self.cross_eval_data[model_temp] and 
                isinstance(self.cross_eval_data[model_temp][model_temp], dict) and
                self.metric in self.cross_eval_data[model_temp][model_temp]):
                native_perf.append(self.cross_eval_data[model_temp][model_temp][self.metric])
            else:
                native_perf.append(None)
            
            # 轉移性能 (模型在其他溫度數據上的平均表現)
            transfer_values = []
            for data_temp in self.cross_eval_data[model_temp]:
                if data_temp != model_temp:
                    if (isinstance(self.cross_eval_data[model_temp][data_temp], dict) and
                        self.metric in self.cross_eval_data[model_temp][data_temp]):
                        transfer_values.append(self.cross_eval_data[model_temp][data_temp][self.metric])
            
            if transfer_values:
                transfer_perf.append(np.mean(transfer_values))
            else:
                transfer_perf.append(None)
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            self.plotly_fig = go.Figure()
            
            # 添加原生性能
            self.plotly_fig.add_trace(go.Bar(
                x=model_labels,
                y=native_perf,
                name='Native Performance',
                marker_color='green',
                text=[f"{v:.4f}" if v is not None else "N/A" for v in native_perf],
                textposition='auto'
            ))
            
            # 添加轉移性能
            self.plotly_fig.add_trace(go.Bar(
                x=model_labels,
                y=transfer_perf,
                name='Transfer Performance',
                marker_color='orange',
                text=[f"{v:.4f}" if v is not None else "N/A" for v in transfer_perf],
                textposition='auto'
            ))
            
            # 更新佈局
            title = self.config.title or f"Model Transfer Capability - {self.metric.capitalize()}"
            
            self.plotly_fig.update_layout(
                title_text=title,
                template=self.config.theme,
                xaxis_title="Model Temperature",
                yaxis_title=self.metric.capitalize(),
                barmode='group',
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                ),
                hovermode='closest'
            )
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建圖表
            self.fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            
            # 設置x位置
            x = np.arange(len(model_labels))
            width = 0.35
            
            # 繪製條形圖
            rects1 = ax.bar(x - width/2, native_perf, width, label='Native Performance', color='green')
            rects2 = ax.bar(x + width/2, transfer_perf, width, label='Transfer Performance', color='orange')
            
            # 添加數值標籤
            def add_labels(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height is not None:
                        ax.annotate(f'{height:.4f}',
                                   xy=(rect.get_x() + rect.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 點垂直偏移
                                   textcoords="offset points",
                                   ha='center', va='bottom', rotation=0)
            
            add_labels(rects1)
            add_labels(rects2)
            
            # 添加軸標籤和標題
            ax.set_xlabel('Model Temperature')
            ax.set_ylabel(self.metric.capitalize())
            ax.set_title(f"Model Transfer Capability - {self.metric.capitalize()}")
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 調整佈局
            self.fig.tight_layout()
        
        return self


class ResourceUsagePlot(PlotBase):
    """資源使用情況圖表"""
    
    def __init__(self, config: PlotConfig):
        super().__init__(config)
        self.time_points = None
        self.memory_usage = None
        self.gpu_usage = None
        self.events = None
    
    def set_data(self, time_points: List[float], memory_usage: List[float],
                gpu_usage: List[float] = None, events: Dict[float, str] = None) -> 'ResourceUsagePlot':
        """設置數據
        
        Args:
            time_points: 時間點列表 (秒)
            memory_usage: 記憶體使用列表 (GB)
            gpu_usage: GPU使用列表 (百分比)
            events: 事件字典 (時間點 -> 事件描述)
        """
        self.time_points = time_points
        self.memory_usage = memory_usage
        self.gpu_usage = gpu_usage
        self.events = events
        return self
    
    def create(self) -> 'ResourceUsagePlot':
        """創建資源使用情況圖表"""
        if not self.time_points or not self.memory_usage:
            raise ValueError("時間點或記憶體使用數據為空，無法繪製圖表")
        
        if len(self.time_points) != len(self.memory_usage):
            raise ValueError("時間點和記憶體使用數據長度不匹配")
        
        if self.gpu_usage and len(self.time_points) != len(self.gpu_usage):
            raise ValueError("時間點和GPU使用數據長度不匹配")
        
        if self.config.interactive and HAS_PLOTLY:
            # 使用Plotly創建交互式圖表
            self.plotly_fig = make_subplots(
                rows=2 if self.gpu_usage else 1, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=["Memory Usage"] + (["GPU Usage"] if self.gpu_usage else [])
            )
            
            # 添加記憶體使用曲線
            self.plotly_fig.add_trace(
                go.Scatter(
                    x=self.time_points,
                    y=self.memory_usage,
                    mode='lines',
                    name='Memory Usage (GB)',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # 添加GPU使用曲線
            if self.gpu_usage:
                self.plotly_fig.add_trace(
                    go.Scatter(
                        x=self.time_points,
                        y=self.gpu_usage,
                        mode='lines',
                        name='GPU Usage (%)',
                        line=dict(color='green')
                    ),
                    row=2, col=1
                )
            
            # 添加事件標記
            if self.events:
                for time_point, event in self.events.items():
                    # 找出最接近的時間點索引
                    idx = np.abs(np.array(self.time_points) - time_point).argmin()
                    
                    # 添加事件標記
                    self.plotly_fig.add_trace(
                        go.Scatter(
                            x=[self.time_points[idx]],
                            y=[self.memory_usage[idx]],
                            mode='markers+text',
                            marker=dict(size=10, color='red'),
                            text=[event],
                            textposition='top center',
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    
                    # 如果有GPU使用數據，也在GPU圖上添加標記
                    if self.gpu_usage:
                        self.plotly_fig.add_trace(
                            go.Scatter(
                                x=[self.time_points[idx]],
                                y=[self.gpu_usage[idx]],
                                mode='markers',
                                marker=dict(size=10, color='red'),
                                showlegend=False
                            ),
                            row=2, col=1
                        )
            
            # 更新軸標籤
            self.plotly_fig.update_xaxes(title_text="Time (s)", row=2 if self.gpu_usage else 1, col=1)
            self.plotly_fig.update_yaxes(title_text="Memory (GB)", row=1, col=1)
            
            if self.gpu_usage:
                self.plotly_fig.update_yaxes(title_text="GPU Usage (%)", row=2, col=1)
            
            # 更新佈局
            title = self.config.title or "Resource Usage"
            
            self.plotly_fig.update_layout(
                title_text=title,
                template=self.config.theme,
                height=600 if self.gpu_usage else 400,
                width=900,
                font=dict(
                    family=self.config.font_family,
                    size=self.config.font_size
                ),
                hovermode='closest'
            )
            
        else:
            # 使用Matplotlib創建靜態圖表
            self.config.apply_style()
            
            # 創建子圖
            if self.gpu_usage:
                self.fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figsize,
                                                 dpi=self.config.dpi, sharex=True)
            else:
                self.fig, ax1 = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            
            # 繪製記憶體使用曲線
            ax1.plot(self.time_points, self.memory_usage, 'b-', label='Memory Usage')
            ax1.set_ylabel('Memory (GB)')
            ax1.grid(True, alpha=0.3)
            ax1.set_title('Memory Usage')
            
            # 添加事件標記
            if self.events:
                for time_point, event in self.events.items():
                    # 找出最接近的時間點索引
                    idx = np.abs(np.array(self.time_points) - time_point).argmin()
                    
                    # 添加事件標記
                    ax1.plot(self.time_points[idx], self.memory_usage[idx], 'ro')
                    ax1.annotate(event, xy=(self.time_points[idx], self.memory_usage[idx]),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom',
                               arrowprops=dict(arrowstyle='->', color='red'))
            
            # 繪製GPU使用曲線
            if self.gpu_usage:
                ax2.plot(self.time_points, self.gpu_usage, 'g-', label='GPU Usage')
                ax2.set_ylabel('GPU Usage (%)')
                ax2.set_xlabel('Time (s)')
                ax2.grid(True, alpha=0.3)
                ax2.set_title('GPU Usage')
                
                # 如果有事件標記，也在GPU圖上添加
                if self.events:
                    for time_point, event in self.events.items():
                        idx = np.abs(np.array(self.time_points) - time_point).argmin()
                        ax2.plot(self.time_points[idx], self.gpu_usage[idx], 'ro')
            else:
                ax1.set_xlabel('Time (s)')
            
            # 總標題
            title = self.config.title or "Resource Usage"
            self.fig.suptitle(title, fontsize=16)
            
            # 調整佈局
            self.fig.tight_layout()
            self.fig.subplots_adjust(top=0.9)
        
        return self


class PlotFactory:
    """圖表工廠，用於創建不同類型的圖表"""
    
    @staticmethod
    def create(plot_type: str, **kwargs) -> PlotBase:
        """創建圖表
        
        Args:
            plot_type: 圖表類型
            **kwargs: 圖表配置參數
            
        Returns:
            圖表對象
        """
        # 創建配置
        config = PlotConfig(**kwargs)
        
        # 創建圖表
        plots = {
            'training_history': TrainingHistoryPlot,
            'charge_discharge': ChargeDischargePlot,
            'charge_discharge_summary': ChargeDischargeSummaryPlot,
            'cross_temp_evaluation': CrossTempEvaluationPlot,
            'temperature_comparison': TemperatureComparisonPlot,
            'transfer_capability': TransferCapabilityPlot,
            'resource_usage': ResourceUsagePlot,
            # 添加其他圖表類型...
        }
        
        if plot_type not in plots:
            raise ValueError(f"不支持的圖表類型: {plot_type}")
        
        return plots[plot_type](config)


class VisualizationManager:
    """視覺化管理器，提供多種圖表生成功能"""
    
    def __init__(self, save_dir=None, style='seaborn-v0_8-whitegrid', 
                dpi=300, default_figsize=(12, 8), interactive=False):
        """初始化視覺化管理器
        
        Args:
            save_dir: 保存目錄
            style: Matplotlib樣式
            dpi: 圖表分辨率
            default_figsize: 默認圖表大小
            interactive: 是否創建交互式圖表
        """
        self.config = PlotConfig(
            save_dir=save_dir,
            style=style,
            dpi=dpi,
            figsize=default_figsize,
            interactive=interactive
        )
        
        # 確保保存目錄存在
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 設置繪圖樣式
        plt.style.use(self.config.style)
        
        logger.info(f"視覺化管理器初始化完成，圖表將保存到: {self.config.save_dir}")
        
        # 檢查是否支持交互式圖表
        if interactive and not HAS_PLOTLY:
            logger.warning("缺少Plotly庫，無法創建交互式圖表。將使用靜態圖表。")
            self.config.interactive = False
    
    def create_plot(self, plot_type: str, **kwargs) -> PlotBase:
        """創建指定類型的圖表
        
        Args:
            plot_type: 圖表類型
            **kwargs: 圖表配置參數
            
        Returns:
            圖表對象
        """
        # 合併默認配置和自定義配置
        config_kwargs = self.config.__dict__.copy()
        config_kwargs.update(kwargs)
        
        return PlotFactory.create(plot_type, **config_kwargs)
    
    def plot_pipeline(self, data: Any) -> 'VisualizationPipeline':
        """創建視覺化管道
        
        Args:
            data: 初始數據
            
        Returns:
            視覺化管道對象
        """
        return VisualizationPipeline(data, self)
    
    def plot_trainer_results(self, trainer, experiment_name, temps=None):
        """繪製訓練器的所有結果圖表
        
        Args:
            trainer: 訓練器對象 (必須有 training_history, best_performances 屬性)
            experiment_name: 實驗名稱
            temps: 溫度列表 (None表示使用訓練器的所有溫度)
            
        Returns:
            圖表路徑列表
        """
        if not hasattr(trainer, 'training_history') or not hasattr(trainer, 'best_performances'):
            raise ValueError("訓練器缺少必要的屬性")
        
        # 獲取溫度列表
        if temps is None:
            temps = list(trainer.training_history.keys())
        
        # 繪製每個溫度的訓練歷史
        paths = []
        for temp in temps:
            if temp in trainer.training_history:
                history_plot = self.create_plot(
                    'training_history',
                    title=f"{experiment_name} - {temp} Training History",
                    filename=f"{experiment_name}_{temp}_history.png"
                )
                history_plot.set_data(trainer.training_history[temp])
                history_plot.create()
                paths.append(history_plot.save())
        
        # 如果有多個溫度，繪製比較圖表
        if len(temps) > 1:
            # 繪製溫度比較圖表
            temp_plot = self.create_plot(
                'temperature_comparison',
                title=f"{experiment_name} - Temperature Comparison",
                filename=f"{experiment_name}_temp_comparison.png"
            )
            temp_plot.set_data(trainer.best_performances)
            temp_plot.create()
            paths.append(temp_plot.save())
            
            # 如果有交叉評估結果，繪製交叉評估圖表
            if hasattr(trainer, 'cross_evaluation'):
                cross_plot = self.create_plot(
                    'cross_temp_evaluation',
                    title=f"{experiment_name} - Cross-Temperature Evaluation",
                    filename=f"{experiment_name}_cross_eval.png"
                )
                cross_plot.set_data(trainer.cross_evaluation)
                cross_plot.create()
                paths.append(cross_plot.save())
                
                # 繪製轉移能力圖表
                transfer_plot = self.create_plot(
                    'transfer_capability',
                    title=f"{experiment_name} - Transfer Capability",
                    filename=f"{experiment_name}_transfer_capability.png"
                )
                transfer_plot.set_data(trainer.cross_evaluation)
                transfer_plot.create()
                paths.append(transfer_plot.save())
        
        return paths
    
    def plot_charge_discharge_results(self, results, experiment_name, temp):
        """繪製充放電評估結果圖表
        
        Args:
            results: 充放電評估結果字典
            experiment_name: 實驗名稱
            temp: 溫度標識
            
        Returns:
            圖表路徑列表
        """
        paths = []
        
        # 繪製充放電摘要圖表
        summary_plot = self.create_plot(
            'charge_discharge_summary',
            title=f"{experiment_name} - Charge vs Discharge ({temp})",
            filename=f"{experiment_name}_{temp}_cd_summary.png"
        )
        summary_plot.set_data(results, temp)
        summary_plot.create()
        paths.append(summary_plot.save())
        
        return paths


class VisualizationPipeline:
    """視覺化管道，用於鏈式處理視覺化操作"""
    
    def __init__(self, data: Any, manager: VisualizationManager):
        """初始化視覺化管道
        
        Args:
            data: 初始數據
            manager: 視覺化管理器
        """
        self.data = data
        self.manager = manager
        self.plots = []
        self.transformations = []
    
    def transform(self, func: Callable[[Any], Any]) -> 'VisualizationPipeline':
        """轉換數據
        
        Args:
            func: 轉換函數
            
        Returns:
            自身，用於鏈式調用
        """
        self.transformations.append(func)
        return self
    
    def add_plot(self, plot_type: str, **kwargs) -> 'VisualizationPipeline':
        """添加圖表
        
        Args:
            plot_type: 圖表類型
            **kwargs: 圖表配置參數
            
        Returns:
            自身，用於鏈式調用
        """
        self.plots.append((plot_type, kwargs))
        return self
    
    def execute(self) -> List[str]:
        """執行視覺化管道
        
        Returns:
            圖表保存路徑列表
        """
        # 應用轉換
        current_data = self.data
        for transform in self.transformations:
            current_data = transform(current_data)
        
        # 創建並保存圖表
        save_paths = []
        for plot_type, kwargs in self.plots:
            plot = self.manager.create_plot(plot_type, **kwargs)
            
            # 根據圖表類型設置數據
            if plot_type == 'training_history':
                plot.set_data(current_data)
            elif plot_type == 'charge_discharge':
                plot.set_data(
                    current_data.get('charge_metrics', {}),
                    current_data.get('discharge_metrics', {}),
                    current_data.get('epochs', []),
                    metric_names=current_data.get('metric_names'),
                    temp=current_data.get('temp')
                )
            elif plot_type == 'charge_discharge_summary':
                plot.set_data(
                    current_data,
                    temp=kwargs.get('temp')
                )
            elif plot_type == 'cross_temp_evaluation':
                plot.set_data(
                    current_data,
                    metric=kwargs.get('metric', 'loss')
                )
            elif plot_type == 'temperature_comparison':
                plot.set_data(
                    current_data,
                    metric_names=kwargs.get('metric_names')
                )
            elif plot_type == 'transfer_capability':
                plot.set_data(
                    current_data,
                    metric=kwargs.get('metric', 'loss')
                )
            elif plot_type == 'resource_usage':
                plot.set_data(
                    current_data.get('time_points', []),
                    current_data.get('memory_usage', []),
                    current_data.get('gpu_usage'),
                    current_data.get('events')
                )
            # 處理其他圖表類型...
            
            # 創建並保存圖表
            plot.create()
            save_paths.append(plot.save())
        
        return save_paths


# 提供簡單的函數接口
def plot_training_history(history, title="Training History", filename=None, save_dir=None, **kwargs):
    """繪製訓練歷史曲線的便捷函數
    
    Args:
        history: 訓練歷史字典
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('training_history', title=title, filename=filename, **kwargs)
    plot.set_data(history)
    plot.create()
    return plot.save()

def plot_charge_discharge_metrics(charge_metrics, discharge_metrics, epochs, title=None, 
                                filename=None, save_dir=None, temp=None, **kwargs):
    """繪製充放電性能指標比較圖的便捷函數
    
    Args:
        charge_metrics: 充電指標字典
        discharge_metrics: 放電指標字典
        epochs: 輪次列表
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        temp: 溫度標識
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('charge_discharge', title=title, filename=filename, **kwargs)
    plot.set_data(charge_metrics, discharge_metrics, epochs, temp=temp)
    plot.create()
    return plot.save()

def plot_cross_temp_evaluation(cross_eval_data, metric='loss', title=None, 
                             filename=None, save_dir=None, **kwargs):
    """繪製交叉溫度評估圖的便捷函數
    
    Args:
        cross_eval_data: 交叉評估數據字典
        metric: 要顯示的指標
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('cross_temp_evaluation', title=title, filename=filename, **kwargs)
    plot.set_data(cross_eval_data, metric)
    plot.create()
    return plot.save()

def plot_temperature_comparison(metrics_by_temp, metric_names=None, title=None,
                              filename=None, save_dir=None, **kwargs):
    """繪製溫度比較圖的便捷函數
    
    Args:
        metrics_by_temp: 各溫度的指標字典
        metric_names: 要顯示的指標名稱列表
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('temperature_comparison', title=title, filename=filename, **kwargs)
    plot.set_data(metrics_by_temp, metric_names)
    plot.create()
    return plot.save()

def plot_charge_discharge_summary(eval_results, temp=None, title=None,
                                filename=None, save_dir=None, **kwargs):
    """繪製充放電性能摘要圖的便捷函數
    
    Args:
        eval_results: 評估結果字典
        temp: 溫度標識
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('charge_discharge_summary', title=title, filename=filename, **kwargs)
    plot.set_data(eval_results, temp)
    plot.create()
    return plot.save()

def plot_transfer_capability(cross_eval_data, metric='loss', title=None,
                           filename=None, save_dir=None, **kwargs):
    """繪製轉移能力圖的便捷函數
    
    Args:
        cross_eval_data: 交叉評估數據字典
        metric: 要顯示的指標
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('transfer_capability', title=title, filename=filename, **kwargs)
    plot.set_data(cross_eval_data, metric)
    plot.create()
    return plot.save()

def plot_resource_usage(time_points, memory_usage, gpu_usage=None, events=None,
                      title=None, filename=None, save_dir=None, **kwargs):
    """繪製資源使用情況圖的便捷函數
    
    Args:
        time_points: 時間點列表
        memory_usage: 記憶體使用列表
        gpu_usage: GPU使用列表
        events: 事件字典
        title: 圖表標題
        filename: 保存文件名
        save_dir: 保存目錄
        **kwargs: 其他配置參數
        
    Returns:
        圖表保存路徑
    """
    viz = VisualizationManager(save_dir=save_dir)
    plot = viz.create_plot('resource_usage', title=title, filename=filename, **kwargs)
    plot.set_data(time_points, memory_usage, gpu_usage, events)
    plot.create()
    return plot.save()


# 命令行界面
if __name__ == "__main__":
    import argparse
    import json
    
    # 設置命令行界面
    parser = argparse.ArgumentParser(description="電池老化預測系統視覺化工具")
    parser.add_argument("--input", type=str, required=True, help="輸入JSON文件路徑")
    parser.add_argument("--type", type=str, default="training_history", 
                       choices=["training_history", "charge_discharge", "charge_discharge_summary",
                               "cross_temp_evaluation", "temperature_comparison",
                               "transfer_capability", "resource_usage"],
                       help="圖表類型")
    parser.add_argument("--output", type=str, help="輸出圖片路徑")
    parser.add_argument("--title", type=str, default=None, help="圖表標題")
    parser.add_argument("--dpi", type=int, default=300, help="圖表DPI")
    parser.add_argument("--temp", type=str, default=None, help="溫度標識")
    parser.add_argument("--metric", type=str, default="loss", help="要顯示的指標")
    parser.add_argument("--interactive", action="store_true", help="創建交互式圖表")
    
    args = parser.parse_args()
    
    # 加載數據
    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"無法加載輸入文件: {e}")
        exit(1)
    
    # 創建視覺化管理器
    viz = VisualizationManager(dpi=args.dpi, interactive=args.interactive)
    
    # 創建圖表
    try:
        plot = viz.create_plot(
            args.type,
            title=args.title,
            filename=args.output
        )
        
        # 根據圖表類型設置數據
        if args.type == "training_history":
            plot.set_data(data)
        elif args.type == "charge_discharge":
            plot.set_data(
                data.get('charge_metrics', {}),
                data.get('discharge_metrics', {}),
                data.get('epochs', []),
                temp=args.temp
            )
        elif args.type == "charge_discharge_summary":
            plot.set_data(data, temp=args.temp)
        elif args.type == "cross_temp_evaluation":
            plot.set_data(data, metric=args.metric)
        elif args.type == "temperature_comparison":
            plot.set_data(data)
        elif args.type == "transfer_capability":
            plot.set_data(data, metric=args.metric)
        elif args.type == "resource_usage":
            plot.set_data(
                data.get('time_points', []),
                data.get('memory_usage', []),
                data.get('gpu_usage'),
                data.get('events')
            )
        
        # 創建並保存圖表
        plot.create()
        output_path = plot.save()
        print(f"圖表已保存到: {output_path}")
    
    except Exception as e:
        print(f"生成圖表時出錯: {e}")
        import traceback
        print(traceback.format_exc())
        exit(1)