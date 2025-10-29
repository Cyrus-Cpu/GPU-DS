# -*- coding: utf-8 -*-
"""
AI 算牌預測 v3.8.4 - GPU-DS-V2.1.1（高推薦率優化版）錯誤修復
GPU加速深度学习预测系统 - 完整版本
功能：动态数据生成、LSTM预测、多GPU训练、实时预测、强化学习整合、高级分析、自动化调优
作者：完整修复版本
日期：2025-10-29
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import json
import os
from collections import deque, defaultdict
import warnings
import psutil
import GPUtil
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import seaborn as sns
from scipy import stats
import math
import random
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedDynamicDataGenerator:
    """增强型动态数据生成器"""
    
    def __init__(self, sequence_length=50, features=10, trend_types=['linear', 'exponential', 'logarithmic']):
        self.sequence_length = sequence_length
        self.features = features
        self.trend_types = trend_types
        self.scaler = MinMaxScaler()
        self.noise_generators = {
            'gaussian': lambda size: np.random.randn(*size),
            'uniform': lambda size: np.random.uniform(-1, 1, size),
            'poisson': lambda size: np.random.poisson(1, size)
        }
        
    def generate_complex_trend(self, length, trend_type='linear', strength=0.1):
        """生成复杂趋势"""
        x = np.arange(length)
        
        if trend_type == 'linear':
            return strength * x
        elif trend_type == 'exponential':
            return strength * np.exp(0.01 * x)
        elif trend_type == 'logarithmic':
            return strength * np.log1p(x)
        elif trend_type == 'sigmoid':
            return strength / (1 + np.exp(-0.1 * (x - length/2)))
        else:
            return np.zeros(length)
    
    def generate_seasonality(self, length, periods=[30, 90, 365], amplitudes=[3, 2, 1]):
        """生成多周期季节性"""
        seasonal = np.zeros(length)
        t = np.arange(length)
        
        for period, amplitude in zip(periods, amplitudes):
            seasonal += amplitude * np.sin(2 * np.pi * t / period)
            seasonal += 0.5 * amplitude * np.sin(4 * np.pi * t / period)  # 谐波
            
        return seasonal
    
    def generate_regime_switching(self, length, n_regimes=3):
        """生成体制转换数据"""
        regimes = np.random.choice(n_regimes, length)
        data = np.zeros(length)
        
        for i in range(n_regimes):
            regime_mask = (regimes == i)
            regime_length = np.sum(regime_mask)
            if regime_length > 0:
                data[regime_mask] = np.random.normal(i, 0.5 + i*0.2, regime_length)
                
        return data
    
    def generate_synthetic_data(self, num_samples=1000, complexity='high'):
        """生成合成时间序列数据"""
        total_length = num_samples * self.sequence_length
        timestamps = np.arange(total_length)
        
        # 基础组件
        components = []
        
        # 1. 趋势组件
        if complexity == 'high':
            trend = sum(self.generate_complex_trend(total_length, trend_type) 
                       for trend_type in np.random.choice(self.trend_types, 2))
        else:
            trend = self.generate_complex_trend(total_length, 'linear', 0.1)
        
        components.append(trend)
        
        # 2. 季节性组件
        seasonal = self.generate_seasonality(total_length)
        components.append(seasonal)
        
        # 3. 体制转换（高复杂度时）
        if complexity == 'high':
            regime_data = self.generate_regime_switching(total_length)
            components.append(regime_data)
        
        # 4. 外部冲击
        shocks = np.zeros(total_length)
        shock_points = np.random.choice(total_length, total_length//100, replace=False)
        shocks[shock_points] = np.random.normal(0, 2, len(shock_points))
        components.append(shocks)
        
        # 组合主序列
        main_series = sum(components) + 0.1 * np.random.randn(total_length)
        
        # 生成多变量特征
        multivariate_data = []
        for i in range(self.features):
            if i == 0:
                feature = main_series
            else:
                lag = np.random.randint(1, 10)
                correlation = np.random.uniform(-0.8, 0.8)
                noise_type = np.random.choice(list(self.noise_generators.keys()))
                noise = self.noise_generators[noise_type]((total_length,))
                
                feature = correlation * np.roll(main_series, lag) + 0.3 * noise
                feature += 0.5 * np.sin(2 * np.pi * timestamps / (20 + i*5))
            
            multivariate_data.append(feature)
        
        data = np.array(multivariate_data).T
        data = self.scaler.fit_transform(data)
        
        # 转换为序列数据
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

class AdvancedLSTMWithAttention(nn.Module):
    """带注意力机制的高级LSTM"""
    
    def __init__(self, input_size, hidden_size=256, num_layers=4, output_size=1, 
                 dropout=0.3, bidirectional=True, use_attention=True):
        super(AdvancedLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout, 
                           bidirectional=bidirectional)
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * self.directions, hidden_size // 2),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 1),
                nn.Softmax(dim=1)
            )
        
        self.layer_norm = nn.LayerNorm(hidden_size * self.directions)
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size * self.directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        lstm_out, (hn, cn) = self.lstm(x)
        
        if self.use_attention:
            attention_weights = self.attention(lstm_out)
            context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        else:
            if self.bidirectional:
                context_vector = torch.cat([hn[-2], hn[-1]], dim=1)
            else:
                context_vector = hn[-1]
        
        context_vector = self.layer_norm(context_vector)
        output = self.output_layers(context_vector)
        return output

class DistributedMultiGPUTrainer:
    """分布式多GPU训练器"""
    
    def __init__(self, model, device_ids=None, distributed=False):
        self.model = model
        self.distributed = distributed
        self.device_ids = device_ids if device_ids else list(range(torch.cuda.device_count()))
        
        if distributed and len(self.device_ids) > 1:
            self.setup_distributed_training()
        elif len(self.device_ids) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
        
        self.device = torch.device(f'cuda:{self.device_ids[0]}' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_history = {
            'train_loss': [], 'val_loss': [], 'learning_rate': [],
            'grad_norm': [], 'epoch_time': []
        }
        
    def setup_distributed_training(self):
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        self.model = DDP(self.model, device_ids=self.device_ids)
        
    def create_optimizer(self, lr=0.001, optimizer_type='adam', weight_decay=1e-5):
        if optimizer_type == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def train_epoch(self, train_loader, optimizer, criterion, gradient_clip=1.0):
        self.model.train()
        total_loss = 0
        total_grad_norm = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            
            if gradient_clip > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                total_grad_norm += grad_norm.item()
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_grad_norm = total_grad_norm / len(train_loader) if gradient_clip > 0 else 0
        
        return avg_loss, avg_grad_norm
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += criterion(output.squeeze(), target).item()
                
        return total_loss / len(val_loader)
    
    def train_model(self, train_loader, val_loader=None, epochs=100, lr=0.001, 
                   patience=10, min_delta=1e-6, **kwargs):
        optimizer = self.create_optimizer(lr, kwargs.get('optimizer_type', 'adam'))
        criterion = nn.HuberLoss() if kwargs.get('use_huber', False) else nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=patience//2, factor=0.5, min_lr=1e-6
        )
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            train_loss, grad_norm = self.train_epoch(train_loader, optimizer, criterion, 
                                                   kwargs.get('gradient_clip', 1.0))
            
            val_loss = self.validate(val_loader, criterion) if val_loader else train_loss
            
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['learning_rate'].append(current_lr)
            self.train_history['grad_norm'].append(grad_norm)
            self.train_history['epoch_time'].append(epoch_time)
            
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if kwargs.get('save_best', False):
                    self.save_checkpoint(epoch, best_val_loss)
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                print(f"早停在epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f'Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.6f} | '
                      f'Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | '
                      f'Time: {epoch_time:.2f}s')
        
        return self.train_history
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.create_optimizer().state_dict(),
            'loss': loss,
            'train_history': self.train_history
        }
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

class EnhancedRealTimePredictor:
    """增强型实时预测引擎"""
    
    def __init__(self, model, sequence_length=50, confidence_level=0.95, 
                 uncertainty_method='bayesian'):
        self.model = model
        self.sequence_length = sequence_length
        self.confidence_level = confidence_level
        self.uncertainty_method = uncertainty_method
        self.recent_data = deque(maxlen=sequence_length * 2)
        self.prediction_history = []
        self.uncertainty_history = []
        self.model.eval()
        
        self.monte_carlo_samples = 100 if uncertainty_method == 'bayesian' else 1
        
    def update_data(self, new_data_point, timestamp=None):
        data_point = {
            'value': new_data_point,
            'timestamp': timestamp or time.time(),
            'features': self._extract_features(new_data_point)
        }
        self.recent_data.append(data_point)
        
    def _extract_features(self, value):
        if len(self.recent_data) < 10:
            return {'mean': value, 'std': 0, 'trend': 0}
        
        recent_values = [d['value'] for d in list(self.recent_data)[-10:]]
        return {
            'mean': np.mean(recent_values),
            'std': np.std(recent_values),
            'trend': self._calculate_trend(recent_values),
            'volatility': self._calculate_volatility(recent_values)
        }
    
    def _calculate_trend(self, values):
        if len(values) < 2:
            return 0
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope
    
    def _calculate_volatility(self, values):
        if len(values) < 2:
            return 0
        returns = np.diff(values) / values[:-1]
        return np.std(returns) if len(returns) > 0 else 0
    
    def predict_next(self, n_steps=1, return_uncertainty=True):
        if len(self.recent_data) < self.sequence_length:
            return None, None
        
        recent_values = [d['value'] for d in list(self.recent_data)[-self.sequence_length:]]
        predictions = []
        uncertainties = []
        
        current_sequence = np.array(recent_values, dtype=np.float32)
        
        for step in range(n_steps):
            if return_uncertainty:
                pred, uncertainty = self._predict_with_uncertainty(current_sequence)
                uncertainties.append(uncertainty)
            else:
                pred = self._predict_single(current_sequence)
                
            predictions.append(pred)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        if return_uncertainty:
            return predictions, uncertainties
        return predictions, None
    
    def _predict_single(self, sequence):
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            prediction = self.model(sequence_tensor).item()
        return prediction
    
    def _predict_with_uncertainty(self, sequence):
        if self.uncertainty_method == 'bayesian':
            return self._monte_carlo_dropout(sequence)
        elif self.uncertainty_method == 'ensemble':
            return self._ensemble_uncertainty(sequence)
        else:
            pred = self._predict_single(sequence)
            return pred, (pred * 0.1, pred * 0.1)
    
    def _monte_carlo_dropout(self, sequence):
        predictions = []
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).unsqueeze(-1)
        
        self.model.train()
        
        for _ in range(self.monte_carlo_samples):
            with torch.no_grad():
                pred = self.model(sequence_tensor).item()
                predictions.append(pred)
        
        self.model.eval()
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return mean_pred, (lower, upper)
    
    def get_prediction_metrics(self):
        if len(self.prediction_history) < 2:
            return {}
        
        actual = [p['actual'] for p in self.prediction_history if 'actual' in p]
        predicted = [p['predicted'] for p in self.prediction_history if 'actual' in p]
        
        if len(actual) < 2:
            return {}
        
        return {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'r2': r2_score(actual, predicted),
            'accuracy': np.mean([1 if abs(a-p)/a < 0.1 else 0 for a, p in zip(actual, predicted)])
        }

class AdvancedReinforcementLearning:
    """高级强化学习系统"""
    
    def __init__(self, state_size=10, action_size=5, learning_method='q_learning'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_method = learning_method
        
        if learning_method == 'q_learning':
            self.q_table = np.random.uniform(-1, 1, (state_size, action_size))
        elif learning_method == 'sarsa':
            self.q_table = np.random.uniform(-1, 1, (state_size, action_size))
        
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        self.training_history = []
        
    def discretize_state(self, continuous_state, bins=10):
        if isinstance(continuous_state, (list, np.ndarray)):
            discretized = []
            for i, value in enumerate(continuous_state):
                discretized.append(np.digitize(value, np.linspace(-1, 1, bins)) - 1)
            state_index = sum([d * (bins ** i) for i, d in enumerate(discretized)])
            return min(state_index, self.state_size - 1)
        else:
            return min(int((continuous_state + 1) / 2 * (self.state_size - 1)), self.state_size - 1)
    
    def choose_action(self, state):
        discrete_state = self.discretize_state(state)
        
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        return np.argmax(self.q_table[discrete_state])
    
    def update_policy(self, state, action, reward, next_state, next_action=None):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        if self.learning_method == 'q_learning':
            best_next_action = np.argmax(self.q_table[discrete_next_state])
            td_target = reward + self.discount_factor * self.q_table[discrete_next_state][best_next_action]
            td_error = td_target - self.q_table[discrete_state][action]
            self.q_table[discrete_state][action] += self.learning_rate * td_error
            
        elif self.learning_method == 'sarsa':
            if next_action is None:
                next_action = self.choose_action(next_state)
            td_target = reward + self.discount_factor * self.q_table[discrete_next_state][next_action]
            td_error = td_target - self.q_table[discrete_state][action]
            self.q_table[discrete_state][action] += self.learning_rate * td_error
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.training_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'q_value': self.q_table[discrete_state][action]
        })
    
    def get_action_meaning(self, action):
        actions = {
            0: "保守预测-下调10%",
            1: "轻微下调-下调5%", 
            2: "保持原预测",
            3: "轻微上调-上调5%",
            4: "激进预测-上调10%"
        }
        return actions.get(action, "未知动作")
    
    def get_policy_analysis(self):
        if len(self.training_history) == 0:
            return {}
        
        recent_history = self.training_history[-100:]
        
        return {
            'average_reward': np.mean([h['reward'] for h in recent_history]),
            'exploration_rate': self.epsilon,
            'q_value_range': [np.min(self.q_table), np.max(self.q_table)],
            'action_distribution': np.bincount([h['action'] for h in recent_history], minlength=self.action_size)
        }

class ComprehensiveMemoryManager:
    """综合内存管理系统"""
    
    def __init__(self, max_gpu_usage=0.8, max_cpu_usage=0.8, monitoring_interval=5):
        self.max_gpu_usage = max_gpu_usage
        self.max_cpu_usage = max_cpu_usage
        self.monitoring_interval = monitoring_interval
        self.memory_history = defaultdict(list)
        self.monitoring_thread = None
        self.stop_monitoring = False
        
    def start_monitoring(self):
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_resources)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
    def _monitor_resources(self):
        while not self.stop_monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = psutil.virtual_memory()
                
                self.memory_history['cpu_usage'].append(cpu_percent)
                self.memory_history['ram_usage'].append(memory_info.percent)
                self.memory_history['ram_used_gb'].append(memory_info.used / 1024**3)
                
                if torch.cuda.is_available():
                    gpus = GPUtil.getGPUs()
                    for i, gpu in enumerate(gpus):
                        self.memory_history[f'gpu_{i}_usage'].append(gpu.load * 100)
                        self.memory_history[f'gpu_{i}_memory'].append(gpu.memoryUtil * 100)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                time.sleep(self.monitoring_interval)
    
    def get_memory_stats(self):
        stats = {
            'cpu_usage': psutil.cpu_percent(),
            'ram_usage': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1024**3
        }
        
        if torch.cuda.is_available():
            stats.update({
                'gpu_available': True,
                'gpu_count': torch.cuda.device_count(),
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_cached': torch.cuda.memory_reserved() / 1024**3
            })
            
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                stats.update({
                    f'gpu_{i}_name': gpu.name,
                    f'gpu_{i}_usage': gpu.load * 100,
                    f'gpu_{i}_memory_usage': gpu.memoryUtil * 100,
                    f'gpu_{i}_temperature': gpu.temperature
                })
        else:
            stats['gpu_available'] = False
            
        return stats
    
    def clear_gpu_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU缓存已清空")
    
    def optimize_memory_usage(self, model, data_loader):
        if not torch.cuda.is_available():
            return
        
        current_stats = self.get_memory_stats()
        gpu_memory_usage = current_stats.get('gpu_memory_allocated', 0) + current_stats.get('gpu_memory_cached', 0)
                                                                                            
def optimize_memory_usage(self, model, data_loader):
        """优化内存使用"""
        if not torch.cuda.is_available():
            return
        
        current_stats = self.get_memory_stats()
        gpu_memory_usage = current_stats.get('gpu_memory_allocated', 0) + current_stats.get('gpu_memory_cached', 0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_memory_usage / gpu_memory_total > self.max_gpu_usage:
            logger.warning("GPU内存使用过高，正在优化...")
            self.clear_gpu_cache()
            
            # 调整批量大小
            if hasattr(data_loader, 'batch_size'):
                new_batch_size = max(1, data_loader.batch_size // 2)
                logger.info(f"将批量大小从 {data_loader.batch_size} 调整为 {new_batch_size}")
                
        # 设置GPU内存增长策略
        torch.cuda.set_per_process_memory_fraction(self.max_gpu_usage)

class AdvancedAnalyticsTools:
    """高级分析工具集"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_std=None):
        """计算全面的评估指标"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': self._calculate_mape(y_true, y_pred),
            'smape': self._calculate_smape(y_true, y_pred),
            'rmse_std_ratio': self._calculate_rmse_std_ratio(y_true, y_pred)
        }
        
        if y_pred_std is not None:
            metrics.update({
                'sharp_ratio': self._calculate_sharp_ratio(y_true, y_pred, y_pred_std),
                'coverage_probability': self._calculate_coverage_probability(y_true, y_pred, y_pred_std),
                'mean_std': np.mean(y_pred_std)
            })
            
        return metrics
    
    def _calculate_mape(self, y_true, y_pred):
        """计算平均绝对百分比误差"""
        return np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    
    def _calculate_smape(self, y_true, y_pred):
        """计算对称平均绝对百分比误差"""
        return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
    
    def _calculate_rmse_std_ratio(self, y_true, y_pred):
        """计算RMSE与标准差的比率"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        std_true = np.std(y_true)
        return rmse / std_true if std_true > 0 else float('inf')
    
    def _calculate_sharp_ratio(self, y_true, y_pred, y_pred_std):
        """计算夏普比率"""
        returns = (y_pred - y_true) / np.maximum(np.abs(y_true), 1e-8)
        return np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    
    def _calculate_coverage_probability(self, y_true, y_pred, y_pred_std, z_score=1.96):
        """计算覆盖概率"""
        lower_bound = y_pred - z_score * y_pred_std
        upper_bound = y_pred + z_score * y_pred_std
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))
        return coverage
    
    def perform_residual_analysis(self, y_true, y_pred):
        """残差分析"""
        residuals = y_true - y_pred
        
        analysis = {
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals),
            'normality_test_pvalue': stats.normaltest(residuals).pvalue,
            'autocorrelation_lags': self._calculate_autocorrelation(residuals, max_lag=10)
        }
        
        return analysis
    
    def _calculate_autocorrelation(self, residuals, max_lag=10):
        """计算自相关"""
        autocorrs = {}
        for lag in range(1, max_lag + 1):
            if len(residuals) > lag:
                autocorrs[lag] = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
        return autocorrs
    
    def create_advanced_visualizations(self, y_true, y_pred, history=None, save_path=None):
        """创建高级可视化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 预测vs实际
        axes[0, 0].plot(y_true, label='实际值', alpha=0.7)
        axes[0, 0].plot(y_pred, label='预测值', alpha=0.7)
        axes[0, 0].set_title('预测 vs 实际')
        axes[0, 0].legend()
        
        # 2. 残差图
        residuals = y_true - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('预测值')
        axes[0, 1].set_ylabel('残差')
        axes[0, 1].set_title('残差图')
        
        # 3. 训练历史（如果有）
        if history is not None:
            axes[1, 0].plot(history.get('train_loss', []), label='训练损失')
            axes[1, 0].plot(history.get('val_loss', []), label='验证损失')
            axes[1, 0].set_title('训练历史')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
            
        # 4. 分布比较
        axes[1, 1].hist(y_true, alpha=0.5, label='实际分布', bins=20)
        axes[1, 1].hist(y_pred, alpha=0.5, label='预测分布', bins=20)
        axes[1, 1].set_title('分布比较')
        axes[1, 1].legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class AutomatedHyperparameterOptimizer:
    """自动化超参数优化器"""
    
    def __init__(self, search_space, optimization_method='bayesian', n_trials=50):
        self.search_space = search_space
        self.optimization_method = optimization_method
        self.n_trials = n_trials
        self.best_params = None
        self.best_score = float('inf')
        self.trial_history = []
        
    def optimize(self, model_class, data_generator, **kwargs):
        """执行超参数优化"""
        if self.optimization_method == 'random':
            return self._random_search(model_class, data_generator, **kwargs)
        elif self.optimization_method == 'grid':
            return self._grid_search(model_class, data_generator, **kwargs)
        elif self.optimization_method == 'bayesian':
            return self._bayesian_optimization(model_class, data_generator, **kwargs)
        else:
            raise ValueError(f"不支持的优化方法: {self.optimization_method}")
    
    def _random_search(self, model_class, data_generator, **kwargs):
        """随机搜索"""
        for trial in range(self.n_trials):
            params = self._sample_parameters()
            score = self._evaluate_parameters(model_class, data_generator, params, **kwargs)
            
            self.trial_history.append({'params': params, 'score': score})
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"试验 {trial}: 新最佳分数 {score:.4f}")
                
        return self.best_params, self.best_score
    
    def _grid_search(self, model_class, data_generator, **kwargs):
        """网格搜索"""
        from itertools import product
        
        param_combinations = list(product(*[
            values for values in self.search_space.values()
        ]))
        
        for i, param_values in enumerate(param_combinations):
            params = dict(zip(self.search_space.keys(), param_values))
            score = self._evaluate_parameters(model_class, data_generator, params, **kwargs)
            
            self.trial_history.append({'params': params, 'score': score})
            
            if score < self.best_score:
                self.best_score = score
                self.best_params = params
                print(f"组合 {i}: 新最佳分数 {score:.4f}")
                
        return self.best_params, self.best_score
    
    def _bayesian_optimization(self, model_class, data_generator, **kwargs):
        """贝叶斯优化"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            
            # 定义搜索空间
            dimensions = []
            for param_name, param_values in self.search_space.items():
                if isinstance(param_values[0], int):
                    dimensions.append(Integer(param_values[0], param_values[-1], name=param_name))
                elif isinstance(param_values[0], float):
                    dimensions.append(Real(param_values[0], param_values[-1], name=param_name))
                else:
                    dimensions.append(Categorical(param_values, name=param_name))
            
            @use_named_args(dimensions)
            def objective(**params):
                score = self._evaluate_parameters(model_class, data_generator, params, **kwargs)
                return score
            
            result = gp_minimize(objective, dimensions, n_calls=self.n_trials, random_state=42)
            self.best_params = dict(zip(self.search_space.keys(), result.x))
            self.best_score = result.fun
            
            return self.best_params, self.best_score
            
        except ImportError:
            print("skopt未安装，使用随机搜索代替")
            return self._random_search(model_class, data_generator, **kwargs)
    
    def _sample_parameters(self):
        """采样参数"""
        params = {}
        for param_name, param_values in self.search_space.items():
            if isinstance(param_values[0], (int, float)):
                params[param_name] = np.random.choice(param_values)
            else:
                params[param_name] = random.choice(param_values)
        return params
    
    def _evaluate_parameters(self, model_class, data_generator, params, **kwargs):
        """评估参数组合"""
        try:
            # 生成数据
            X, y = data_generator.generate_synthetic_data(1000)
            
            # 分割数据
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # 创建模型
            model = model_class(
                input_size=X.shape[2],
                hidden_size=params.get('hidden_size', 128),
                num_layers=params.get('num_layers', 3),
                dropout=params.get('dropout', 0.2)
            )
            
            # 训练模型
            trainer = DistributedMultiGPUTrainer(model)
            train_loader = DataLoader(list(zip(X_train, y_train)), 
                                    batch_size=params.get('batch_size', 32), 
                                    shuffle=True)
            val_loader = DataLoader(list(zip(X_val, y_val)), 
                                 batch_size=params.get('batch_size', 32))
            
            history = trainer.train_model(
                train_loader, val_loader, 
                epochs=kwargs.get('eval_epochs', 50),
                lr=params.get('learning_rate', 0.001)
            )
            
            # 返回最佳验证损失
            return min(history['val_loss'])
            
        except Exception as e:
            print(f"参数评估失败: {e}")
            return float('inf')
    
    def plot_optimization_history(self):
        """绘制优化历史"""
        if not self.trial_history:
            return
        
        scores = [trial['score'] for trial in self.trial_history]
        plt.figure(figsize=(10, 6))
        plt.plot(scores, 'o-', alpha=0.7)
        plt.xlabel('试验次数')
        plt.ylabel('验证损失')
        plt.title('超参数优化历史')
        plt.grid(True, alpha=0.3)
        plt.show()

class PredictionSystemGUI:
    """完整的GUI预测系统"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU加速深度学习预测系统 - 完整版")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 系统组件
        self.data_generator = None
        self.model = None
        self.trainer = None
        self.predictor = None
        self.rl_agent = None
        self.memory_manager = None
        self.analytics_tools = AdvancedAnalyticsTools()
        
        # 设置样式
        self.setup_styles()
        self.setup_gui()
        
        # 启动内存监控
        self.memory_manager = ComprehensiveMemoryManager()
        self.memory_manager.start_monitoring()
        
    def setup_styles(self):
        """设置GUI样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 配置样式
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0')
        style.configure('Custom.TButton', font=('Arial', 10), padding=5)
        style.configure('Status.TLabel', font=('Arial', 9), background='#e0e0e0')
        
    def setup_gui(self):
        """设置GUI界面"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="GPU加速深度学习预测系统", style='Title.TLabel')
        title_label.pack(pady=10)
        
        # 创建标签页
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 系统控制标签页
        control_frame = ttk.Frame(notebook, padding="10")
        self.setup_control_tab(control_frame)
        notebook.add(control_frame, text="系统控制")
        
        # 训练监控标签页
        training_frame = ttk.Frame(notebook, padding="10")
        self.setup_training_tab(training_frame)
        notebook.add(training_frame, text="训练监控")
        
        # 实时预测标签页
        prediction_frame = ttk.Frame(notebook, padding="10")
        self.setup_prediction_tab(prediction_frame)
        notebook.add(prediction_frame, text="实时预测")
        
        # 分析工具标签页
        analytics_frame = ttk.Frame(notebook, padding="10")
        self.setup_analytics_tab(analytics_frame)
        notebook.add(analytics_frame, text="分析工具")
        
        # 系统状态标签页
        status_frame = ttk.Frame(notebook, padding="10")
        self.setup_status_tab(status_frame)
        notebook.add(status_frame, text="系统状态")
        
        # 状态栏
        self.setup_status_bar(main_frame)
    
    def setup_control_tab(self, parent):
        """设置系统控制标签页"""
        # 系统初始化区域
        init_frame = ttk.LabelFrame(parent, text="系统初始化", padding="10")
        init_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(init_frame, text="初始化完整系统", 
                  command=self.initialize_complete_system, style='Custom.TButton').pack(pady=5)
        
        # 数据生成区域
        data_frame = ttk.LabelFrame(parent, text="数据管理", padding="10")
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(data_frame, text="生成训练数据", 
                  command=self.generate_training_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_frame, text="加载外部数据", 
                  command=self.load_external_data).pack(side=tk.LEFT, padx=5)
        
        # 模型训练区域
        train_frame = ttk.LabelFrame(parent, text="模型训练", padding="10")
        train_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(train_frame, text="开始训练", 
                  command=self.start_training).pack(side=tk.LEFT, padx=5)
        ttk.Button(train_frame, text="超参数优化", 
                  command=self.run_hyperparameter_optimization).pack(side=tk.LEFT, padx=5)
        
        # 实时预测区域
        pred_frame = ttk.LabelFrame(parent, text="预测引擎", padding="10")
        pred_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(pred_frame, text="启动实时预测", 
                  command=self.start_realtime_prediction).pack(side=tk.LEFT, padx=5)
        ttk.Button(pred_frame, text="批量预测", 
                  command=self.run_batch_prediction).pack(side=tk.LEFT, padx=5)
    
    def setup_training_tab(self, parent):
        """设置训练监控标签页"""
        # 训练进度区域
        progress_frame = ttk.LabelFrame(parent, text="训练进度", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.train_text = scrolledtext.ScrolledText(progress_frame, height=15, width=80)
        self.train_text.pack(fill=tk.BOTH, expand=True)
        
        # 训练控制按钮
        control_frame = ttk.Frame(progress_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="清空日志", 
                  command=self.clear_training_log).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="导出结果", 
                  command=self.export_training_results).pack(side=tk.LEFT, padx=5)
    
    def setup_prediction_tab(self, parent):
        """设置实时预测标签页"""
        # 预测显示区域
        pred_display_frame = ttk.LabelFrame(parent, text="预测结果", padding="10")
        pred_display_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.pred_text = scrolledtext.ScrolledText(pred_display_frame, height=10, width=80)
        self.pred_text.pack(fill=tk.BOTH, expand=True)
        
        # 预测控制区域
        pred_control_frame = ttk.LabelFrame(parent, text="预测控制", padding="10")
        pred_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(pred_control_frame, text="开始模拟预测", 
                  command=self.start_prediction_simulation).pack(side=tk.LEFT, padx=5)
        ttk.Button(pred_control_frame, text="停止预测", 
                  command=self.stop_prediction).pack(side=tk.LEFT, padx=5)
        
        # 强化学习控制
        rl_frame = ttk.LabelFrame(parent, text="强化学习", padding="10")
        rl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rl_frame, text="启用RL优化", 
                  command=self.enable_rl_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(rl_frame, text="RL策略分析", 
                  command=self.analyze_rl_policy).pack(side=tk.LEFT, padx=5)
    
    def setup_analytics_tab(self, parent):
        """设置分析工具标签页"""
        # 分析按钮区域
        analysis_buttons_frame = ttk.LabelFrame(parent, text="分析工具", padding="10")
        analysis_buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(analysis_buttons_frame, text="模型评估", 
                  command=self.run_model_evaluation).pack(side=tk.LEFT, padx=5)
        ttk.Button(analysis_buttons_frame, text="残差分析", 
                  command=self.run_residual_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(analysis_buttons_frame, text="生成报告", 
                  command=self.generate_analysis_report).pack(side=tk.LEFT, padx=5)
        
        # 结果显示区域
        results_frame = ttk.LabelFrame(parent, text="分析结果", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.analysis_text = scrolledtext.ScrolledText(results_frame, height=15, width=80)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_status_tab(self, parent):
        """设置系统状态标签页"""
        # 资源监控区域
        resource_frame = ttk.LabelFrame(parent, text="资源监控", padding="10")
        resource_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.status_text = scrolledtext.ScrolledText(resource_frame, height=20, width=80)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # 系统控制按钮
        system_control_frame = ttk.Frame(resource_frame)
        system_control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(system_control_frame, text="刷新状态", 
                  command=self.update_system_status).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_control_frame, text="优化内存", 
                  command=self.optimize_system_memory).pack(side=tk.LEFT, padx=5)
        ttk.Button(system_control_frame, text="系统信息", 
                  command=self.show_system_info).pack(side=tk.LEFT, padx=5)
    
    def setup_status_bar(self, parent):
        """设置状态栏"""
        status_bar = ttk.Frame(parent, relief=tk.SUNKEN, height=20)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = ttk.Label(status_bar, text="系统就绪", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.memory_label = ttk.Label(status_bar, text="内存: --", style='Status.TLabel')
        self.memory_label.pack(side=tk.RIGHT, padx=5)
        
        # 启动状态更新
        self.update_status_bar()
    
    def update_status_bar(self):
        """更新状态栏"""
        if self.memory_manager:
            stats = self.memory_manager.get_memory_stats()
            memory_text = f"内存: {stats['ram_usage']:.1f}%"
            if stats['gpu_available']:
                memory_text += f" | GPU: {stats['gpu_0_usage']:.1f}%"
            self.memory_label.config(text=memory_text)
        
        self.root.after(2000, self.update_status_bar)  # 每2秒更新一次
    
    def initialize_complete_system(self):
        """初始化完整系统"""
        def _initialize():
            self.log_status("正在初始化完整系统...")
            
            try:
                # 1. 初始化数据生成器
                self.data_generator = EnhancedDynamicDataGenerator(
                    sequence_length=50, features=10
                )
                self.log_status("✓ 数据生成器已初始化")
                
                # 2. 生成训练数据
                X, y = self.data_generator.generate_synthetic_data(2000, complexity='high')
                self.log_status(f"✓ 训练数据已生成: {X.shape}")
                
                # 3. 创建模型
                self.model = AdvancedLSTMWithAttention(
                    input_size=X.shape[2],
                    hidden_size=256,
                    num_layers=4,
                    dropout=0.3
                )
                self.log_status("✓ LSTM模型已创建")
                
                # 4. 初始化训练器
                self.trainer = DistributedMultiGPUTrainer(self.model)
                self.log_status("✓ 多GPU训练器已初始化")
                
                # 5. 初始化预测器
                self.predictor = EnhancedRealTimePredictor(self.model)
                self.log_status("✓ 实时预测器已初始化")
                
                # 6. 初始化强化学习
                self.rl_agent = AdvancedReinforcementLearning()
                self.log_status("✓ 强化学习代理已初始化")
                
                self.log_status("🎉 系统初始化完成！")
                
            except Exception as e:
                self.log_status(f"❌ 初始化失败: {e}")
        
        threading.Thread(target=_initialize, daemon=True).start()
    
    def start_training(self):
        """开始训练"""
        def _train():
            try:
                if not self.data_generator or not self.model:
                    self.log_status("请先初始化系统")
                    return
                
                self.log_status("开始训练模型...")
                
                # 生成数据
                X, y = self.data_generator.generate_synthetic_data(2000)
                
                # 创建数据加载器
                dataset = list(zip(X, y))
                train_size = int(0.8 * len(dataset))
                train_dataset, val_dataset = dataset[:train_size], dataset[train_size:]
                
                train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=64)
                
                # 训练模型
                history = self.trainer.train_model(
                    train_loader, val_loader, 
                    epochs=200, lr=0.001, patience=20,
                    save_best=True
                )
                
                self.log_status("训练完成！")
                
                # 显示训练结果
                self.display_training_results(history)
                
            except Exception as e:
                self.log_status(f"训练失败: {e}")

        def display_training_results(self, history):
        """显示训练结果"""
        self.log_status("\n=== 训练结果 ===")
        self.log_status(f"最终训练损失: {history['train_loss'][-1]:.6f}")
        self.log_status(f"最终验证损失: {history['val_loss'][-1]:.6f}")
        self.log_status(f"最佳验证损失: {min(history['val_loss']):.6f}")
        self.log_status(f"总训练时间: {sum(history['epoch_time']):.2f}秒")
        
        # 绘制训练曲线
        self.plot_training_curves(history)
    
    def plot_training_curves(self, history):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='训练损失')
        axes[0, 0].plot(history['val_loss'], label='验证损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 学习率变化
        axes[0, 1].plot(history['learning_rate'])
        axes[0, 1].set_title('学习率变化')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 梯度范数
        axes[1, 0].plot(history['grad_norm'])
        axes[1, 0].set_title('梯度范数')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 每个epoch时间
        axes[1, 1].plot(history['epoch_time'])
        axes[1, 1].set_title('每个epoch时间(秒)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_training_data(self):
        """生成训练数据"""
        def _generate():
            try:
                self.log_status("正在生成训练数据...")
                X, y = self.data_generator.generate_synthetic_data(2000, complexity='high')
                self.log_status(f"数据生成完成: X.shape={X.shape}, y.shape={y.shape}")
            except Exception as e:
                self.log_status(f"数据生成失败: {e}")
        
        threading.Thread(target=_generate, daemon=True).start()
    
    def run_hyperparameter_optimization(self):
        """运行超参数优化"""
        def _optimize():
            try:
                self.log_status("开始超参数优化...")
                
                search_space = {
                    'hidden_size': [64, 128, 256, 512],
                    'num_layers': [2, 3, 4, 5],
                    'dropout': [0.1, 0.2, 0.3, 0.4],
                    'learning_rate': [0.001, 0.0005, 0.0001],
                    'batch_size': [32, 64, 128]
                }
                
                optimizer = AutomatedHyperparameterOptimizer(
                    search_space, optimization_method='random', n_trials=20
                )
                
                best_params, best_score = optimizer.optimize(
                    AdvancedLSTMWithAttention, self.data_generator
                )
                
                self.log_status(f"优化完成！最佳分数: {best_score:.6f}")
                self.log_status(f"最佳参数: {best_params}")
                
                # 使用最佳参数重新训练模型
                self.log_status("使用最佳参数重新训练模型...")
                X, y = self.data_generator.generate_synthetic_data(2000)
                
                self.model = AdvancedLSTMWithAttention(
                    input_size=X.shape[2],
                    hidden_size=best_params['hidden_size'],
                    num_layers=best_params['num_layers'],
                    dropout=best_params['dropout']
                )
                
                self.trainer = DistributedMultiGPUTrainer(self.model)
                self.predictor = EnhancedRealTimePredictor(self.model)
                
            except Exception as e:
                self.log_status(f"超参数优化失败: {e}")
        
        threading.Thread(target=_optimize, daemon=True).start()
    
    def start_realtime_prediction(self):
        """开始实时预测"""
        def _predict():
            try:
                self.log_status("启动实时预测引擎...")
                
                # 生成测试数据流
                test_stream = self._generate_test_stream(500)
                
                predictions = []
                actuals = []
                confidence_intervals = []
                
                for i, (data_point, actual) in enumerate(test_stream):
                    if i % 10 == 0:
                        self.log_status(f"处理第 {i} 个数据点...")
                    
                    # 更新预测器数据
                    self.predictor.update_data(data_point)
                    
                    # 进行预测
                    pred, confidence = self.predictor.predict_next(n_steps=1, return_uncertainty=True)
                    
                    if pred is not None:
                        predictions.append(pred[0])
                        actuals.append(actual)
                        confidence_intervals.append(confidence[0] if confidence else (pred[0], pred[0]))
                        
                        # 使用强化学习调整预测
                        if self.rl_agent and len(predictions) > 10:
                            self._apply_rl_correction(predictions, actuals, i)
                    
                    # 每50个点显示一次结果
                    if i > 0 and i % 50 == 0:
                        self._display_prediction_stats(predictions, actuals, i)
                    
                    time.sleep(0.1)  # 模拟实时数据流
                
                self.log_status("实时预测完成！")
                
                # 最终评估
                self._evaluate_predictions(predictions, actuals)
                
            except Exception as e:
                self.log_status(f"实时预测失败: {e}")
        
        threading.Thread(target=_predict, daemon=True).start()
    
    def _generate_test_stream(self, n_points):
        """生成测试数据流"""
        # 生成基础数据
        X, y = self.data_generator.generate_synthetic_data(n_points // 50 + 1)
        data_flat = X.reshape(-1, X.shape[-1])
        
        # 创建数据流
        for i in range(min(n_points, len(data_flat))):
            yield data_flat[i, 0], y[i // 50] if i % 50 == 0 else data_flat[i-1, 0]
    
    def _apply_rl_correction(self, predictions, actuals, step):
        """应用强化学习修正"""
        if len(predictions) < 2:
            return
        
        # 计算最近预测的误差
        recent_errors = [abs(p - a) for p, a in zip(predictions[-10:], actuals[-10:])]
        avg_error = np.mean(recent_errors)
        
        # 状态：基于最近误差和趋势
        state = [avg_error, np.std(recent_errors), predictions[-1] - predictions[-2]]
        
        # 选择动作
        action = self.rl_agent.choose_action(state)
        
        # 应用动作修正
        correction_factors = {0: 0.9, 1: 0.95, 2: 1.0, 3: 1.05, 4: 1.1}
        correction = correction_factors.get(action, 1.0)
        
        # 应用修正到最新预测
        predictions[-1] = predictions[-1] * correction
        
        # 计算奖励（负的误差改进）
        if len(predictions) > 1:
            old_error = abs(predictions[-2] - actuals[-2])
            new_error = abs(predictions[-1] - actuals[-1])
            reward = old_error - new_error  # 误差减少为正奖励
            
            # 更新RL策略
            next_state = [new_error, np.std([abs(p - a) for p, a in zip(predictions[-9:], actuals[-9:])]), 0]
            self.rl_agent.update_policy(state, action, reward, next_state)
    
    def _display_prediction_stats(self, predictions, actuals, step):
        """显示预测统计"""
        if len(predictions) < 2:
            return
        
        metrics = self.analytics_tools.calculate_comprehensive_metrics(
            np.array(actuals), np.array(predictions)
        )
        
        self.log_status(f"\n--- 第 {step} 步预测统计 ---")
        self.log_status(f"MAE: {metrics['mae']:.4f}")
        self.log_status(f"RMSE: {metrics['rmse']:.4f}")
        self.log_status(f"R²: {metrics['r2']:.4f}")
        self.log_status(f"MAPE: {metrics['mape']:.2f}%")
    
    def _evaluate_predictions(self, predictions, actuals):
        """评估预测结果"""
        if len(predictions) < 10:
            self.log_status("数据点不足，无法进行完整评估")
            return
        
        metrics = self.analytics_tools.calculate_comprehensive_metrics(
            np.array(actuals), np.array(predictions)
        )
        
        self.log_status("\n=== 最终预测评估 ===")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.log_status(f"{metric.upper()}: {value:.6f}")
        
        # 创建评估图表
        self.create_prediction_evaluation_plot(predictions, actuals)
    
    def create_prediction_evaluation_plot(self, predictions, actuals):
        """创建预测评估图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 预测vs实际
        axes[0, 0].plot(actuals, label='实际值', alpha=0.7, linewidth=1)
        axes[0, 0].plot(predictions, label='预测值', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('预测 vs 实际')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 误差分布
        errors = [a - p for a, p in zip(actuals, predictions)]
        axes[0, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_title('预测误差分布')
        axes[0, 1].set_xlabel('误差')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 残差图
        axes[1, 0].scatter(predictions, errors, alpha=0.5)
        axes[1, 0].axhline(y=0, color='red', linestyle='--')
        axes[1, 0].set_xlabel('预测值')
        axes[1, 0].set_ylabel('残差')
        axes[1, 0].set_title('残差图')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 累积误差
        cumulative_error = np.cumsum([abs(e) for e in errors])
        axes[1, 1].plot(cumulative_error)
        axes[1, 1].set_title('累积绝对误差')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('累积误差')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def start_prediction_simulation(self):
        """开始预测模拟"""
        self.log_status("启动预测模拟...")
        self.start_realtime_prediction()
    
    def stop_prediction(self):
        """停止预测"""
        self.log_status("预测已停止")
    
    def enable_rl_optimization(self):
        """启用RL优化"""
        if not self.rl_agent:
            self.rl_agent = AdvancedReinforcementLearning()
        
        self.log_status("强化学习优化已启用")
    
    def analyze_rl_policy(self):
        """分析RL策略"""
        if not self.rl_agent:
            self.log_status("请先初始化强化学习代理")
            return
        
        analysis = self.rl_agent.get_policy_analysis()
        
        self.log_status("\n=== 强化学习策略分析 ===")
        self.log_status(f"平均奖励: {analysis['average_reward']:.4f}")
        self.log_status(f"探索率: {analysis['exploration_rate']:.4f}")
        self.log_status(f"Q值范围: [{analysis['q_value_range'][0]:.4f}, {analysis['q_value_range'][1]:.4f}]")
        
        action_dist = analysis['action_distribution']
        total_actions = np.sum(action_dist)
        if total_actions > 0:
            self.log_status("动作分布:")
            for action, count in enumerate(action_dist):
                percentage = (count / total_actions) * 100
                meaning = self.rl_agent.get_action_meaning(action)
                self.log_status(f"  {meaning}: {percentage:.1f}%")
    
    def run_model_evaluation(self):
        """运行模型评估"""
        def _evaluate():
            try:
                self.log_status("开始模型评估...")
                
                # 生成测试数据
                X_test, y_test = self.data_generator.generate_synthetic_data(500)
                
                # 进行预测
                self.model.eval()
                predictions = []
                
                with torch.no_grad():
                    for i in range(len(X_test)):
                        x_tensor = torch.FloatTensor(X_test[i]).unsqueeze(0)
                        pred = self.model(x_tensor).item()
                        predictions.append(pred)
                
                # 计算指标
                metrics = self.analytics_tools.calculate_comprehensive_metrics(
                    y_test, np.array(predictions)
                )
                
                # 显示结果
                self.analysis_text.delete(1.0, tk.END)
                self.analysis_text.insert(tk.END, "=== 模型评估结果 ===\n\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        self.analysis_text.insert(tk.END, f"{metric.upper()}: {value:.6f}\n")
                
                # 残差分析
                residuals = y_test - predictions
                residual_analysis = self.analytics_tools.perform_residual_analysis(
                    y_test, np.array(predictions)
                )
                
                self.analysis_text.insert(tk.END, "\n=== 残差分析 ===\n")
                for stat, value in residual_analysis.items():
                    if isinstance(value, (int, float)):
                        self.analysis_text.insert(tk.END, f"{stat}: {value:.6f}\n")
                
                self.log_status("模型评估完成！")
                
            except Exception as e:
                self.log_status(f"模型评估失败: {e}")
        
        threading.Thread(target=_evaluate, daemon=True).start()
    
    def run_residual_analysis(self):
        """运行残差分析"""
        self.log_status("运行残差分析...")
        self.run_model_evaluation()  # 复用模型评估功能
    
    def generate_analysis_report(self):
        """生成分析报告"""
        def _generate_report():
            try:
                self.log_status("生成分析报告...")
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'system_info': self.get_system_info(),
                    'model_architecture': str(self.model) if self.model else None,
                    'training_history': self.trainer.train_history if self.trainer else None
                }
                
                # 保存报告
                filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                self.log_status(f"分析报告已保存: {filename}")
                
            except Exception as e:
                self.log_status(f"生成报告失败: {e}")
        
        threading.Thread(target=_generate_report, daemon=True).start()
    
    def update_system_status(self):
        """更新系统状态"""
        if self.memory_manager:
            stats = self.memory_manager.get_memory_stats()
            
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, "=== 系统状态 ===\n\n")
            self.status_text.insert(tk.END, f"CPU使用率: {stats['cpu_usage']:.1f}%\n")
            self.status_text.insert(tk.END, f"内存使用率: {stats['ram_usage']:.1f}%\n")
            self.status_text.insert(tk.END, f"已用内存: {stats['ram_used_gb']:.1f} GB\n")
            
            if stats['gpu_available']:
                self.status_text.insert(tk.END, "\n=== GPU状态 ===\n")
                for i in range(stats['gpu_count']):
                    self.status_text.insert(tk.END, 
                        f"GPU {i} ({stats[f'gpu_{i}_name']}):\n")
                    self.status_text.insert(tk.END, 
                        f"  使用率: {stats[f'gpu_{i}_usage']:.1f}%\n")
                    self.status_text.insert(tk.END, 
                        f"  显存使用: {stats[f'gpu_{i}_memory_usage']:.1f}%\n")
                    self.status_text.insert(tk.END, 
                        f"  温度: {stats[f'gpu_{i}_temperature']}°C\n")
            
            self.status_text.insert(tk.END, f"\n更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def optimize_system_memory(self):
        """优化系统内存"""
        if self.memory_manager:
            self.memory_manager.clear_gpu_cache()
            self.log_status("系统内存已优化")
    
    def show_system_info(self):
        """显示系统信息"""
        info = self.get_system_info()
        
        info_text = "=== 系统信息 ===\n\n"
        for key, value in info.items():
            info_text += f"{key}: {value}\n"
        
        messagebox.showinfo("系统信息", info_text)
    
    def get_system_info(self):
        """获取系统信息"""
        return {
            'Python版本': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'PyTorch版本': torch.__version__,
            'CUDA可用': torch.cuda.is_available(),
            'GPU数量': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            '系统平台': sys.platform,
            '处理器': psutil.cpu_count(),
            '总内存': f"{psutil.virtual_memory().total / 1024**3:.1f} GB",
            '当前时间': datetime.now().isoformat()
        }
    
    def load_external_data(self):
        """加载外部数据"""
        filename = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        
        if filename:
            self.log_status(f"加载外部数据: {filename}")
            # 这里可以添加具体的数据加载逻辑
    
    def clear_training_log(self):
        """清空训练日志"""
        self.train_text.delete(1.0, tk.END)
    
    def export_training_results(self):
        """导出训练结果"""
        filename = filedialog.asksaveasfilename(
            title="保存训练结果",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if filename and self.trainer:
            try:
                results = {
                    'training_history': self.trainer.train_history,
                    'model_info': str(self.model),
                    'export_time': datetime.now().isoformat()
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                self.log_status(f"训练结果已导出: {filename}")
            except Exception as e:
                self.log_status(f"导出失败: {e}")
    
    def log_status(self, message):
        """记录状态信息"""
        def _log():
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            
            # 更新训练日志
            self.train_text.insert(tk.END, log_message)
            self.train_text.see(tk.END)
            
            # 更新预测日志（如果包含预测相关消息）
            if any(keyword in message.lower() for keyword in ['预测', '实际', '误差', 'mae', 'rmse']):
                                                              
          # 更新预测日志（如果包含预测相关消息）
            if any(keyword in message.lower() for keyword in ['预测', '实际', '误差', 'mae', 'rmse']):
                self.pred_text.insert(tk.END, log_message)
                self.pred_text.see(tk.END)
            
            # 更新状态栏
            self.status_label.config(text=message[:50] + "..." if len(message) > 50 else message)
            
            # 确保GUI更新
            self.root.update_idletasks()
        
        # 确保在GUI线程中执行
        self.root.after(0, _log)
    
    def run_batch_prediction(self):
        """运行批量预测"""
        def _batch_predict():
            try:
                self.log_status("开始批量预测...")
                
                # 生成批量测试数据
                X_test, y_test = self.data_generator.generate_synthetic_data(1000)
                
                predictions = []
                confidence_intervals = []
                
                # 批量预测
                self.model.eval()
                with torch.no_grad():
                    for i in range(len(X_test)):
                        x_tensor = torch.FloatTensor(X_test[i]).unsqueeze(0)
                        pred = self.model(x_tensor).item()
                        predictions.append(pred)
                        
                        # 计算置信区间（简化版本）
                        confidence = (pred * 0.9, pred * 1.1)  # ±10%
                        confidence_intervals.append(confidence)
                        
                        if i % 100 == 0:
                            self.log_status(f"已处理 {i}/{len(X_test)} 个样本")
                
                # 计算评估指标
                metrics = self.analytics_tools.calculate_comprehensive_metrics(
                    y_test, np.array(predictions)
                )
                
                # 显示结果
                self.log_status("\n=== 批量预测结果 ===")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        self.log_status(f"{metric.upper()}: {value:.6f}")
                
                # 保存预测结果
                self._save_prediction_results(y_test, predictions, confidence_intervals, metrics)
                
                self.log_status("批量预测完成！")
                
            except Exception as e:
                self.log_status(f"批量预测失败: {e}")
        
        threading.Thread(target=_batch_predict, daemon=True).start()
    
    def _save_prediction_results(self, actuals, predictions, confidence_intervals, metrics):
        """保存预测结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_prediction_results_{timestamp}.csv"
            
            results_df = pd.DataFrame({
                'actual': actuals,
                'predicted': predictions,
                'confidence_lower': [ci[0] for ci in confidence_intervals],
                'confidence_upper': [ci[1] for ci in confidence_intervals],
                'error': [a - p for a, p in zip(actuals, predictions)],
                'absolute_error': [abs(a - p) for a, p in zip(actuals, predictions)]
            })
            
            # 添加评估指标
            metrics_df = pd.DataFrame([metrics])
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=== 预测结果 ===\n")
                results_df.to_csv(f, index=False)
                f.write("\n=== 评估指标 ===\n")
                metrics_df.to_csv(f, index=False)
            
            self.log_status(f"预测结果已保存到: {filename}")
            
        except Exception as e:
            self.log_status(f"保存结果失败: {e}")
    
    def clear_all_logs(self):
        """清空所有日志"""
        self.train_text.delete(1.0, tk.END)
        self.pred_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.status_text.delete(1.0, tk.END)
        self.log_status("所有日志已清空")
    
    def run_system_diagnostics(self):
        """运行系统诊断"""
        def _diagnose():
            try:
                self.log_status("运行系统诊断...")
                
                diagnostics = {
                    'system': self._check_system_health(),
                    'gpu': self._check_gpu_health(),
                    'memory': self._check_memory_health(),
                    'model': self._check_model_health()
                }
                
                # 显示诊断结果
                self.status_text.delete(1.0, tk.END)
                self.status_text.insert(tk.END, "=== 系统诊断结果 ===\n\n")
                
                for category, results in diagnostics.items():
                    self.status_text.insert(tk.END, f"{category.upper()}诊断:\n")
                    for test, (status, message) in results.items():
                        status_symbol = "✓" if status else "✗"
                        self.status_text.insert(tk.END, f"  {status_symbol} {test}: {message}\n")
                    self.status_text.insert(tk.END, "\n")
                
                self.log_status("系统诊断完成")
                
            except Exception as e:
                self.log_status(f"诊断失败: {e}")
        
        threading.Thread(target=_diagnose, daemon=True).start()
    
    def _check_system_health(self):
        """检查系统健康状态"""
        results = {}
        
        # 检查Python版本
        python_ok = sys.version_info >= (3, 7)
        results['Python版本'] = (python_ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # 检查关键库
        libraries = ['torch', 'numpy', 'pandas', 'sklearn']
        for lib in libraries:
            try:
                __import__(lib)
                results[f'{lib}库'] = (True, "已安装")
            except ImportError:
                results[f'{lib}库'] = (False, "未安装")
        
        # 检查文件系统
        disk_usage = psutil.disk_usage('/')
        disk_ok = disk_usage.percent < 90
        results['磁盘空间'] = (disk_ok, f"{disk_usage.percent:.1f}% 已使用")
        
        return results
    
    def _check_gpu_health(self):
        """检查GPU健康状态"""
        results = {}
        
        if torch.cuda.is_available():
            results['CUDA可用'] = (True, "GPU支持已启用")
            
            # 检查每个GPU
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_cached = torch.cuda.memory_reserved(i) / 1024**3
                    
                    gpu_ok = memory_allocated / props.total_memory * 1024**3 < 0.9
                    results[f'GPU{i}状态'] = (
                        gpu_ok, 
                        f"{props.name}, 内存使用: {memory_allocated:.1f}/{props.total_memory/1024**3:.1f}GB"
                    )
                except Exception as e:
                    results[f'GPU{i}状态'] = (False, f"检测失败: {e}")
        else:
            results['CUDA可用'] = (False, "无GPU支持")
        
        return results
    
    def _check_memory_health(self):
        """检查内存健康状态"""
        results = {}
        
        # RAM使用情况
        ram = psutil.virtual_memory()
        ram_ok = ram.percent < 85
        results['RAM使用'] = (ram_ok, f"{ram.percent:.1f}% 已使用")
        
        # SWAP使用情况
        swap = psutil.swap_memory()
        swap_ok = swap.percent < 50
        results['SWAP使用'] = (swap_ok, f"{swap.percent:.1f}% 已使用" if swap.total > 0 else "未启用")
        
        return results
    
    def _check_model_health(self):
        """检查模型健康状态"""
        results = {}
        
        if self.model is not None:
            # 检查模型参数
            param_count = sum(p.numel() for p in self.model.parameters())
            results['模型参数'] = (True, f"{param_count:,} 个参数")
            
            # 检查模型状态
            if hasattr(self.model, 'training'):
                results['模型模式'] = (True, "训练模式" if self.model.training else "评估模式")
            else:
                results['模型模式'] = (False, "未知模式")
        else:
            results['模型状态'] = (False, "模型未初始化")
        
        return results
    
    def create_performance_report(self):
        """创建性能报告"""
        def _create_report():
            try:
                self.log_status("创建性能报告...")
                
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'system_info': self.get_system_info(),
                    'performance_metrics': self._collect_performance_metrics(),
                    'training_history': self.trainer.train_history if self.trainer else {},
                    'model_architecture': str(self.model) if self.model else None
                }
                
                # 保存报告
                filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                
                self.log_status(f"性能报告已保存: {filename}")
                
                # 生成可视化报告
                self._generate_visual_report(report, filename.replace('.json', '.png'))
                
            except Exception as e:
                self.log_status(f"创建报告失败: {e}")
        
        threading.Thread(target=_create_report, daemon=True).start()
    
    def _collect_performance_metrics(self):
        """收集性能指标"""
        metrics = {}
        
        if self.trainer and self.trainer.train_history:
            history = self.trainer.train_history
            metrics.update({
                'final_train_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else float('inf'),
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else float('inf'),
                'total_training_time': sum(history['epoch_time']) if history['epoch_time'] else 0,
                'average_epoch_time': np.mean(history['epoch_time']) if history['epoch_time'] else 0
            })
        
        # 添加系统性能指标
        if self.memory_manager:
            stats = self.memory_manager.get_memory_stats()
            metrics.update({
                'max_cpu_usage': max(self.memory_manager.memory_history['cpu_usage'], default=0),
                'max_ram_usage': max(self.memory_manager.memory_history['ram_usage'], default=0),
                'avg_gpu_usage': np.mean(self.memory_manager.memory_history.get('gpu_0_usage', [0]))
            })
        
        return metrics
    
    def _generate_visual_report(self, report, filename):
        """生成可视化报告"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 训练历史
            if report['training_history']:
                history = report['training_history']
                axes[0, 0].plot(history.get('train_loss', []), label='训练损失')
                axes[0, 0].plot(history.get('val_loss', []), label='验证损失')
                axes[0, 0].set_title('训练历史')
                axes[0, 0].legend()
                axes[0, 0].set_yscale('log')
                axes[0, 0].grid(True, alpha=0.3)
            
            # 资源使用
            if self.memory_manager:
                axes[0, 1].plot(self.memory_manager.memory_history.get('cpu_usage', []), label='CPU使用率')
                axes[0, 1].plot(self.memory_manager.memory_history.get('ram_usage', []), label='内存使用率')
                axes[0, 1].set_title('资源使用情况')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # GPU使用情况
            if torch.cuda.is_available():
                gpu_usage = self.memory_manager.memory_history.get('gpu_0_usage', [])
                if gpu_usage:
                    axes[1, 0].plot(gpu_usage, label='GPU使用率', color='red')
                    axes[1, 0].set_title('GPU使用情况')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # 关键指标表格
            metrics = report['performance_metrics']
            table_data = [
                ['最终训练损失', f"{metrics.get('final_train_loss', 'N/A'):.6f}"],
                ['最终验证损失', f"{metrics.get('final_val_loss', 'N/A'):.6f}"],
                ['最佳验证损失', f"{metrics.get('best_val_loss', 'N/A'):.6f}"],
                ['总训练时间', f"{metrics.get('total_training_time', 'N/A'):.1f}s"],
                ['平均epoch时间', f"{metrics.get('average_epoch_time', 'N/A'):.2f}s"]
            ]
            
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=table_data, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            axes[1, 1].set_title('性能指标')
            
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log_status(f"可视化报告已保存: {filename}")
            
        except Exception as e:
            self.log_status(f"生成可视化报告失败: {e}")
    
    def run_comprehensive_test(self):
        """运行全面测试"""
        def _comprehensive_test():
            try:
                self.log_status("开始全面系统测试...")
                
                # 1. 系统诊断
                self.run_system_diagnostics()
                time.sleep(2)
                
                # 2. 数据生成测试
                self.log_status("测试数据生成...")
                X, y = self.data_generator.generate_synthetic_data(100)
                self.log_status(f"数据生成测试通过: X.shape={X.shape}, y.shape={y.shape}")
                time.sleep(1)
                
                # 3. 模型推理测试
                self.log_status("测试模型推理...")
                self.model.eval()
                with torch.no_grad():
                    test_input = torch.FloatTensor(X[0]).unsqueeze(0)
                    prediction = self.model(test_input).item()
                    self.log_status(f"模型推理测试通过: 预测值={prediction:.4f}")
                time.sleep(1)
                
                # 4. 训练功能测试
                self.log_status("测试训练功能...")
                test_dataset = list(zip(X[:10], y[:10]))
                test_loader = DataLoader(test_dataset, batch_size=2)
                
                test_optimizer = optim.Adam(self.model.parameters(), lr=0.001)
                test_criterion = nn.MSELoss()
                
                self.model.train()
                for data, target in test_loader:
                    test_optimizer.zero_grad()
                    output = self.model(data)
                    loss = test_criterion(output.squeeze(), target)
                    loss.backward()
                    test_optimizer.step()
                    break
                
                self.log_status(f"训练功能测试通过: 损失={loss.item():.6f}")
                time.sleep(1)
                
                # 5. 预测功能测试
                self.log_status("测试预测功能...")
                self.predictor.update_data(0.5)
                for i in range(10):
                    self.predictor.update_data(0.5 + i * 0.1)
                
                prediction, confidence = self.predictor.predict_next()
                self.log_status(f"预测功能测试通过: 预测={prediction:.4f}, 置信区间={confidence}")
                
                self.log_status("🎉 全面系统测试完成！所有功能正常")
                
            except Exception as e:
                self.log_status(f"❌ 系统测试失败: {e}")
        
        threading.Thread(target=_comprehensive_test, daemon=True).start()
    
    def emergency_shutdown(self):
        """紧急关闭系统"""
        result = messagebox.askyesno("紧急关闭", "确定要紧急关闭系统吗？所有未保存的数据将会丢失！")
        
        if result:
            self.log_status("⚠️ 紧急关闭系统...")
            
            # 停止所有线程
            self.memory_manager.stop_monitoring = True
            
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 保存最后状态
            try:
                emergency_save = {
                    'timestamp': datetime.now().isoformat(),
                    'system_state': 'emergency_shutdown',
                    'last_operations': '系统紧急关闭'
                }
                
                with open('emergency_shutdown.json', 'w', encoding='utf-8') as f:
                    json.dump(emergency_save, f, indent=2)
                
                self.log_status("紧急状态已保存")
            except:
                pass
            
            # 延迟关闭
            self.root.after(1000, self.root.destroy)
    
    def on_closing(self):
        """关闭窗口时的处理"""
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            self.memory_manager.stop_monitoring = True
            self.root.destroy()
    
    def run(self):
        """运行GUI"""
        # 设置关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 添加菜单
        self._create_menu()
        
        # 启动系统
        self.log_status("系统启动完成，等待用户操作...")
        self.root.mainloop()
    
    def _create_menu(self):
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="新建", command=self.initialize_complete_system)
        file_menu.add_command(label="打开", command=self.load_external_data)
        file_menu.add_command(label="保存", command=self.export_training_results)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.on_closing)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="系统诊断", command=self.run_system_diagnostics)
        tools_menu.add_command(label="性能报告", command=self.create_performance_report)
        tools_menu.add_command(label="全面测试", command=self.run_comprehensive_test)
        tools_menu.add_separator()
        tools_menu.add_command(label="清空日志", command=self.clear_all_logs)
        menubar.add_cascade(label="工具", menu=tools_menu)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="系统信息", command=self.show_system_info)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def show_about(self):
        """显示关于信息"""
        about_text = """GPU加速深度学习预测系统 - 完整版

功能特性:
• 动态多变量时间序列数据生成
• 带注意力机制的LSTM神经网络
• 多GPU分布式训练支持
• 实时预测与不确定性估计
• 强化学习优化预测策略
• 高级分析与可视化工具
• 自动化超参数优化
• 智能内存管理系统

版本: 2.0.0
作者: AI系统开发团队
日期: 2025-10-29"""
        
        messagebox.showinfo("关于", about_text)

# 主函数
def main():
    """主函数"""
    print("=" * 60)
    print("GPU加速深度学习预测系统 - 完整版")
    print("=" * 60)
    
    # 检查系统环境
    print("检查系统环境...")
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("警告: 未检测到GPU，将使用CPU运行")
    
    print("\n选择运行模式:")
    print("1. 命令行模式 (快速测试)")
    print("2. GUI完整模式 (推荐)")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "1":
            run_command_line_demo()
        else:
            # 启动GUI
            app = PredictionSystemGUI()
            app.run()
            
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序错误: {e}")

def run_command_line_demo():
    """运行命令行演示"""
    print("\n启动命令行演示模式...")
    
    try:
        # 1. 初始化组件
        print("1. 初始化系统组件...")
        data_generator = EnhancedDynamicDataGenerator()
        X, y = data_generator.generate_synthetic_data(1000)
        print(f"   数据生成完成: X.shape={X.shape}, y.shape={y.shape}")
        
        # 2. 创建模型
        print("2. 创建深度学习模型...")
        model = AdvancedLSTMWithAttention(input_size=X.shape[2])
        print(f"   模型创建完成: {sum(p.numel() for p in model.parameters()):,} 参数")
        
        # 3. 训练模型
        print("3. 训练模型...")
        trainer = DistributedMultiGPUTrainer(model)
        dataset = list(zip(X, y))
        train_loader = DataLoader(dataset[:800], batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset[800:], batch_size=32)
        
        history = trainer.train_model(train_loader, val_loader, epochs=50, lr=0.001)
        print(f"   训练完成! 最终验证损失: {min(history['val_loss']):.6f}")
        
        # 4. 实时预测演示
        print("4. 演示实时预测...")
        predictor = EnhancedRealTimePredictor(model)
        
        # 生成测试数据流
        test_data = np.random.randn(100)
        predictions = []
        
        for i, point in enumerate(test_data):
            predictor.update_data(point)
            pred, confidence = predictor.predict_next()
            
            if pred is not None:
                predictions.append(pred[0])
                if i % 20 == 0:
                    print(f"   预测 {i}: {pred[0]:.4f} ± {(confidence[0][1]-confidence[0][0])/2:.4f}")
        
        # 5. 评估结果
        print("5. 评估预测结果...")
        actuals = test_data[50:]  # 前50个点用于初始化
        if len(predictions) > len(actuals):
            predictions = predictions[:len(actuals)]
        
        metrics = {
            'MAE': mean_absolute_error(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'R2': r2_score(actuals, predictions)
        }
        
        print(f"   预测性能: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R²={metrics['R2']:.4f}")
        
        print("\n🎉 命令行演示完成!")
        print("建议使用GUI模式获得完整功能体验")
        
    except Exception as e:
        print(f"演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
