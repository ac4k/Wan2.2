# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Forward Hook Recorder - Record input/output shape and dtype for each module
"""

import torch
import torch.nn as nn
import csv
import os
from typing import Any, Dict, List, Tuple, Union
from tabulate import tabulate

class TensorStat:
    def __init__(self, tensor: Any):
        self.shape = None
        self.dtype = None
        self.device = None
        self.size_bytes = 0
        self.is_tensor = False
        self.children = [] # List of TensorStat
        self.type_name = type(tensor).__name__
        
        if isinstance(tensor, torch.Tensor):
            self.is_tensor = True
            self.device = str(tensor.device)
            self.dtype = str(tensor.dtype).replace("torch.", "")
            
            # Handle FSDP
            if hasattr(tensor, "_unpadded_unsharded_size"):
                self.shape = tuple(tensor._unpadded_unsharded_size)
            elif hasattr(tensor, "_unpadded_unsharded_shape"):
                self.shape = tuple(tensor._unpadded_unsharded_shape)
            elif hasattr(tensor, "_original_shape"):
                self.shape = tuple(tensor._original_shape)
            else:
                self.shape = tuple(tensor.shape)
            
            numel = 1
            for d in self.shape: numel *= d
            self.size_bytes = numel * tensor.element_size()
            
        elif isinstance(tensor, (list, tuple)):
            self.children = [TensorStat(x) for x in tensor]
            self.size_bytes = sum(c.size_bytes for c in self.children)
        elif isinstance(tensor, dict):
            self.children = [TensorStat(v) for v in tensor.values()]
            self.size_bytes = sum(c.size_bytes for c in self.children)

    def __str__(self):
        if self.is_tensor:
            return f"shape={self.shape}, dtype={self.dtype}, device={self.device}"
        elif self.children:
            return "[" + ", ".join(str(c) for c in self.children) + "]"
        else:
            return self.type_name

class FSDPRuntimeDumper:
    def __init__(self, dump_path: str = "hook_logs/trace.csv"):
        self.dump_path = dump_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        
        self.file = open(dump_path, 'w', newline='')
        self.writer = csv.writer(self.file, delimiter=';')
        # Columns: index; model; module name; output tensors; module type; input tensors; parameters; input size; weight size; output size; memory used; gpu time
        self.writer.writerow(['index', 'model', 'module name', 'output tensors', 'module type', 'input tensors', 'parameters', 'input size', 'weight size', 'output size', 'memory used', 'gpu time'])
        self.handles = []
        # Cache stores a list of (params_info_str, weight_size_bytes) for each module call
        self.param_cache: Dict[str, List[Tuple[str, int]]] = {}
        self.time_cache: Dict[str, List[torch.cuda.Event]] = {}
        self.execution_index = 0
        
        # Initialize parameter stats file
        self.param_stats_path = os.path.join(os.path.dirname(dump_path), "model_parameters.md")
        if os.path.exists(self.param_stats_path):
            os.remove(self.param_stats_path)

    def log_parameter_stats(self, model: nn.Module, model_name: str):
        """Logs parameter statistics for the given model."""
        rows = []
        total_size_bytes = 0
        
        for name, param in model.named_parameters():
            shape_tuple = tuple(param.shape)
            # Handle FSDP/sharded parameters to get original shape
            if hasattr(param, "_unpadded_unsharded_size"):
                shape_tuple = tuple(param._unpadded_unsharded_size)
            elif hasattr(param, "_unpadded_unsharded_shape"):
                shape_tuple = tuple(param._unpadded_unsharded_shape)
            elif hasattr(param, "_original_shape"):
                shape_tuple = tuple(param._original_shape)
                
            shape = str(shape_tuple)
            dtype = str(param.dtype).replace("torch.", "")
            
            # Calculate size based on the reported shape (unsharded if applicable)
            numel = 1
            for dim in shape_tuple:
                numel *= dim
                
            element_size = param.element_size()
            size_bytes = numel * element_size
            
            rows.append([name, shape, dtype, f"{size_bytes:,}"])
            total_size_bytes += size_bytes
            
        headers = ["Name", "Shape", "Dtype", "Size (Bytes)"]
        
        # Use tabulate for formatting
        # colalign to ensure the size column (string with commas) is right-aligned
        table = tabulate(rows, headers=headers, tablefmt="github", colalign=("left", "left", "left", "right"))
        
        with open(self.param_stats_path, 'a') as f:
            f.write(f"### Model: {model_name}\n\n")
            f.write(table)
            f.write(f"\n\n**Total Size:** {total_size_bytes:,} bytes ({total_size_bytes / (1024**2):.2f} MB)\n")
            f.write("\n---\n\n")

    def _get_parameters_info(self, module: nn.Module) -> Tuple[str, int]:
        """Extracts parameter information and total size."""
        params_info_list = []
        total_size = 0
        
        # Helper to process a parameter
        def process_param(name, param):
            stat = TensorStat(param)
            return f"{name}: {str(stat)}", stat.size_bytes

        # Check named parameters
        seen_params = set()
        for name, param in module.named_parameters(recurse=False):
            s, size = process_param(name, param)
            params_info_list.append(s)
            total_size += size
            seen_params.add(name)
        
        # Fallback for weights not in named_parameters
        if 'weight' not in seen_params and hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
             s, size = process_param('weight', module.weight)
             params_info_list.append(s)
             total_size += size
        
        if 'bias' not in seen_params and hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
             s, size = process_param('bias', module.bias)
             params_info_list.append(s)
             total_size += size

        return "; ".join(params_info_list), total_size

    def _pre_hook(self, module: nn.Module, inputs: Any, name: str):
        """Pre-hook to capture parameter info when they are gathered."""
        try:
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                if name not in self.time_cache:
                    self.time_cache[name] = []
                self.time_cache[name].append(start_event)

            info, size = self._get_parameters_info(module)
            if name not in self.param_cache:
                self.param_cache[name] = []
            self.param_cache[name].append((info, size))
        except Exception:
            pass

    def _post_hook(self, module: nn.Module, inputs: Any, outputs: Any, name: str, model_name: str):
        """Post-hook to capture inputs, outputs, memory and write to CSV."""
        try:
            self.execution_index += 1
            gpu_time = 0.0
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated()
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record()
                if name in self.time_cache and self.time_cache[name]:
                    start_event = self.time_cache[name].pop()
                    end_event.synchronize()
                    gpu_time = start_event.elapsed_time(end_event)
            else:
                memory_used = 0

            input_stat = TensorStat(inputs)
            output_stat = TensorStat(outputs)
            module_type = type(module).__name__
            
            # Retrieve param info captured in pre-hook
            params_info = ""
            weight_size = 0
            
            if name in self.param_cache and self.param_cache[name]:
                params_info, weight_size = self.param_cache[name].pop()
            else:
                # Fallback if pre-hook didn't run or cache issue
                params_info, weight_size = self._get_parameters_info(module)

            self.writer.writerow([
                self.execution_index,
                model_name,
                name, 
                str(output_stat), 
                module_type, 
                str(input_stat), 
                params_info, 
                input_stat.size_bytes, 
                weight_size, 
                output_stat.size_bytes, 
                memory_used,
                gpu_time
            ])
            self.file.flush()
        except Exception as e:
            print(f"FSDPRuntimeDumper Error in post_hook for {name}: {e}")

    def register_hooks(self, model: nn.Module, prefix: str = ""):
        """Registers forward hooks on all modules."""
        
        model_name = prefix if prefix else "root"
        
        # Log parameter stats for this model
        self.log_parameter_stats(model, model_name)

        for name, module in model.named_modules():
            # Skip modules that have children (containers) to only log leaf modules (actual computation)
            if len(list(module.children())) > 0:
                continue

            full_name = f"{prefix}.{name}" if prefix else name
            if not full_name:
                full_name = "root"
                
            # Register pre-hook for params
            h1 = module.register_forward_pre_hook(
                lambda m, i, n=full_name: self._pre_hook(m, i, n)
            )
            self.handles.append(h1)
            
            # Register post-hook for inputs/outputs/memory
            h2 = module.register_forward_hook(
                lambda m, i, o, n=full_name, mn=model_name: self._post_hook(m, i, o, n, mn)
            )
            self.handles.append(h2)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self.file:
            self.file.close()
            self.file = None
