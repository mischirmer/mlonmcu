#
# Copyright (c) 2022 TUM Department of Electrical and Computer Engineering.
#
# This file is part of MLonMCU.
# See https://github.com/tum-ei-eda/mlonmcu.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import glob
import os
from collections import OrderedDict

import pandas as pd


def find_newest_report():
    home = os.getenv("MLONMCU_HOME")
    list_of_files = glob.glob(os.path.join(home, "results", "*"))  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


def tabularize_latest_report():
    report_file = find_newest_report()
    df = pd.read_csv(report_file, sep=",")
    return df


def get_all_results():
    home = os.getenv("MLONMCU_HOME")
    list_of_files = glob.glob(os.path.join(home, "results", "*.csv"))
    dfs = [pd.read_csv(file, sep=",") for file in list_of_files]
    return pd.concat(dfs, ignore_index=True)


def get_comparison_dfs(csv_specs=None, as_dict=False, results_dir=None):
    """Load comparison CSVs.

    Parameters
    ----------
    csv_specs:
        - dict: {"Label": "file.csv", ...}
        - list/tuple: ["file1.csv", "file2.csv", ...] (labels inferred from names)
        - None: uses legacy defaults (baseline/abft/abyzft/freivald_*).
    as_dict:
        If True, always return OrderedDict[label] = DataFrame.
        If False and csv_specs is None, keep legacy tuple return order.
    results_dir:
        Optional directory override. Defaults to "$MLONMCU_HOME/results".
    """
    home = os.getenv("MLONMCU_HOME")
    if results_dir is None:
        results_dir = os.path.join(home, "results")

    if csv_specs is None:
        specs = OrderedDict(
            [
                ("Baseline", "baseline.csv"),
                ("ABFT", "abft.csv"),
                ("AByzFT", "abyzft.csv"),
                ("Freivald (Standard)", "freivald_standard.csv"),
                ("Freivald (Binary)", "freivald_binary.csv"),
            ]
        )
    elif isinstance(csv_specs, dict):
        specs = OrderedDict(csv_specs.items())
    else:
        specs = OrderedDict()
        for name in csv_specs:
            label = os.path.splitext(os.path.basename(name))[0]
            specs[label] = name

    result = OrderedDict()
    for label, filename in specs.items():
        path = filename if os.path.isabs(filename) else os.path.join(results_dir, filename)
        result[label] = pd.read_csv(path, sep=",")

    if not as_dict and csv_specs is None:
        # Legacy return shape for existing notebooks.
        return tuple(result.values())
    return result


def plot_comparison(*dfs, names=None, results=None, baseline_label=None):
    import matplotlib.pyplot as plt

    if results is None:
        if len(dfs) == 1 and isinstance(dfs[0], dict):
            results = OrderedDict(dfs[0].items())
        else:
            if names is None:
                names = ["Baseline", "ABFT", "AByzFT", "Freivald (Standard)", "Freivald (Binary)"][: len(dfs)]
            if len(names) != len(dfs):
                raise ValueError("names length must match number of DataFrames")
            results = OrderedDict(zip(names, dfs))
    else:
        results = OrderedDict(results.items())

    if len(results) == 0:
        raise ValueError("No results to plot")

    if baseline_label is None:
        baseline_label = "Baseline" if "Baseline" in results else next(iter(results.keys()))
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
    
    # Use the correct column names from the dataframe
    cycles_col = 'Total Cycles'
    memory_col = 'Total RAM'
    rom_readonly_col = 'ROM read-only'
    rom_code_col = 'ROM code'
    
    cycles = {name: results[name][cycles_col].sum() for name in results.keys()}
    memory = {name: results[name][memory_col].sum() for name in results.keys()}
    rom_readonly = {name: results[name][rom_readonly_col].sum() for name in results.keys()}
    rom_code = {name: results[name][rom_code_col].sum() for name in results.keys()}
    
    # Calculate relative overhead compared to baseline
    baseline_cycles = cycles[baseline_label]
    baseline_memory = memory[baseline_label]
    baseline_rom_readonly = rom_readonly[baseline_label]
    baseline_rom_code = rom_code[baseline_label]
    
    cycles_overhead = {name: ((cycles[name] - baseline_cycles) / baseline_cycles * 100) 
                       for name in results.keys()}
    memory_overhead = {name: ((memory[name] - baseline_memory) / baseline_memory * 100) 
                       for name in results.keys()}
    rom_readonly_overhead = {name: ((rom_readonly[name] - baseline_rom_readonly) / baseline_rom_readonly * 100) 
                             for name in results.keys()}
    rom_code_overhead = {name: ((rom_code[name] - baseline_rom_code) / baseline_rom_code * 100) 
                         for name in results.keys()}
    
    # Plot cycles with absolute numbers
    colors = ["green" if name == baseline_label else "orange" for name in results.keys()]
    bars1 = ax1.bar(cycles.keys(), cycles.values(), color=colors)
    ax1.set_ylabel('Total Cycles')
    ax1.set_title('Cycles Comparison')
    
    # Add overhead percentage labels above bars
    for i, (name, value) in enumerate(cycles.items()):
        overhead = cycles_overhead[name]
        label = f'{overhead:.1f}%' if overhead != 0 else 'baseline'
        ax1.text(i, value, label, ha='center', va='bottom', fontweight='bold')
    
    # Plot memory with absolute numbers
    bars2 = ax2.bar(memory.keys(), memory.values(), color=colors)
    ax2.set_ylabel('Total RAM (bytes)')
    ax2.set_title('Dynamic Memory Comparison')
    
    # Add overhead percentage labels above bars
    for i, (name, value) in enumerate(memory.items()):
        overhead = memory_overhead[name]
        label = f'{overhead:.1f}%' if overhead != 0 else 'baseline'
        ax2.text(i, value, label, ha='center', va='bottom', fontweight='bold')
    
    # Plot ROM read-only with absolute numbers
    bars3 = ax3.bar(rom_readonly.keys(), rom_readonly.values(), color=colors)
    ax3.set_ylabel('ROM read-only (bytes)')
    ax3.set_title('ROM Read-Only Comparison')
    
    # Add overhead percentage labels above bars
    for i, (name, value) in enumerate(rom_readonly.items()):
        overhead = rom_readonly_overhead[name]
        label = f'{overhead:.1f}%' if overhead != 0 else 'baseline'
        ax3.text(i, value, label, ha='center', va='bottom', fontweight='bold')
    
    # Plot ROM code with absolute numbers
    bars4 = ax4.bar(rom_code.keys(), rom_code.values(), color=colors)
    ax4.set_ylabel('ROM code (bytes)')
    ax4.set_title('ROM Code Comparison')
    
    # Add overhead percentage labels above bars
    for i, (name, value) in enumerate(rom_code.items()):
        overhead = rom_code_overhead[name]
        label = f'{overhead:.1f}%' if name != baseline_label else 'baseline'
        ax4.text(i, value, label, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def plot_comparsion(*args, **kwargs):
    """Backward-compatible typo alias."""
    return plot_comparison(*args, **kwargs)
