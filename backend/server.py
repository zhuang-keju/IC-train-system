#!/usr/bin/env python3
"""
硅基未来AI仿真系统 - 后端服务
完全封装Qucs-S/ngspice，对外提供统一的仿真API
"""
import os
import sys
import json
import subprocess
import tempfile
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import re
from auto_layout import layout_spice_graph


app = Flask(__name__)
CORS(app)  # 允许跨域请求

# ==================== 系统配置 ====================
CONFIG = {
    "system_name": "硅基未来AI仿真系统",
    "version": "2.0",
    "simulator_path": "C:\\硅基未来\\ngspice-45.2_64\\Spice64\\bin\\ngspice.exe",
    "graphviz_path": "C:\\硅基未来\\windows_10_cmake_Release_Graphviz-14.1.1-win64\\Graphviz-14.1.1-win64\\bin",
    "max_sim_time": 30,  # 最大仿真时间(秒)
    "result_dir": "./simulation_results"
}

if CONFIG["graphviz_path"] not in os.environ["PATH"]:
    os.environ["PATH"] = CONFIG["graphviz_path"] + os.pathsep + os.environ["PATH"]

# ==================== CMOS电路库定义 ====================
CIRCUIT_LIBRARY = {
    "cmos_inverter": {
        "name": "CMOS反相器",
        "description": "基础数字电路单元，用于信号反相与缓冲",
        "category": "数字电路",
        "parameters": {
            "W_nmos": {"default": "1u", "min": "0.18u", "max": "10u", "unit": "μm", "desc": "NMOS宽度"},
            "L_nmos": {"default": "0.18u", "min": "0.18u", "max": "1u", "unit": "μm", "desc": "NMOS长度"},
            "W_pmos": {"default": "2u", "min": "0.18u", "max": "20u", "unit": "μm", "desc": "PMOS宽度"},
            "L_pmos": {"default": "0.18u", "min": "0.18u", "max": "1u", "unit": "μm", "desc": "PMOS长度"},
            "Vdd": {"default": "1.8", "min": "0.8", "max": "3.3", "unit": "V", "desc": "电源电压"},
            "Cload": {"default": "10f", "min": "1f", "max": "100p", "unit": "F", "desc": "负载电容"}
        },
        "analysis": ["dc", "tran", "ac"]
    },
    "two_stage_opamp": {
        "name": "两级运算放大器",
        "description": "带米勒补偿的CMOS运算放大器，用于模拟信号处理",
        "category": "模拟电路",
        "parameters": {
            "W1": {"default": "10u", "min": "1u", "max": "100u", "unit": "μm", "desc": "输入对管宽度"},
            "L1": {"default": "0.18u", "min": "0.18u", "max": "1u", "unit": "μm", "desc": "输入对管长度"},
            "W5": {"default": "20u", "min": "5u", "max": "200u", "unit": "μm", "desc": "尾电流源宽度"},
            "L5": {"default": "0.5u", "min": "0.18u", "max": "2u", "unit": "μm", "desc": "尾电流源长度"},
            "Cc": {"default": "2p", "min": "0.1p", "max": "20p", "unit": "F", "desc": "补偿电容"},
            "Rc": {"default": "2k", "min": "100", "max": "20k", "unit": "Ω", "desc": "补偿电阻"},
            "CL": {"default": "5p", "min": "0.1p", "max": "50p", "unit": "F", "desc": "负载电容"}
        },
        "analysis": ["op", "ac", "tran", "noise"]
    },
    "ring_oscillator": {
        "name": "环形振荡器",
        "description": "由奇数个反相器组成的振荡电路，用于时钟生成",
        "category": "数字电路",
        "parameters": {
            "stages": {"default": "11", "min": "3", "max": "101", "unit": "", "desc": "反相器级数(奇数)"},
            "W_nmos": {"default": "1u", "min": "0.18u", "max": "10u", "unit": "μm", "desc": "NMOS宽度"},
            "L_nmos": {"default": "0.18u", "min": "0.18u", "max": "1u", "unit": "μm", "desc": "NMOS长度"},
            "W_pmos": {"default": "2u", "min": "0.18u", "max": "20u", "unit": "μm", "desc": "PMOS宽度"},
            "L_pmos": {"default": "0.18u", "min": "0.18u", "max": "1u", "unit": "μm", "desc": "PMOS长度"},
            "Vdd": {"default": "1.8", "min": "0.8", "max": "3.3", "unit": "V", "desc": "电源电压"}
        },
        "analysis": ["tran"]
    }, 

    "common_source": {
        "name": "共源放大器 (Common Source)",
        "description": "模拟电路的核心单元。学习如何通过调整偏置和负载来控制增益。",
        "category": "模拟电路",
        "parameters": {
            "W": {"default": "10u", "min": "1u", "max": "100u", "unit": "μm", "desc": "MOS管宽度"},
            "L": {"default": "1u", "min": "0.18u", "max": "5u", "unit": "μm", "desc": "MOS管长度"},
            "Rd": {"default": "5k", "min": "1k", "max": "50k", "unit": "Ω", "desc": "负载电阻"},
            "Vbias": {"default": "0.7", "min": "0.4", "max": "1.2", "unit": "V", "desc": "栅极偏置电压"},
            "Vdd": {"default": "1.8", "min": "1.0", "max": "3.3", "unit": "V", "desc": "电源电压"}
        },
        "analysis": ["tran", "ac", "op"]
    },

    # ↓↓↓ 新增这一块 ↓↓↓
    "custom_draw": {
        "name": "自由SPICE设计",
        "description": "用户自定义SPICE网表仿真",
        "category": "工具",
        "parameters": {}, # 不需要预设参数
        "analysis": ["tran", "ac", "op"]
    },


}


# ==================== 核心逻辑函数 ====================

def generate_spice_netlist(circuit_type, params, analysis_type="tran"):
    """生成SPICE网表文件"""
    # 确保结果目录存在
    os.makedirs(CONFIG["result_dir"], exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{circuit_type}_{timestamp}.cir"
    filepath = os.path.join(CONFIG["result_dir"], filename)



    # ... (文件路径定义代码保持不变) ...
    
    # ↓↓↓ 新增这一块 ↓↓↓
    if circuit_type == "custom_draw":
        # 直接获取用户传来的完整 SPICE 代码
        spice_code = params.get('spice_code', '')
        if not spice_code:
            raise ValueError("未接收到 SPICE 代码")
        
        # 1. 【安检】无情地抹杀用户写的所有 wrdata 命令
        # 使用正则将 'wrdata ...' 替换为注释
        spice_code = re.sub(r'^\s*wrdata\s+.*$', '* [系统安全拦截] 用户定义的 wrdata 已被屏蔽', spice_code, flags=re.IGNORECASE|re.MULTILINE)

        # 2. 准备系统指令
        # 直接使用原始路径变量，不做任何 replace 清洗
        expected_raw_path = filepath.replace('.cir', '.raw')
        
        # 强制命令：保存 v(out)
        # 这里的路径就是 Python 原生的路径 (在 Windows 上是带反斜杠 \ 的)
        # 加引号是为了防止路径中间有空格
        system_cmd = f"\n* --- 系统强制注入 ---\nwrdata {expected_raw_path} v(out)\n"

        # 3. 【注入】将系统指令插入到正确的位置
        if '.endc' in spice_code:
            # 如果用户写了控制块，插在 .endc 之前
            spice_code = spice_code.replace('.endc', system_cmd + "\n.endc")
        else:
            # 如果用户没写控制块，帮他补全一套
            spice_code += f"\n\n.control\nrun{system_cmd}.endc\n"




        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(spice_code)
            
        return filepath
    # ↑↑↑ 新增结束 ↑↑↑




    # 根据电路类型生成不同的网表
    if circuit_type == "cmos_inverter":
        netlist = f"""* 硅基未来AI仿真系统 - CMOS反相器
* 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
* 参数: Wn={params.get('W_nmos', '1u')}, Wp={params.get('W_pmos', '2u')}
.title CMOS Inverter Simulation
** 电源与输入信号 **
Vdd vdd 0 {params.get('Vdd', '1.8')}
Vin in 0 pulse(0 {params.get('Vdd', '1.8')} 0 10p 10p 1n 2n)
** MOSFET晶体管 **
* NMOS: 漏 栅 源 衬底 模型 W L
Mn1 out in 0 0 nmos W={params.get('W_nmos', '1u')} L={params.get('L_nmos', '0.18u')}
* PMOS: 漏 栅 源 衬底 模型 W L
Mp1 out in vdd vdd pmos W={params.get('W_pmos', '2u')} L={params.get('L_pmos', '0.18u')}
** 负载 **
Cload out 0 {params.get('Cload', '10f')}
** 模型定义 (使用典型0.18μm CMOS模型) **
.model nmos nmos level=54
+ version=4.8 binunit=1 paramchk=1 mobmod=3
+ capmod=2 igcmod=1 igbmod=1 geomod=1
+ diomod=1 rdsmod=0 rbodymod=1 rgatemod=1
+ permod=1 acnqsmod=0 trnqsmod=0
+ tox=4e-9
+ vth0=0.45 rdsw=200
.model pmos pmos level=54
+ version=4.8 binunit=1 paramchk=1 mobmod=3
+ capmod=2 igcmod=1 igbmod=1 geomod=1
+ diomod=1 rdsmod=0 rbodymod=1 rgatemod=1
+ permod=1 acnqsmod=0 trnqsmod=0
+ tox=4e-9
+ vth0=-0.45 rdsw=200
** 分析类型 **
.control
save all
tran 10p 4n
wrdata {filepath.replace('.cir', '.raw')} v(in) v(out) i(vdd)
.endc
.end
"""
    elif circuit_type == "two_stage_opamp":
        netlist = f"""* 硅基未来AI仿真系统 - 两级CMOS运算放大器
* 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
.title Two-Stage CMOS Op-Amp
** 电源 **
Vdd vdd 0 {params.get('Vdd', '1.8')}
Vss 0 vss {params.get('Vdd', '1.8')}
** 差分输入对 **
M1 3 in+ 1 vss nmos W={params.get('W1', '10u')} L={params.get('L1', '0.18u')}
M2 4 in- 1 vss nmos W={params.get('W1', '10u')} L={params.get('L1', '0.18u')}
** 有源负载 **
M3 3 3 vdd vdd pmos W=20u L=0.18u
M4 4 3 vdd vdd pmos W=20u L=0.18u
** 尾电流源 **
M5 1 bias vss vss nmos W={params.get('W5', '20u')} L={params.get('L5', '0.5u')}
** 第二级（共源级） **
M6 out 4 vdd vdd pmos W=40u L=0.18u
M7 out bias vss vss nmos W=20u L=0.5u
** 偏置电路 **
Ibias bias vss 20u
Vbias bias 0 0.9
** 补偿网络 **
Cc 4 out {params.get('Cc', '2p')}
Rc 4 5 {params.get('Rc', '2k')}
** 负载与测试信号 **
CL out 0 {params.get('CL', '5p')}
Vin+ in+ 0 ac 1 sin(0.9 0.1 1MEG)
Vin- in- 0 0.9
** 模型定义 **
.model nmos nmos level=54
+ tox=4e-9 vth0=0.45 u0=350
.model pmos pmos level=54
+ tox=4e-9 vth0=-0.45 u0=150
** 分析 **
.control
save all
op
ac dec 10 1 1G
tran 0.01u 2u
wrdata {filepath.replace('.cir', '.raw')} v(out) v(in+) i(vdd)
.endc
.end
"""

    elif circuit_type == "common_source":
        netlist = f"""* 共源放大器仿真
* 考察点：增益(Gain)与输出摆幅(Swing)
.title Common Source Amp

** 电源 **
Vdd vdd 0 {params.get('Vdd', '1.8')}

** 输入信号 (正弦波 + 直流偏置) **
* Vin = Vbias + sin(amplitude=10mV, freq=1kHz)
Vin in 0 dc {params.get('Vbias', '0.7')} ac 1 sin({params.get('Vbias', '0.7')} 10m 1k)

** 电路主体 **
* 负载电阻 Rd 连接 Vdd 和 Out
Rd vdd out {params.get('Rd', '5k')}
* NMOS 管连接 Out, In, GND
Mn1 out in 0 0 nmos W={params.get('W', '10u')} L={params.get('L', '1u')}

** 模型定义 **
.model nmos nmos level=54 version=4.8 tox=4e-9 vth0=0.5 u0=350

** 仿真设置 **
.control
save all
* 1. 瞬态分析 (看波形是否失真)
tran 10u 5m
* 2. 交流分析 (看增益和带宽)
ac dec 10 1 100Meg
* 3. 直流工作点 (看管子是否饱和)
op

* 导出数据
wrdata {filepath.replace('.cir', '.raw')} v(in) v(out)
.endc
.end
"""

    # 写入网表文件
    with open(filepath, 'w') as f:
        f.write(netlist)

    return filepath


def run_simulation(netlist_file):
    """调用ngspice运行仿真"""
    try:
        # 构造命令
        cmd = [CONFIG["simulator_path"], "-b", netlist_file]

        # 执行仿真
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CONFIG["max_sim_time"]
        )

        # 检查结果文件
        result_file = netlist_file.replace('.cir', '.raw')

        if os.path.exists(result_file):
            return {
                "success": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "data_file": result_file,
                "netlist_file": netlist_file
            }
        else:
            return {
                "success": False,
                "error": "仿真未生成数据文件",
                "stdout": result.stdout,
                "stderr": result.stderr
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"仿真超时（>{CONFIG['max_sim_time']}秒）"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_simulation_results(data_file):
    """解析仿真结果文件"""
    try:
        with open(data_file, 'r') as f:
            lines = f.readlines()

        # 简化的解析逻辑（实际需要根据.raw格式完整解析）
        results = {
            "data_points": len(lines) - 5 if len(lines) > 5 else 0,
            "raw_data": lines[:100],  # 只返回前100行作为预览
            "file_size": os.path.getsize(data_file),
            "has_voltage_data": any("v(" in line.lower() for line in lines),
            "has_current_data": any("i(" in line.lower() for line in lines)
        }

        return results
    except Exception as e:
        return {"error": f"解析结果失败: {str(e)}"}


def ai_analyze_results(circuit_type, params, sim_results):
    """AI分析仿真结果并生成建议"""

    suggestions = []
    performance_metrics = {}

    if circuit_type == "cmos_inverter":
        # 分析反相器性能
        suggestions.append({
            "id": "inv_switching_speed",
            "title": "开关速度分析",
            "description": "基于W/L比例估算开关特性",
            "insight": f"NMOS宽长比: {params.get('W_nmos', '1u')}/{params.get('L_nmos', '0.18u')}, PMOS宽长比: {params.get('W_pmos', '2u')}/{params.get('L_pmos', '0.18u')}",
            "recommendation": "增加PMOS宽度可改善上升时间，增加NMOS宽度可改善下降时间",
            "confidence": 0.85
        })

        # 计算理论性能
        w_ratio = float(params.get('W_pmos', '2u').replace('u', '')) / float(
            params.get('W_nmos', '1u').replace('u', ''))
        performance_metrics["w_ratio_pn"] = round(w_ratio, 2)
        performance_metrics["vdd"] = params.get('Vdd', '1.8')

        if w_ratio < 1.5:
            suggestions.append({
                "id": "inv_ratio_optimize",
                "title": "宽长比优化建议",
                "description": "PMOS/NMOS宽度比较低可能导致上升沿缓慢",
                "recommendation": "将PMOS宽度增加到NMOS宽度的1.5-2.5倍，以实现对称的上升/下降时间",
                "confidence": 0.9
            })

    elif circuit_type == "two_stage_opamp":
        # 分析运放性能
        suggestions.append({
            "id": "opamp_compensation",
            "title": "频率补偿分析",
            "description": "米勒补偿电容Cc影响带宽与稳定性",
            "insight": f"补偿电容: {params.get('Cc', '2p')}, 补偿电阻: {params.get('Rc', '2k')}",
            "recommendation": "增大Cc可提高相位裕度但降低带宽，需要权衡设计",
            "confidence": 0.88
        })

        # 估算增益带宽积
        gm_estimate = np.sqrt(float(params.get('W1', '10u').replace('u', '')) / float(
            params.get('L1', '0.18u').replace('u', ''))) * 100e-6
        gbw_estimate = gm_estimate / (2 * np.pi * float(params.get('Cc', '2p').replace('p', '')) * 1e-12) / 1e6

        performance_metrics["estimated_gbw_mhz"] = round(gbw_estimate, 1)
        performance_metrics["estimated_gain_db"] = 60  # 典型值

        if gbw_estimate < 50:
            suggestions.append({
                "id": "opamp_gbw_low",
                "title": "带宽优化建议",
                "description": f"估算增益带宽积约为{round(gbw_estimate, 1)}MHz，有提升空间",
                "recommendation": "增加输入管跨导（增大W1或减小L1）或减小补偿电容Cc",
                "confidence": 0.8
            })



    if circuit_type == "custom_draw":
        # 调用上面的通用分析器
        final_score, suggestions, performance_metrics = analyze_custom_circuit(sim_results)
        
        # 如果没有生成任何建议，给个默认的
        if not suggestions:
            suggestions.append({
                "title": "仿真完成",
                "description": "电路已成功运行，请查看波形图进行详细分析。",
                "recommendation": "确保你的输出节点在 SPICE wrdata 命令的最后一位。",
                "confidence": 0.5
            })

        return {
            "score": final_score,
            "suggestions": suggestions,
            "performance_metrics": performance_metrics,
            "analysis_timestamp": datetime.now().isoformat()
        }

    return {
        "suggestions": suggestions,
        "performance_metrics": performance_metrics,
        "analysis_timestamp": datetime.now().isoformat()
    }







def analyze_custom_circuit(sim_results):
    """
    全能电路分析器 (Industrial Grade)
    输入: sim_results (包含 raw_data 的字典)
    输出: score (总分), suggestions (建议列表), metrics (关键指标字典)
    """
    suggestions = []
    metrics = {}
    score = 0
    
    # ----------------------------------------------------
    # 1. 数据清洗与分流 (Data Cleaning & Routing)
    # ----------------------------------------------------
    raw_data = sim_results.get('raw_data', [])
    if not raw_data:
        return 0, [{"title": "无数据", "description": "仿真未生成有效数据", "confidence": 1.0}], {}

    # 我们需要把原始文本数据解析成 numpy 数组
    # 同时也判断这是时域数据(.tran)还是频域数据(.ac)
    data_matrix = []
    is_ac_analysis = False
    
    try:
        for line in raw_data:
            line = line.strip()
            # 跳过标题和空行
            if not line or line.startswith('*') or line.startswith('Title') or line.startswith('Date') or line.startswith('Plotname'):
                continue
            
            # 检测是否包含频率信息 (AC分析通常第一列是 frequency)
            # 这是一个简单的启发式检测
            if 'Hz' in line or 'frequency' in line.lower():
                is_ac_analysis = True
                continue
            
            # 尝试解析数字
            parts = line.split()
            try:
                # 过滤非数字行
                nums = [float(x) for x in parts]
                if nums:
                    data_matrix.append(nums)
            except ValueError:
                continue
                
        if not data_matrix:
            return 0, [{"title": "解析失败", "description": "数据格式无法识别", "confidence": 1.0}], {}

        data_np = np.array(data_matrix)
        
        # ----------------------------------------------------
        # 2. 智能路由：根据第一列的数据特征决定分析模式
        # ----------------------------------------------------
        # 如果第一列包含极大范围的数值(比如1Hz到1GHz)，通常是AC分析
        # 如果第一列是微小的线性递增(比如0到10us)，通常是Tran分析
        
        col0_range = np.ptp(data_np[:, 0]) # Peak-to-peak (max - min)
        col0_max = np.max(data_np[:, 0])
        
        # 简单的判定逻辑：如果X轴最大值很大(>1MHz)或者在这个上下文中被标记为AC
        # 注意：这里为了稳健，最好是在 request 里明确告知是 AC 还是 TRAN
        # 但既然是 Custom Circuit，我们尝试自动探测：
        
        # 如果数据行数很多，且第一列是对数分布，或者是频率，走 AC 逻辑
        # 这里为了演示，我们假设数据里包含 .ac 和 .tran 的混合结果可能会很复杂
        # 建议：让 SPICE 网表只输出一种主要分析数据，或者分两个文件。
        # 现在的逻辑假设：如果用户跑了 AC，数据就是频域的。
        
        if col0_max > 1e6 or is_ac_analysis: 
            # ---> 进入频域分析 (AC Domain)
            score, s_list, m_dict = _analyze_ac_domain(data_np)
        else:
            # ---> 进入时域分析 (Time Domain)
            score, s_list, m_dict = _analyze_time_domain(data_np)
            
        return score, suggestions + s_list, {**metrics, **m_dict}

    except Exception as e:
        return 0, [{"title": "分析崩溃", "description": f"分析器内部错误: {str(e)}", "confidence": 0.5}], {}


def _analyze_time_domain(data):
    """时域分析子函数：看波形、摆幅、是否存活"""
    score = 0
    suggestions = []
    metrics = {}
    
    # 假设：第1列(索引0)是时间，最后一列是输出，倒数第二列是输入(如果有的话)
    time = data[:, 0]
    # 自动寻找波动最大的一列作为 Output (排除时间列)
    voltage_cols = data[:, 1:]
    std_devs = np.std(voltage_cols, axis=0)
    output_idx = np.argmax(std_devs)
    v_out = voltage_cols[:, output_idx]
    
    # 1. 生存检查
    v_max = np.max(v_out)
    v_min = np.min(v_out)
    swing = v_max - v_min
    metrics['swing_v'] = round(swing, 3)
    metrics['dc_offset_v'] = round(np.mean(v_out), 3)
    
    if swing < 0.01: # 小于 10mV 认为是死的
        return 10, [{"title": "电路无响应", "description": "输出端几乎没有信号波动 (Dead Circuit)。", "recommendation": "检查偏置电压或连接是否断开。", "confidence": 0.9}], metrics
    
    score += 40 # 活的就给基础分
    
    # 2. 削顶检测 (Assuming 1.8V Vdd)
    if v_max > 1.7 or v_min < 0.1:
        score -= 10
        suggestions.append({"title": "严重失真", "description": "波形触顶/触底，发生削顶失真。", "recommendation": "调整偏置点或减小输入幅度。", "confidence": 0.8})
    else:
        score += 20
        suggestions.append({"title": "波形完整", "description": f"输出摆幅 {swing:.2f}V，位于线性区。", "confidence": 1.0})

    metrics['mode'] = 'Transient (时域)'
    return score, suggestions, metrics


    """频域分析子函数：看增益、带宽"""
    score = 0
    suggestions = []
    metrics = {}
    
    # AC 数据通常格式：Frequency, Magnitude(dB), Phase(Degree)
    # 或者 Frequency, Real, Imag。
    # 我们假设使用 wrdata 时，ngspice 输出的是 mag(v(out)) 和 ph(v(out)) 
    # 或者我们需要计算。这里假设数据已经是 [Freq, V_out_mag, ...]
    
    freq = data[:, 0]
    # 假设第2列是幅度。如果是复数需要处理，这里简化假设是幅度
    mag = data[:, 1] 
    
    # 如果数据不是 dB，转换成 dB: 20 * log10(mag)
    # 简单的启发式：如果幅度都在 -100 到 100 之间，可能是 dB。如果都是 0.001 这种，是线性值。
    if np.mean(mag) < 5 and np.mean(mag) > -5: # 可能是线性的小信号增益
         mag_db = 20 * np.log10(mag + 1e-12) # 避免 log(0)
    else:
         mag_db = mag # 假设已经是 dB
         
    # 1. 计算低频增益 (DC Gain)
    dc_gain = mag_db[0]
    metrics['gain_db'] = round(dc_gain, 2)
    
    # 2. 计算带宽 (-3dB)
    target_gain = dc_gain - 3.0
    # 找最接近 -3dB 点的频率
    idx_3db = (np.abs(mag_db - target_gain)).argmin()
    bw = freq[idx_3db]
    metrics['bandwidth_hz'] = f"{bw:.2e}"
    
    # 3. 打分
    if dc_gain > 0:
        score += 30
        if dc_gain > 20: score += 20 # 增益够大
    else:
        suggestions.append({"title": "无增益", "description": "电路表现为衰减器。", "confidence": 0.9})
        
    if bw > 1e6: # > 1MHz
        score += 20
        suggestions.append({"title": "高频性能好", "description": f"带宽达到 {bw/1e6:.1f}MHz。", "confidence": 0.8})
    else:
        suggestions.append({"title": "带宽受限", "description": f"带宽仅为 {bw/1000:.1f}kHz。", "confidence": 0.8})
        
    metrics['mode'] = 'AC Analysis (频域)'
    return score, suggestions, metrics



def _analyze_ac_domain(data):
    """频域分析子函数 (完美支持复数/反相信号版)"""
    score = 0
    suggestions = []
    metrics = {}
    
    try:
        if data.size == 0:
            return 0, [], {}

        freq = data[:, 0]
        
        # 判断数据列数
        # 如果是3列 (Freq, Real, Imag)，说明是 v(out)
        # 如果是2列 (Freq, Mag)，说明是 vdb(out)
        if data.shape[1] >= 3:
            real = data[:, 1]
            imag = data[:, 2]
            # 【关键修复】计算复数的模长 (Magnitude)
            # 这样 -4.078 就变成了 4.078，解决了 log(负数) 的 NaN 问题
            mag = np.sqrt(real**2 + imag**2)
        else:
            mag = data[:, 1] # 假设已经是幅度

        # 防止 log(0)
        mag = np.maximum(mag, 1e-12)
        
        # 转换 dB
        # 判定：如果平均值很小(<10)，可能是线性增益，需要转dB
        # 这里的 4.078 显然是线性值
        if np.mean(mag) < 20: 
             mag_db = 20 * np.log10(mag)
        else:
             mag_db = mag # 假设已经是dB
             
        # 1. 计算直流增益 (取低频点)
        if len(mag_db) > 0:
            dc_gain = mag_db[0]
            metrics['gain_db'] = round(float(dc_gain), 2)
        else:
            metrics['gain_db'] = 0.0
        
        # 2. 计算带宽 (-3dB)
        try:
            target_gain = metrics['gain_db'] - 3.0
            # 找最接近 target_gain 的点
            idx_3db = (np.abs(mag_db - target_gain)).argmin()
            
            # 只有当增益真的下降了才算有效带宽
            if mag_db[idx_3db] <= target_gain + 0.5:
                bw = freq[idx_3db]
                metrics['bandwidth_hz'] = f"{bw:.2e}"
            else:
                metrics['bandwidth_hz'] = "> 1GHz" # 你的这个电路目前就是这种情况
                
        except:
            metrics['bandwidth_hz'] = "N/A"

        # 3. 打分逻辑
        if metrics['gain_db'] > 10: 
            score += 30
            suggestions.append({"title": "增益达标", "description": f"低频增益 {metrics['gain_db']}dB (约 {round(10**(metrics['gain_db']/20), 1)}倍)", "confidence": 0.9})
        elif metrics['gain_db'] < 0:
            suggestions.append({"title": "信号衰减", "description": "输出信号幅度小于输入", "confidence": 0.8})

        if "1G" in str(metrics['bandwidth_hz']):
             score += 20
             suggestions.append({"title": "带宽极宽", "description": "模型未包含寄生电容，呈现理想高频特性", "confidence": 0.8, "insight": "在 .model 中添加 Cgs/Cgd 可模拟真实带宽"})

        metrics['mode'] = 'AC Analysis'
        
    except Exception as e:
        print(f"AC分析计算错误: {e}")
        # 兜底防止 NaN
        metrics['gain_db'] = 0.0
        metrics['bandwidth_hz'] = "Error"
        return 0, [{"title": "分析错误", "description": "无法计算频域指标"}], metrics
        
    return score, suggestions, metrics





import random
import copy

def optimize_circuit_loop(circuit_type, initial_code, max_steps=20):
    """
    核心 AI 优化循环：自动寻找最优参数
    """
    # 1. 提取所有可调参数 (寻找 .param name=value)
    # 正则逻辑：匹配 .param 变量名=数值(可能带单位)
    param_pattern = re.compile(r'\.param\s+(\w+)\s*=\s*([\d\.]+[munpkm]?)', re.IGNORECASE)
    
    # 找到初始参数表 {'R_load': '10k', 'W_in': '5u'}
    params = dict(param_pattern.findall(initial_code))
    
    if not params:
        return {"error": "未找到可优化参数 (请使用 .param 定义变量)"}

    # 记录最佳状态
    best_params = params.copy()
    best_score = -9999
    best_metrics = {}
    history = [] # 记录优化过程，给前端画折线图用

    current_params = params.copy()
    
    print(f"--- 开始 AI 优化 (共 {max_steps} 轮) ---")
    
    for step in range(max_steps):
        # A. 突变：基于当前参数随机微调
        # 策略：随机选一个参数，乘以 0.5 ~ 1.5 之间的系数
        trial_params = _mutate_params(best_params)
        
        # B. 生成代码：把新参数填回 SPICE 代码
        trial_code = _apply_params_to_code(initial_code, trial_params)
        
        # C. 仿真：复用你现有的基础设施！
        # 注意：这里要生成临时文件，别把用户原始文件覆盖了
        # 我们利用 generate_spice_netlist 的沙盒机制，传入新代码即可
        try:
            # 伪造一个 params 字典传入
            temp_req_params = {'spice_code': trial_code}
            # netlist_path, data_path = generate_spice_netlist("custom_draw", temp_req_params)
            netlist_path = generate_spice_netlist("custom_draw", temp_req_params)
            
            # 运行仿真
            sim_res = run_simulation(netlist_path)
            
            # D. 判卷：调用 analyze_custom_circuit 打分
            # 我们只关心 simulation_results 里的 raw_data
            parsed_res = parse_simulation_results(sim_res['data_file'])
            score, _, metrics = analyze_custom_circuit(parsed_res)
            
            # E. 优胜劣汰
            # 这里我们简单粗暴：主要看 'gain_db' (如果是放大器) 或者 total 'score'
            # 假设 analyze_custom_circuit 返回的 score 是综合得分
            
            improved = False
            if score > best_score:
                best_score = score
                best_params = trial_params.copy()
                best_metrics = metrics
                improved = True
            
            # 记录历史
            history.append({
                "step": step + 1,
                "score": score,
                "gain": metrics.get('gain_db', 0),
                "params": trial_params,
                "improved": improved
            })
            
        except Exception as e:
            print(f"优化步骤 {step} 失败: {e}")
            continue

    return {
        "success": True,
        "best_params": best_params,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "history": history
    }

# --- 辅助函数 ---

def _mutate_params(params):
    """随机突变参数"""
    new_params = params.copy()
    # 随机选一个参数修改
    key = random.choice(list(params.keys()))
    val_str = params[key]
    
    # 解析数值和单位 (简单处理：假设无单位或常见单位)
    # 比如 10k -> 10, k
    match = re.match(r'([\d\.]+)([a-zA-Z]*)', val_str)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        
        # 突变逻辑：0.8倍 到 1.2倍 之间波动
        factor = random.uniform(0.8, 1.2)
        new_val = val * factor
        
        # 格式化回去 (保留2位小数)
        new_params[key] = f"{new_val:.2f}{unit}"
        
    return new_params

def _apply_params_to_code(code, params):
    """将参数字典替换回 SPICE 代码"""
    new_code = code
    for k, v in params.items():
        # 正则替换 .param k=old_val 为 .param k=new_val
        new_code = re.sub(fr'\.param\s+{k}\s*=\s*[\w\.]+', f'.param {k}={v}', new_code, flags=re.IGNORECASE)
    return new_code




# ==================== API 路由 ====================

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """获取系统信息"""
    return jsonify({
        "name": CONFIG["system_name"],
        "version": CONFIG["version"],
        "status": "运行正常",
        "circuits_available": len(CIRCUIT_LIBRARY),
        "simulator_engine": "硅基未来仿真引擎"
    })


@app.route('/api/circuits', methods=['GET'])
def get_circuits():
    """获取可用电路列表"""
    circuits = []
    for circuit_id, circuit_info in CIRCUIT_LIBRARY.items():
        circuits.append({
            "id": circuit_id,
            "name": circuit_info["name"],
            "description": circuit_info["description"],
            "category": circuit_info["category"],
            "parameters": circuit_info["parameters"],
            "analyses": circuit_info["analysis"]
        })
    return jsonify(circuits)


@app.route('/api/simulate', methods=['POST'])
def simulate():
    """执行电路仿真"""
    try:
        data = request.json
        circuit_type = data.get('circuit_type')
        params = data.get('parameters', {})
        analysis = data.get('analysis', 'tran')

        if circuit_type not in CIRCUIT_LIBRARY:
            return jsonify({"error": "未知的电路类型"}), 400

        # 1. 生成SPICE网表
        netlist_file = generate_spice_netlist(circuit_type, params, analysis)

        # 2. 运行仿真
        sim_result = run_simulation(netlist_file)

        if not sim_result.get("success"):
            return jsonify({
                "success": False,
                "error": sim_result.get("error"),
                "details": sim_result.get("stderr", "")
            }), 500

        # 3. 解析结果
        parsed_results = parse_simulation_results(sim_result["data_file"])

        # 4. AI分析
        ai_analysis = ai_analyze_results(circuit_type, params, parsed_results)

        # 5. 返回完整结果
        return jsonify({
            "success": True,
            "circuit_type": circuit_type,
            "circuit_name": CIRCUIT_LIBRARY[circuit_type]["name"],
            "parameters": params,
            "simulation": {
                "netlist_file": netlist_file,
                "data_file": sim_result["data_file"],
                "stdout_preview": sim_result["stdout"][:500] + "..." if len(sim_result["stdout"]) > 500 else sim_result[
                    "stdout"],
                "return_code": sim_result.get("returncode", 0)
            },
            "results": parsed_results,
            "ai_analysis": ai_analysis,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/examples/<circuit_type>', methods=['GET'])
def get_example(circuit_type):
    """获取电路示例配置"""
    if circuit_type in CIRCUIT_LIBRARY:
        example_params = {}
        for param_name, param_info in CIRCUIT_LIBRARY[circuit_type]["parameters"].items():
            example_params[param_name] = param_info["default"]

        return jsonify({
            "circuit_type": circuit_type,
            "name": CIRCUIT_LIBRARY[circuit_type]["name"],
            "parameters": example_params,
            "description": CIRCUIT_LIBRARY[circuit_type]["description"]
        })
    else:
        return jsonify({"error": "电路类型不存在"}), 404




@app.route('/api/draw/auto', methods=['POST'])
def draw_auto_layout():
    try:
        spice_code = request.json.get('spice_code', '')
        if not spice_code:
            return jsonify({"error": "代码为空"}), 400

        # 这一步就是"自动整洁画布"的魔法
        svg_content = layout_spice_graph(spice_code)

        return jsonify({"svg": svg_content})
    except Exception as e:
        # 如果没装 Graphviz 软件会报错
        return jsonify({"error": f"Graphviz 错误: {str(e)}"}), 500



@app.route('/api/optimize', methods=['POST'])
def optimize():
    try:
        data = request.json
        spice_code = data.get('spice_code', '')
        
        if not spice_code:
            return jsonify({"error": "代码为空"}), 400
            
        # 启动优化循环
        result = optimize_circuit_loop("custom_draw", spice_code)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ============= 前端 ==============
@app.route('/')
@app.route('/index.html')
def index():
    return send_file('../frontend/index.html')

if __name__ == '__main__':
    print(f"=== {CONFIG['system_name']} v{CONFIG['version']} ===")
    print(f"仿真引擎: {CONFIG['simulator_path']}")
    print(f"可用电路: {len(CIRCUIT_LIBRARY)} 种")
    print(f"API服务运行在: http://localhost:5000")
    print("=" * 50)

    app.run(host='0.0.0.0', port=5000, debug=True)