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

app = Flask(__name__)
CORS(app)  # 允许跨域请求
# 系统配置
CONFIG = {
    "system_name": "硅基未来AI仿真系统",
    "version": "2.0",
    "simulator_path": "ngspice",  # 假设ngspice已在PATH中
    "max_sim_time": 30,  # 最大仿真时间(秒)
    "result_dir": "./simulation_results"}
# CMOS电路库定义
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
    }}
def generate_spice_netlist(circuit_type, params, analysis_type="tran"):
    """生成SPICE网表文件"""
    
    # 确保结果目录存在
    os.makedirs(CONFIG["result_dir"], exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{circuit_type}_{timestamp}.cir"
    filepath = os.path.join(CONFIG["result_dir"], filename)
    
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
save all{analysis_type} 0 4n 0.01p
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
        w_ratio = float(params.get('W_pmos', '2u').replace('u', '')) / float(params.get('W_nmos', '1u').replace('u', ''))
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
        gm_estimate = np.sqrt(float(params.get('W1', '10u').replace('u', '')) / float(params.get('L1', '0.18u').replace('u', ''))) * 100e-6
        gbw_estimate = gm_estimate / (2 * np.pi * float(params.get('Cc', '2p').replace('p', '')) * 1e-12) / 1e6
        
        performance_metrics["estimated_gbw_mhz"] = round(gbw_estimate, 1)
        performance_metrics["estimated_gain_db"] = 60  # 典型值
        
        if gbw_estimate < 50:
            suggestions.append({
                "id": "opamp_gbw_low",
                "title": "带宽优化建议",
                "description": f"估算增益带宽积约为{round(gbw_estimate,1)}MHz，有提升空间",
                "recommendation": "增加输入管跨导（增大W1或减小L1）或减小补偿电容Cc",
                "confidence": 0.8
            })
    
    return {
        "suggestions": suggestions,
        "performance_metrics": performance_metrics,
        "analysis_timestamp": datetime.now().isoformat()
    }
# ==================== API 路由 ====================
@app.route('/api/system/info', methods=['GET'])def system_info():
    """获取系统信息"""
    return jsonify({
        "name": CONFIG["system_name"],
        "version": CONFIG["version"],
        "status": "运行正常",
        "circuits_available": len(CIRCUIT_LIBRARY),
        "simulator_engine": "硅基未来仿真引擎"
    })
@app.route('/api/circuits', methods=['GET'])def get_circuits():
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
@app.route('/api/simulate', methods=['POST'])def simulate():
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
                "stdout_preview": sim_result["stdout"][:500] + "..." if len(sim_result["stdout"]) > 500 else sim_result["stdout"],
                "return_code": sim_result.get("returncode", 0)
            },
            "results": parsed_results,
            "ai_analysis": ai_analysis,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
@app.route('/api/analyze/performance', methods=['POST'])def analyze_performance():
    """专业性能分析"""
    data = request.json
    circuit_type = data.get('circuit_type')
    params = data.get('parameters', {})
    
    analysis = ai_analyze_results(circuit_type, params, {})
    
    return jsonify({
        "circuit_type": circuit_type,
        "parameters": params,
        "performance_analysis": analysis,
        "recommendations": analysis["suggestions"]
    })
@app.route('/api/examples/<circuit_type>', methods=['GET'])def get_example(circuit_type):
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
if __name__ == '__main__':
    print(f"=== {CONFIG['system_name']} v{CONFIG['version']} ===")
    print(f"仿真引擎: {CONFIG['simulator_path']}")
    print(f"可用电路: {len(CIRCUIT_LIBRARY)} 种")
    print(f"API服务运行在: http://localhost:5000")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
