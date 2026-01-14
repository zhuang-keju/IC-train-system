import graphviz
import os


def layout_spice_graph(netlist_str):
    """
    将 SPICE 网表转换为自动布局的图
    原理：SPICE Component = Edge (边), SPICE Node = Node (节点)
    """
    # 创建一个有向图 (Digraph)，方向从左到右 (LR)
    dot = graphviz.Digraph(comment='Circuit Graph', format='svg')
    dot.attr(rankdir='LR')  # Left to Right 布局，像电路图一样

    # 设置节点样式 (电路节点画成小圆点)
    dot.attr('node', shape='circle', style='filled', color='black',
             fontcolor='white', width='0.3', fixedsize='true')

    # 设置边样式 (连线)
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    lines = netlist_str.strip().split('\n')

    # 简单的解析器
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.') or line.startswith('wrdata') or line.startswith('run'):
        # if not line or line.startswith('*') or line.startswith('.'):
            continue

        parts = line.split()
        component_name = parts[0]

        # 提取节点 (SPICE 语法的第2、3个词通常是节点)
        # 比如: R1 nodeA nodeB 10k
        if len(parts) >= 3:
            node1 = parts[1]
            node2 = parts[2]

            # 稍微处理一下 MOSFET (通常有4个节点: D G S B)
            # M1 D G S B model... -> 我们简单点，只画 D 和 S 的连线，把 G 标在边上
            if component_name.startswith('M'):
                node_d = parts[1]
                node_g = parts[2]
                node_s = parts[3]
                # 画一条从 D 到 S 的线，标签写上名字
                dot.edge(node_d, node_s, label=f"{component_name}\n(Gate: {node_g})")

            # 处理双端元件 (R, C, L, V)
            else:
                # 在 Graphviz 里，边就是元件
                # 如果是电源 V1，我们画成反向（因为电流从正流出）
                if component_name.startswith('V'):
                    dot.edge(node2, node1, label=component_name, color='red')
                else:
                    dot.edge(node1, node2, label=component_name)

    # Graphviz 会自动计算布局！
    # 返回 SVG 源码
    return dot.pipe().decode('utf-8')