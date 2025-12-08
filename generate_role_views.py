from collections import Counter, defaultdict

from pm4py.objects.log.importer.xes import importer as xes_importer
from graphviz import Digraph

XES_PATH = "/home/zzh/data_lab/hw3/output_simplified.xes"
OUT_DIR = "/home/zzh/data_lab/hw3"


def load_log(path):
    return xes_importer.apply(path)


def build_role_activity_flows(log):
    """节点为(活动,角色)，边为直接跟随次数；同时构建角色到角色的流向"""
    act_edges = Counter()
    role_edges = Counter()
    nodes_by_role = defaultdict(set)

    for trace in log:
        prev_node = None
        prev_role = None
        for evt in trace:
            role = evt.get("org:role", "UNKNOWN")
            act = evt.get("concept:name", "UNKNOWN")
            node = f"{act}\\n({role})"
            nodes_by_role[role].add(node)
            if prev_node is not None:
                act_edges[(prev_node, node)] += 1
                role_edges[(prev_role, role)] += 1
            prev_node = node
            prev_role = role
    return act_edges, role_edges, nodes_by_role


def render_role_swimlane(act_edges, nodes_by_role, out_path, top_edges=80, min_freq=1):
    """基于(活动,角色)节点的直接跟随，按角色分cluster生成泳道图"""
    # 选边：先按频次排序，取前 top_edges，再过滤频次阈值
    sorted_edges = sorted(act_edges.items(), key=lambda kv: kv[1], reverse=True)
    if top_edges:
        sorted_edges = sorted_edges[:top_edges]
    edges = [(s, t, c) for (s, t), c in sorted_edges if c >= min_freq]

    dot = Digraph("role_swimlane", format="png")
    dot.attr(rankdir="LR", fontsize="10", labelloc="t", label="Role Swimlane (top edges)")

    # cluster by role
    for role, nodes in nodes_by_role.items():
        with dot.subgraph(name=f"cluster_{role}") as sub:
            sub.attr(label=role, style="rounded", fontsize="10")
            for n in nodes:
                sub.node(n, shape="box", fontsize="9")

    for s, t, c in edges:
        dot.edge(s, t, label=str(c), fontsize="9")

    dot.render(out_path, cleanup=True)


def render_role_flow(role_edges, out_path, top_edges=30, min_freq=1):
    """角色到角色的直接跟随图"""
    sorted_edges = sorted(role_edges.items(), key=lambda kv: kv[1], reverse=True)
    if top_edges:
        sorted_edges = sorted_edges[:top_edges]
    edges = [(s, t, c) for (s, t), c in sorted_edges if c >= min_freq]

    roles = set()
    for s, t, _ in edges:
        roles.add(s)
        roles.add(t)

    dot = Digraph("role_flow", format="png")
    dot.attr(rankdir="LR", fontsize="10", labelloc="t", label="Role-to-Role Directly-Follows")
    for r in roles:
        dot.node(r, shape="ellipse", fontsize="10")
    for s, t, c in edges:
        dot.edge(s, t, label=str(c), fontsize="9")

    dot.render(out_path, cleanup=True)


def main():
    log = load_log(XES_PATH)
    act_edges, role_edges, nodes_by_role = build_role_activity_flows(log)

    render_role_swimlane(
        act_edges,
        nodes_by_role,
        f"{OUT_DIR}/role_swimlane",
        top_edges=80,
        min_freq=2,
    )
    render_role_flow(
        role_edges,
        f"{OUT_DIR}/role_flow",
        top_edges=30,
        min_freq=2,
    )
    print("生成完成：role_swimlane.png, role_flow.png")


if __name__ == "__main__":
    main()

