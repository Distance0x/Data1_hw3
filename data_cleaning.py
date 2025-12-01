#!/usr/bin/env python3
"""
HIS数据清洗与XES转换脚本

功能:
1. 数据清洗与预处理
2. 转换为XES格式（符合 http://www.xes-standard.org/ 标准）
3. 支持多种清洗策略

Usage:
    python data_cleaning.py HISData2025.csv output.xes [--strategy basic|merged|simplified]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import Element, SubElement, ElementTree
from xml.dom import minidom

# ============================================================================
# 数据清洗配置
# ============================================================================

# 活动合并映射（可选策略）
ACTIVITY_MERGE_MAP = {
    # 将首次接诊和最后一次接诊合并
    "首次接诊": "接诊",
    "最后一次接诊": None,  # None 表示删除该活动
    # 报告相关活动合并（可选）
    # "报告": None,
    # "报告审核": None,
    # "报告发布": "报告完成",
}

# 活动优先级排序（同一时间戳时的排序依据）
ACTIVITY_PRIORITY = {
    "挂号": 1,
    "首次接诊": 2,
    "接诊": 2,
    "开单": 3,
    "开方": 3,
    "收费": 4,
    "检查": 5,
    "检验": 5,
    "报告审核": 6,
    "报告": 7,
    "报告发布": 8,
    "报告完成": 8,
    "配药": 9,
    "发药": 10,
    "最后一次接诊": 11,
}

# 中英文翻译映射
TRANSLATION_MAP = {
    # Activities
    "挂号": "Registration",
    "首次接诊": "First Consultation",
    "接诊": "Consultation",
    "最后一次接诊": "Last Consultation",
    "开单": "Order Test",
    "开方": "Prescribe Medicine",
    "收费": "Payment",
    "检查": "Examination",
    "检验": "Lab Test",
    "报告审核": "Report Verification",
    "报告": "Report",
    "报告发布": "Report Release",
    "报告完成": "Report Completed",
    "配药": "Dispense Medicine",
    "发药": "Deliver Medicine",
    
    # Roles
    "收费人员": "Cashier",
    "病人": "Patient",
    "门诊科室医生": "Outpatient Doctor",
    "检查检验科医生": "Lab/Exam Doctor",
    "药房": "Pharmacy",
}


# ============================================================================
# 数据清洗函数
# ============================================================================

def load_csv(filepath: str) -> List[Dict]:
    """加载CSV文件"""
    events = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(row)
    return events


def parse_timestamp(ts_str: str) -> datetime:
    """解析时间戳"""
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")


def clean_events_basic(events: List[Dict]) -> List[Dict]:
    """
    基础清洗策略：
    - 保留所有原始活动
    - 按案例ID分组，按时间排序
    - 同一时间戳按活动优先级排序
    """
    cleaned = []
    for event in events:
        cleaned_event = {
            'case_id': event['GUAHAO_ID'],
            'activity': event['ACTIVITY'],
            'timestamp': parse_timestamp(event['ACTIVITY_START']),
            'end_time': parse_timestamp(event['ACTIVITY_END']),
            'duration': int(event['DURATION']),
            'resource': event['USER_ID'],
            'role': event['ROLE'],
        }
        cleaned.append(cleaned_event)
    
    # 排序：先按case_id，再按时间戳，最后按活动优先级
    cleaned.sort(key=lambda x: (
        x['case_id'],
        x['timestamp'],
        ACTIVITY_PRIORITY.get(x['activity'], 99)
    ))
    
    return cleaned


def clean_events_merged(events: List[Dict]) -> List[Dict]:
    """
    合并清洗策略：
    - 合并"首次接诊"和"最后一次接诊"为"接诊"
    - 去除完全重复的事件
    """
    cleaned = []
    seen = set()  # 用于去重
    
    for event in events:
        activity = event['ACTIVITY']
        
        # 应用合并映射
        if activity in ACTIVITY_MERGE_MAP:
            new_activity = ACTIVITY_MERGE_MAP[activity]
            if new_activity is None:
                continue  # 跳过该活动
            activity = new_activity
        
        # 创建去重键
        dedup_key = (
            event['GUAHAO_ID'],
            activity,
            event['ACTIVITY_START'],
        )
        
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        
        cleaned_event = {
            'case_id': event['GUAHAO_ID'],
            'activity': activity,
            'timestamp': parse_timestamp(event['ACTIVITY_START']),
            'end_time': parse_timestamp(event['ACTIVITY_END']),
            'duration': int(event['DURATION']),
            'resource': event['USER_ID'],
            'role': event['ROLE'],
        }
        cleaned.append(cleaned_event)
    
    # 排序
    cleaned.sort(key=lambda x: (
        x['case_id'],
        x['timestamp'],
        ACTIVITY_PRIORITY.get(x['activity'], 99)
    ))
    
    return cleaned


def clean_events_simplified(events: List[Dict]) -> List[Dict]:
    """
    简化清洗策略：
    - 合并接诊活动
    - 合并报告相关活动为"报告完成"
    - 每个活动类型在同一时间戳只保留一个
    """
    # 扩展合并映射
    simplified_merge = {
        "首次接诊": "接诊",
        "最后一次接诊": None,
        "报告": None,
        "报告审核": None,
        "报告发布": "报告完成",
    }
    
    cleaned = []
    seen = set()
    
    for event in events:
        activity = event['ACTIVITY']
        
        # 应用简化映射
        if activity in simplified_merge:
            new_activity = simplified_merge[activity]
            if new_activity is None:
                continue
            activity = new_activity
        
        # 创建去重键
        dedup_key = (
            event['GUAHAO_ID'],
            activity,
            event['ACTIVITY_START'],
        )
        
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        
        cleaned_event = {
            'case_id': event['GUAHAO_ID'],
            'activity': activity,
            'timestamp': parse_timestamp(event['ACTIVITY_START']),
            'end_time': parse_timestamp(event['ACTIVITY_END']),
            'duration': int(event['DURATION']),
            'resource': event['USER_ID'],
            'role': event['ROLE'],
        }
        cleaned.append(cleaned_event)
    
    cleaned.sort(key=lambda x: (
        x['case_id'],
        x['timestamp'],
        ACTIVITY_PRIORITY.get(x['activity'], 99)
    ))
    
    return cleaned


# ============================================================================
# XES 生成函数
# ============================================================================

def generate_xes(events: List[Dict], output_path: str):
    """
    生成符合XES标准的XML文件
    参考: http://www.xes-standard.org/xesstandarddefinition
    """
    # 按case_id分组
    cases = defaultdict(list)
    for event in events:
        cases[event['case_id']].append(event)
    
    # 创建XES根元素
    log = Element('log')
    log.set('xes.version', '1.0')
    log.set('xes.features', '')
    log.set('openxes.version', '1.0RC7')
    log.set('xmlns', 'http://www.xes-standard.org/')
    
    # 添加扩展声明
    extensions = [
        ('Concept', 'concept', 'http://www.xes-standard.org/concept.xesext'),
        ('Time', 'time', 'http://www.xes-standard.org/time.xesext'),
        ('Organizational', 'org', 'http://www.xes-standard.org/org.xesext'),
        ('Lifecycle', 'lifecycle', 'http://www.xes-standard.org/lifecycle.xesext'),
    ]
    
    for name, prefix, uri in extensions:
        ext = SubElement(log, 'extension')
        ext.set('name', name)
        ext.set('prefix', prefix)
        ext.set('uri', uri)
    
    # 添加全局属性定义
    # Trace级别
    trace_globals = SubElement(log, 'global')
    trace_globals.set('scope', 'trace')
    attr = SubElement(trace_globals, 'string')
    attr.set('key', 'concept:name')
    attr.set('value', 'UNKNOWN')
    
    # Event级别
    event_globals = SubElement(log, 'global')
    event_globals.set('scope', 'event')
    
    global_attrs = [
        ('string', 'concept:name', 'UNKNOWN'),
        ('date', 'time:timestamp', '1970-01-01T00:00:00.000+00:00'),
        ('string', 'org:resource', 'UNKNOWN'),
        ('string', 'org:role', 'UNKNOWN'),
        ('string', 'lifecycle:transition', 'complete'),
    ]
    
    for tag, key, value in global_attrs:
        attr = SubElement(event_globals, tag)
        attr.set('key', key)
        attr.set('value', value)
    
    # 添加分类器
    classifier = SubElement(log, 'classifier')
    classifier.set('name', 'Activity')
    classifier.set('keys', 'concept:name')
    
    classifier2 = SubElement(log, 'classifier')
    classifier2.set('name', 'Resource')
    classifier2.set('keys', 'org:resource')
    
    # 添加日志级属性
    log_name = SubElement(log, 'string')
    log_name.set('key', 'concept:name')
    log_name.set('value', 'HIS Outpatient Process Log')
    
    # 创建traces
    for case_id, case_events in sorted(cases.items()):
        trace = SubElement(log, 'trace')
        
        # Trace属性：案例ID
        trace_name = SubElement(trace, 'string')
        trace_name.set('key', 'concept:name')
        trace_name.set('value', str(case_id))
        
        # 添加事件
        for evt in case_events:
            event_elem = SubElement(trace, 'event')
            
            # 活动名称
            activity_attr = SubElement(event_elem, 'string')
            activity_attr.set('key', 'concept:name')
            activity_attr.set('value', evt['activity'])
            
            # 时间戳 (XES标准要求UTC格式，带时区和毫秒)
            time_attr = SubElement(event_elem, 'date')
            time_attr.set('key', 'time:timestamp')
            time_attr.set('value', evt['timestamp'].strftime('%Y-%m-%dT%H:%M:%S.000+08:00'))
            
            # 资源
            resource_attr = SubElement(event_elem, 'string')
            resource_attr.set('key', 'org:resource')
            resource_attr.set('value', evt['resource'])
            
            # 角色
            role_attr = SubElement(event_elem, 'string')
            role_attr.set('key', 'org:role')
            role_attr.set('value', evt['role'])
            
            # 生命周期状态
            lifecycle_attr = SubElement(event_elem, 'string')
            lifecycle_attr.set('key', 'lifecycle:transition')
            lifecycle_attr.set('value', 'complete')
            
            # 自定义属性：持续时间
            duration_attr = SubElement(event_elem, 'int')
            duration_attr.set('key', 'duration')
            duration_attr.set('value', str(evt['duration']))
            
            # 自定义属性：结束时间 (XES标准要求UTC格式)
            end_time_attr = SubElement(event_elem, 'date')
            end_time_attr.set('key', 'end_timestamp')
            end_time_attr.set('value', evt['end_time'].strftime('%Y-%m-%dT%H:%M:%S.000+08:00'))
    
    # 写入XES文件
    tree = ElementTree(log)
    with open(output_path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)
    
    # 读取并美化格式
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    dom = minidom.parseString(content)
    # 移除多余空白行
    pretty_xml = '\n'.join(
        line for line in dom.toprettyxml(indent='  ').split('\n') 
        if line.strip()
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)
    
    return len(cases), sum(len(v) for v in cases.values())


def print_statistics(events: List[Dict], title: str = "数据统计"):
    """打印清洗后的数据统计"""
    cases = defaultdict(list)
    activities = defaultdict(int)
    roles = defaultdict(int)
    
    for event in events:
        cases[event['case_id']].append(event)
        activities[event['activity']] += 1
        roles[event['role']] += 1
    
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    print(f"案例数量: {len(cases)}")
    print(f"事件总数: {len(events)}")
    print(f"平均每案例事件数: {len(events)/len(cases):.2f}")
    
    print(f"\n活动类型分布:")
    for act, count in sorted(activities.items(), key=lambda x: -x[1]):
        print(f"  {act}: {count}")
    
    print(f"\n角色分布:")
    for role, count in sorted(roles.items(), key=lambda x: -x[1]):
        print(f"  {role}: {count}")
    
    # 计算案例长度分布
    lengths = [len(v) for v in cases.values()]
    print(f"\n案例长度统计:")
    print(f"  最短: {min(lengths)} 事件")
    print(f"  最长: {max(lengths)} 事件")
    print(f"  中位数: {sorted(lengths)[len(lengths)//2]} 事件")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='HIS数据清洗与XES转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
清洗策略说明:
  basic      - 保留所有原始活动，仅排序
  merged     - 合并首次接诊和最后一次接诊为"接诊"
  simplified - 进一步合并报告相关活动

示例:
  python data_cleaning.py HISData2025.csv output.xes
  python data_cleaning.py HISData2025.csv output.xes --strategy merged
  python data_cleaning.py HISData2025.csv output.xes --strategy simplified --stats
        """
    )
    parser.add_argument('input', help='输入CSV文件路径')
    parser.add_argument('output', help='输出XES文件路径')
    parser.add_argument('--strategy', choices=['basic', 'merged', 'simplified'],
                        default='basic', help='清洗策略 (默认: basic)')
    parser.add_argument('--stats', action='store_true', help='显示详细统计信息')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}", file=sys.stderr)
        return 1
    
    print(f"加载数据: {args.input}")
    events = load_csv(args.input)
    print(f"原始事件数: {len(events)}")
    
    # 选择清洗策略
    strategy_map = {
        'basic': clean_events_basic,
        'merged': clean_events_merged,
        'simplified': clean_events_simplified,
    }
    
    print(f"应用清洗策略: {args.strategy}")
    clean_func = strategy_map[args.strategy]
    cleaned_events = clean_func(events)
    print(f"清洗后事件数: {len(cleaned_events)}")
    
    # 翻译事件数据
    print("正在翻译为英文...")
    for event in cleaned_events:
        event['activity'] = TRANSLATION_MAP.get(event['activity'], event['activity'])
        event['role'] = TRANSLATION_MAP.get(event['role'], event['role'])
    
    if args.stats:
        print_statistics(cleaned_events, f"清洗后数据统计 ({args.strategy})")
    
    # 生成XES
    print(f"\n生成XES文件: {args.output}")
    num_cases, num_events = generate_xes(cleaned_events, args.output)
    print(f"完成! 共 {num_cases} 个案例, {num_events} 个事件")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
