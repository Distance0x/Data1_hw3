#! /usr/bin/env python
#coding=utf-8
import os
import random
from pathlib import Path

# 设置随机种子，保证可复现
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
TEST_DIR = DATA_DIR / 'test'
TRAIN_FILE = DATA_DIR / 'train'

test_dir = os.fspath(TEST_DIR)
train_file = os.fspath(TRAIN_FILE)

class Event:
    """事件节点：包含一个 activity（谓词）以及角色 role。"""
    def __init__(self, activity, role):
        self.activity = activity
        self.role = role
        
    def __str__(self):
        return f"Event(activity='{self.activity}', role={self.role})"
        

class Question:
    """完形填空式问题：前缀事件链、候选事件集合与正确答案索引。
    example:
        Question(answer=0, 
                context=["Event(activity='挂号', role=['病人'])", "Event(activity='开方', role=['门诊科室医生'])", "Event(activity='最后一次接诊', role=['门诊科室医生'])", "Event(activity='首次接诊', role=['门诊科室医生'])", "Event(activity='收费', role=['收费人员'])", "Event(activity='配药', role=['药房'])"], 
                choices=["Event(activity='发药', role=['药房'])", "Event(activity='首次接诊', role=['门诊科室医生'])", "Event(activity='配药', role=['药房'])", "Event(activity='开方', role=['门诊科室医生'])", "Event(activity='收费', role=['收费人员'])"])    """
    def __init__(self, answer, context, choices):
        self.answer = answer
        self.context = context
        self.choices = choices
    
    def __str__(self):
        context_str = [str(e) for e in self.context]
        choices_str = [str(e) for e in self.choices]
        return f"Question(answer={self.answer}, context={context_str}, choices={choices_str})"


def get_activity_role_map():
    """从训练集构建 activity -> roles 的映射"""
    activity_roles = {}
    all_roles = set()
    
    if not os.path.exists(train_file):
        return activity_roles, list(all_roles)

    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            parts = line.split('<@>')
            # parts[0] is answer index
            for i in range(1, len(parts)):
                event_str = parts[i].strip()
                if not event_str: continue
                
                event_parts = event_str.split('<|>')
                activity = event_parts[0] if len(event_parts) > 0 else ''
                # 只有当角色存在时才记录
                if len(event_parts) > 1 and event_parts[1]:
                    roles = event_parts[1].split('<&>')
                    for r in roles:
                        if r:
                            if activity not in activity_roles:
                                activity_roles[activity] = set()
                            activity_roles[activity].add(r)
                            all_roles.add(r)
                            
    return activity_roles, list(all_roles)


def read_question_corpus():
    """读取训练文件并转换
    
    文件格式：每行以 '<@>' 分隔，第一部分是答案索引（int），然后是前缀事件，最后5个是候选事件，
        事件字符串用'<|>'分隔activity和role，role，用'<&>'分隔实体。
    返回Question列表。
    """

    # TODO start：在下方实现函数
    questions = []
    
    # 获取补充角色所需的信息
    activity_roles, all_roles = get_activity_role_map()
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('<@>')
            
            # 第一部分是答案索引
            answer = int(parts[0])
            
            # 解析所有事件
            events = []
            valid_question = True # 标记该问题是否有效（所有事件都有角色）
            
            for i in range(1, len(parts)):
                event_str = parts[i].strip()
                if not event_str:
                    continue
                # 解析 activity<|>role 格式
                event_parts = event_str.split('<|>')
                activity = event_parts[0] if len(event_parts) > 0 else ''
                role = event_parts[1].split('<&>') if len(event_parts) > 1 and event_parts[1] else []
                
                # 如果没有角色，尝试补充
                if not role:
                    if activity in activity_roles:
                        role = [random.choice(list(activity_roles[activity]))]
                    else:
                        # 如果无法从训练集中找到对应的角色，则标记为无效
                        valid_question = False
                        break
                
                events.append(Event(activity, role))
            
            if not valid_question:
                continue

            # 最后5个是候选事件（choices），前面的都是前缀事件（context）
            # 前缀事件数量不固定，候选事件固定为5个
            if len(events) >= 6:  # 至少1个context + 5个choices
                context = events[:-5]  # 除最后5个外都是前缀事件
                choices = events[-5:]  # 最后5个是候选事件
                questions.append(Question(answer, context, choices))
    
    print(f"Loaded {len(questions)} valid training questions.")
    return questions
    # TODO end

def read_c_and_j_corpus():
    """读取测试集，构造Question对象列表。
    
    从目录读取文件，每个文件包含1个流程（每行格式：id activity role）。
    这里流程中没有候选事件，需要**从所有活动中随机选择构造候选事件**，候选包含1个正确答案和4个随机干扰。
    生成候选事件时可在正确的事件外通过随机生成的方式加入4个候选事件，以匹配任务的统一格式。
    返回Question列表。
    """
    
    # TODO start：在下方实现函数
    questions = []
    
    # 首先收集所有可能的活动和角色，用于生成干扰项
    all_activities = set()
    all_roles = set()
    all_events = []  # 存储所有事件用于随机选择
    
    # 第一遍扫描：收集所有活动和角色
    test_files = [f for f in os.listdir(test_dir) if f.startswith('chain-')]
    
    for filename in test_files:
        filepath = os.path.join(test_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    activity = parts[1]
                    role = parts[2] if len(parts) > 2 else ''
                    all_activities.add(activity)
                    all_roles.add(role)
                    all_events.append(Event(activity, [role] if role else []))
    
    all_activities = list(all_activities)
    all_roles = list(all_roles)
    
    # 第二遍扫描：构造Question对象
    for filename in test_files:
        filepath = os.path.join(test_dir, filename)
        events = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    # id activity role
                    activity = parts[1]
                    role = parts[2] if len(parts) > 2 else ''
                    events.append(Event(activity, [role] if role else []))
        
        # 对于每个流程，我们可以构造多个问题
        # 使用前n-1个事件作为context，最后一个作为正确答案
        if len(events) >= 2:
            context = events[:-1]
            correct_event = events[-1]
            
            # 生成4个随机干扰项
            choices = [correct_event]  # 正确答案放在第一个位置
            
            # 随机选择4个不同的干扰事件
            available_events = [e for e in all_events if e.activity != correct_event.activity]
            if len(available_events) >= 4:
                distractors = random.sample(available_events, 4)
            else:
                # 如果可用事件不足，则随机生成
                distractors = []
                for _ in range(4):
                    rand_activity = random.choice(all_activities)
                    rand_role = random.choice(all_roles) if all_roles else ''
                    distractors.append(Event(rand_activity, [rand_role] if rand_role else []))
            
            choices.extend(distractors)
            
            # # 清空干扰项的角色（与训练集格式一致：正确答案有角色，干扰项无角色）
            # for i in range(1, len(choices)):
            #     choices[i].role = []
            
            # 打乱候选顺序，并记录正确答案的新位置
            answer_idx = 0  # 正确答案当前在位置0
            indices = list(range(5))
            random.shuffle(indices)
            shuffled_choices = [choices[i] for i in indices]
            new_answer = indices.index(0)  # 找到正确答案在打乱后的位置
            
            questions.append(Question(new_answer, context, shuffled_choices))
    
    return questions
    # TODO end