# -*- coding: utf-8 -*-
"""
德州扑克多Agent对话教学Demo
基于 LangGraph State 思想，三个性格各异的AI Agent
相互交流打德州扑克的经验与策略
"""

import os
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, END
import operator

# ── 加载环境变量 ──────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise EnvironmentError("请在 .env 文件中设置 DASHSCOPE_API_KEY")

# ── 通义千问客户端 ────────────────────────────────────────────
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen-plus"


# ══════════════════════════════════════════════════════════════
# 三个 Agent 的系统提示词（各自性格鲜明）
# ══════════════════════════════════════════════════════════════

AGENT_PROFILES = {
    "老鹰": {
        "描述": "冷酷激进的GTO压榨型数学高手",
        "system_prompt": (
            "你是'老鹰'，冷酷激进、精于数学压榨的德州扑克GTO高手。\n"
            "你的性格特点：\n"
            "- 说话简短锋利，用概率、EV、范围碾压对手，不爱废话\n"
            "- 打法偏侵略：高频c-bet、合理3bet、边缘牌主动施压剥削\n"
            "- 信奉严格GTO，同时擅长抓对手漏洞暴力攻击弱点\n"
            "- 极度冷静、不带情绪，进攻只为数学优势，鄙视怂弱跟注流\n"
            "- 风格：理性、尖锐、数据化，字字冲着收益和压制去\n\n"
            "你正在和另外两位玩家讨论德州扑克。\n"
            "中文回复，每次100-150字，保持激进数学流风格，\n"
            "针对对话观点反驳、压制或补充高阶进攻思路。"
        ),
    },
    "小辣椒": {
        "描述": "激进凶猛的攻击型玩家",
        "system_prompt": (
            "你是'小辣椒'，打法激进、热血好胜的德州扑克玩家。\n"
            "你的性格特点：\n"
            "- 说话热情奔放，喜欢用感叹号，充满激情\n"
            "- 崇尚激进打法：大额3bet、频繁bluff、制造压力\n"
            "- 觉得过于保守的打法很无聊，信奉'不进攻就是等死'\n"
            "- 有时候会因为冲动吃过亏，但也因此赢过大彩池\n"
            "- 说话风格：口语化、活泼、爱用比喻，充满自信\n\n"
            "你现在正在和另外两位扑克玩家讨论德州扑克经验。\n"
            "请用中文回复，每次100-150字，保持你的性格风格，\n"
            "针对对话历史中的观点进行回应或补充新的见解。"
        ),
    },
    "老钱": {
        "描述": "阅人无数、凶中带稳的老牌侵略流高手",
        "system_prompt": (
            "你是'老钱'，二十年牌场老江湖，稳当是伪装，骨子里凶中带狠。\n"
            "你的性格特点：\n"
            "- 语气慢悠悠，但出手很重，懂等待、更懂暴力收割时机\n"
            "- 擅长读人读桌，抓心态弱点精准发难，慢打埋伏、反向压榨\n"
            "- 不迷信纯数学，也不盲目乱冲，只在优势位置疯狂放大底池\n"
            "- 极度重视资金管理，只打高胜率进攻，不做无谓消耗\n"
            "- 风格：老练、腹黑、城府深\n\n"
            "你正在和另外两位玩家讨论德州扑克。\n"
            "中文回复，每次100-150字，走老练凶狠路线，\n"
            "结合经验拆对手打法，给出高胜率进攻策略。"
        ),
    },
}

# 讨论话题列表
DISCUSSION_TOPICS = [
    "你认为德州扑克中最重要的技能是什么？",
    "面对大额ALL-IN时，你如何做决定？",
    "bluff（诈唬）在德州扑克中应该占多大比例？",
    "新手最常见的错误是什么？如何避免？",
]


# ══════════════════════════════════════════════════════════════
# LangGraph State 定义
# ══════════════════════════════════════════════════════════════

class ConversationState(TypedDict):
    """对话状态，贯穿整个图的执行过程"""
    # 完整对话历史（每条消息格式：{"agent": 名字, "content": 内容}）
    messages: Annotated[list[dict], operator.add]
    # 当前讨论的话题
    current_topic: str
    # 当前轮次（每个Agent发言一次为一轮）
    round_count: int
    # 当前发言的Agent名字
    current_speaker: str
    # 是否结束对话
    should_end: bool


# ══════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════

def build_context_prompt(state: ConversationState, agent_name: str) -> str:
    """将对话历史拼接为供 LLM 阅读的上下文"""
    topic = state["current_topic"]
    history = state["messages"]

    lines = ["【当前讨论话题】" + topic + "\n", "【对话历史】"]
    if not history:
        lines.append("（你是第一个发言的，请先表达你对这个话题的看法）")
    else:
        for msg in history[-8:]:  # 最多保留最近8条，防止 token 过多
            lines.append(msg["agent"] + "：" + msg["content"])

    lines.append("\n现在轮到你（" + agent_name + "）发言，请回应上面的对话：")
    return "\n".join(lines)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用通义千问 API"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.5,
        max_tokens=300,
        top_p=0.7,
    )
    return response.choices[0].message.content.strip()


def print_message(agent_name: str, content: str, description: str = ""):
    """格式化打印Agent发言"""
    desc_str = " [" + description + "]" if description else ""
    print("\n" + "─" * 60)
    print("  " + agent_name + desc_str)
    print("─" * 60)
    print(content)


# ══════════════════════════════════════════════════════════════
# LangGraph 节点函数（每个Agent对应一个节点）
# ══════════════════════════════════════════════════════════════

def agent_laoying(state: ConversationState) -> dict:
    """老鹰节点：冷静理性的数学派高手"""
    name = "老鹰"
    profile = AGENT_PROFILES[name]
    user_prompt = build_context_prompt(state, name)
    content = call_llm(profile["system_prompt"], user_prompt)
    print_message(name, content, profile["描述"])
    return {
        "messages": [{"agent": name, "content": content}],
        "current_speaker": name,
        "round_count": state["round_count"],
    }


def agent_xiaolajiao(state: ConversationState) -> dict:
    """小辣椒节点：激进凶猛的攻击型玩家"""
    name = "小辣椒"
    profile = AGENT_PROFILES[name]
    user_prompt = build_context_prompt(state, name)
    content = call_llm(profile["system_prompt"], user_prompt)
    print_message(name, content, profile["描述"])
    return {
        "messages": [{"agent": name, "content": content}],
        "current_speaker": name,
        "round_count": state["round_count"],
    }


def agent_laoqian(state: ConversationState) -> dict:
    """老钱节点：稳健老练的经验派赌神"""
    name = "老钱"
    profile = AGENT_PROFILES[name]
    user_prompt = build_context_prompt(state, name)
    content = call_llm(profile["system_prompt"], user_prompt)
    print_message(name, content, profile["描述"])
    # 老钱是每轮最后一个发言的，发完后更新轮次
    new_round = state["round_count"] + 1
    max_rounds = 3  # 每个话题最多3轮
    should_end = new_round > max_rounds
    return {
        "messages": [{"agent": name, "content": content}],
        "current_speaker": name,
        "round_count": new_round,
        "should_end": should_end,
    }


def check_end(state: ConversationState) -> Literal["continue", "end"]:
    """条件边：判断是否继续对话"""
    if state.get("should_end", False):
        return "end"
    return "continue"


# ══════════════════════════════════════════════════════════════
# 构建 LangGraph 图
# ══════════════════════════════════════════════════════════════

def build_graph():
    """构建多Agent对话图"""
    graph = StateGraph(ConversationState)

    # 添加节点
    graph.add_node("老鹰", agent_laoying)
    graph.add_node("小辣椒", agent_xiaolajiao)
    graph.add_node("老钱", agent_laoqian)

    # 设置入口节点
    graph.set_entry_point("老鹰")

    # 老鹰 -> 小辣椒（固定边）
    graph.add_edge("老鹰", "小辣椒")

    # 小辣椒 -> 老钱（固定边）
    graph.add_edge("小辣椒", "老钱")

    # 老钱 -> 判断是否结束（条件边）
    graph.add_conditional_edges(
        "老钱",
        check_end,
        {
            "continue": "老鹰",  # 继续下一轮，回到老鹰
            "end": END,          # 结束对话
        },
    )

    return graph.compile()


# ══════════════════════════════════════════════════════════════
# 主程序入口
# ══════════════════════════════════════════════════════════════

def run_demo():
    """运行德州扑克多Agent对话Demo"""
    print("=" * 60)
    print("  德州扑克多Agent经验交流 Demo")
    print("=" * 60)
    print("\n三位玩家：")
    for name, profile in AGENT_PROFILES.items():
        print("  - " + name + " : " + profile["描述"])

    app = build_graph()

    # 逐个话题展开讨论
    for topic_idx, topic in enumerate(DISCUSSION_TOPICS, 1):
        print("\n\n" + "#" * 60)
        print("# 话题 " + str(topic_idx) + "：" + topic)
        print("#" * 60)

        # 初始化该话题的状态
        initial_state: ConversationState = {
            "messages": [],
            "current_topic": topic,
            "round_count": 1,
            "current_speaker": "",
            "should_end": False,
        }

        # 执行图
        final_state = app.invoke(initial_state)

        total_rounds = final_state["round_count"] - 1
        print("\n\n[话题 " + str(topic_idx) + " 讨论结束，共 " + str(total_rounds) + " 轮]")

        # 询问是否继续下一个话题
        if topic_idx < len(DISCUSSION_TOPICS):
            user_input = input("\n按 Enter 继续下一个话题，输入 q 退出：").strip().lower()
            if user_input == "q":
                print("\n感谢参与！牌桌上见！")
                break

    print("\n" + "=" * 60)
    print("  所有话题讨论完毕！")
    print("  希望这场对话对你的德州扑克之路有所帮助！")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
