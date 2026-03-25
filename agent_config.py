# -*- coding: utf-8 -*-
"""
Agent 配置和 LLM 调用模块
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise EnvironmentError("请在 .env 文件中设置 DASHSCOPE_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "qwen-plus"


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
        "描述": "阅人无数、稳中带凶的老牌侵略流高手",
        "system_prompt": (
            "你是'老钱'，二十年牌场老江湖，稳健只是伪装，骨子里稳中带狠。\n"
            "你的性格特点：\n"
            "- 语气慢悠悠，但出手很重，懂等待、更懂暴力收割时机\n"
            "- 擅长读人读桌，抓心态弱点精准发难，慢打埋伏、反向压榨\n"
            "- 不迷信纯数学，也不盲目乱冲，只在优势位置疯狂放大底池\n"
            "- 极度重视资金管理，只打高胜率进攻，不做无谓消耗\n"
            "- 风格：老练、腹黑、城府深，常用实战故事讲侵略逻辑\n\n"
            "你正在和另外两位玩家讨论德州扑克。\n"
            "中文回复，每次100-150字，走老练凶狠路线，\n"
            "结合经验拆对手打法，给出高胜率进攻策略。"
        ),
    },
    "火线": {
        "描述": "高压连打型进攻玩家",
        "system_prompt": (
            "你是'火线'，节奏极快、持续施压的进攻型德州扑克玩家。\n"
            "你的性格特点：\n"
            "- 偏好主动抢先手，用连续下注逼迫对手犯错\n"
            "- 面对犹豫玩家会放大下注尺寸，争取直接拿下底池\n"
            "- 不怕对抗，敢在关键节点做高压决策\n"
            "- 风格：短促、有压迫感、行动导向\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请严格按动作规范输出。"
        ),
    },
    "铁拳": {
        "描述": "重锤加注型进攻玩家",
        "system_prompt": (
            "你是'铁拳'，偏好重注压制的进攻型德州扑克玩家。\n"
            "你的性格特点：\n"
            "- 倾向用大额 raise 争夺主动权\n"
            "- 对弱势信号反应敏锐，喜欢直接施加强压\n"
            "- 在优势牌力下愿意快速做大底池\n"
            "- 风格：强硬、直接、侵略性高\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请严格按动作规范输出。"
        ),
    },
}


def call_llm_for_action(agent_name: str, hole_cards: list, game_context: str) -> tuple:
    """
    调用 LLM 让 Agent 决定行动
    返回 (行动类型, 金额, 说话内容)
    """
    profile = AGENT_PROFILES[agent_name]
    cards_str = "、".join([str(c) for c in hole_cards])
    stage_tips = {
        "preflop": "翻牌前（preflop）：尚未看到公共牌，请务必谨慎，不要轻易全押，拿到很强的手牌全压了反而让对手害怕不敢跟了。也不要轻易弃牌，可以赌后面的公共牌，也可以吓唬影响对手的判断",
        "flop":    "翻牌（flop）：已有3张公共牌，可以结合公共牌判断自己的成牌潜力，再决定下注力度。",
        "turn":    "转牌（turn）：已有4张公共牌，局势更加明朗，请结合当前牌面和底池赔率做出理性决策。",
        "river":   "河牌（river）：5张公共牌已全部亮出，这是最后一轮下注，请综合全局判断是否价值下注、诈唬或弃牌。",
    }
    # 从 game_context 中提取当前街道
    current_stage = "preflop"
    for s in ["preflop", "flop", "turn", "river"]:
        if s in game_context:
            current_stage = s
            break
    stage_hint = stage_tips.get(current_stage, "")

    user_prompt = (
    f"你的手牌：{cards_str}\n\n"
    f"【当前阶段提示】{stage_hint}\n\n"
    f"游戏情况：\n{game_context}\n\n"
    "请根据上面的实时信息与规则作出决策。\n"
    "【硬性输出规范】\n"
    "你必须只输出一行，格式严格为：action amount|speech\n"
    "- action 只能是：fold / check / call / raise\n"
    "- amount 必须是整数；fold/check 时 amount 必须为 0\n"
    "- 必须遵守游戏情况里的规则限制（前注、最小加注到、每街最大加注次数）\n"
    "- 如果规则提示本街不能raise，就不要输出raise\n"
    "- 如果要raise，amount 必须让你的总下注至少达到‘最小加注到’\n"
    "- raise/call 的 amount 绝对不能超过你当前筹码数（游戏情况中'你当前筹码'一栏有标注）\n"
    "- 锦标赛不是最后一手：不要把每手都当生死局，非强牌请控制全押频率\n"
    "- 不要输出多余解释、不要换行、不要加引号\n"
    "示例：raise 120|我在按钮位用范围优势持续施压。"
)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": profile["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.85,
        max_tokens=200,
    )
    
    content = response.choices[0].message.content.strip()
    
    try:
        if "|" in content:
            action_part, speech = content.split("|", 1)
            speech = speech.strip()
        else:
            action_part = content
            speech = ""
        
        parts = action_part.strip().split()
        action = parts[0].lower() if parts else "check"
        amount = int(parts[1]) if len(parts) > 1 else 0
        
        if action in ["fold", "弃牌"]:
            action = "fold"
        elif action in ["check", "过牌"]:
            action = "check"
        elif action in ["call", "跟注"]:
            action = "call"
        elif action in ["raise", "加注"]:
            action = "raise"
        else:
            action = "check"
        
        return (action, amount, speech)
    except:
        return ("check", 0, "我过牌")
