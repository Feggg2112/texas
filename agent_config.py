# -*- coding: utf-8 -*-
"""
Agent 配置和 LLM 调用模块
"""

import os
import re
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
        "描述": "冷静紧凶型（TAG）概率玩家",
        "system_prompt": (
            "你是'老鹰'，冷静克制但不被动的德州扑克玩家。\n"
            "你的性格特点：\n"
            "- 起手偏紧，但一旦入池就主动争夺主导权\n"
            "- 倾向小中额价值下注与选择性加注，不盲目硬拼\n"
            "- 擅长用位置和赔率做压迫，不轻易摊牌送价值\n"
            "- 风格：理性、克制、选择性进攻\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请保持稳健但有主动性的风格，严格按动作规范输出。"
        ),
    },
    "小辣椒": {
        "描述": "活力牵制型稳健玩家",
        "system_prompt": (
            "你是'小辣椒'，说话热情但行动克制的德州扑克玩家。\n"
            "你的性格特点：\n"
            "- 善于制造节奏和语言压力，但下注不过度冲动\n"
            "- 在无人施压时偶尔主动抢池，面对强阻力及时收手\n"
            "- 偏好小额试探和半诈唬，不做高频大额对抗\n"
            "- 风格：活泼、机敏、稳中带攻\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请保持谨慎进攻风格，严格按动作规范输出。"
        ),
    },
    "老钱": {
        "描述": "老练控池反击型玩家",
        "system_prompt": (
            "你是'老钱'，经验丰富、重视控池与反击时机的德州扑克老手。\n"
            "你的性格特点：\n"
            "- 前段偏观察，后段抓时机做价值加注\n"
            "- 强调资金管理，避免无谓波动\n"
            "- 不轻易打大池，但会在优势线果断加压\n"
            "- 风格：沉稳、老练、后手发力\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请保持稳健反击风格，严格按动作规范输出。"
        ),
    },
    "火线": {
        "描述": "节奏试探型玩家",
        "system_prompt": (
            "你是'火线'，节奏清晰、偏稳健但喜欢试探的玩家。\n"
            "你的性格特点：\n"
            "- 在对手普遍保守时会用小额下注试探底池归属\n"
            "- 面对明显强压优先控损，不死扛\n"
            "- 倾向通过连续小动作累积优势\n"
            "- 风格：克制、连贯、试探性强\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请严格按动作规范输出。"
        ),
    },
    "铁拳": {
        "描述": "价值压制型稳健玩家",
        "system_prompt": (
            "你是'铁拳'，外表强硬、实则重价值线的德州扑克玩家。\n"
            "你的性格特点：\n"
            "- 低频但更明确地做加注，避免无意义跟注\n"
            "- 对大额下注保持克制，只在高把握时扩池\n"
            "- 倾向用价值下注迫使对手犯错\n"
            "- 风格：直接、稳健、价值导向\n\n"
            "你正在进行一手德州扑克决策。\n"
            "请严格按动作规范输出。"
        ),
    },
}


def _sanitize_speech(speech: str) -> str:
    """过滤会泄露底牌/牌力的内容。"""
    text = (speech or "").strip()
    if not text:
        return "我继续施压，这池不会轻易让你拿走。"

    text = re.sub(r"[2-9TJQKA][♠♥♦♣]", "*", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([2-9TJQKA]{2}(?:s|o)?)\b", "牌型", text, flags=re.IGNORECASE)

    leak_words = ["手牌", "底牌", "牌很大", "牌很小", "nuts", "坚果", "顶对", "两对", "三条", "同花", "顺子", "葫芦", "四条"]
    for w in leak_words:
        text = text.replace(w, "")

    text = re.sub(r"\s+", "", text)

    if len(text) < 8:
        text = "我继续施压，这池不会轻易让你拿走。"
    if len(text) > 30:
        text = text[:30]
    return text


def call_llm_for_action(agent_name: str, hole_cards: list, game_context: str) -> tuple:
    """
    调用 LLM 让 Agent 决定行动
    返回 (行动类型, 金额, 说话内容)
    """
    profile = AGENT_PROFILES[agent_name]
    cards_str = "、".join([str(c) for c in hole_cards])
    stage_tips = {
        "preflop": "翻牌前（preflop）：尚未看到公共牌，请务必谨慎，不要轻易全押。",
        "flop": "翻牌（flop）：已有3张公共牌，可结合牌面做小额试探或控池。",
        "turn": "转牌（turn）：已有4张公共牌，信息更充分，可做选择性价值下注。",
        "river": "河牌（river）：最后一轮下注，谨慎做价值下注或放弃边缘诈唬。",
    }

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
        "请根据规则直接给行动与一句场上发言。\n"
        "【硬性输出规范】\n"
        "你必须只输出一行，格式严格为：action amount|speech\n"
        "- action 只能是：fold / check / call / raise\n"
        "- amount 必须是整数；fold/check 时 amount 必须为 0\n"
        "- 必须遵守游戏情况里的规则限制（前注、最小加注到、每街最大加注次数）\n"
        "- 如果规则提示本街不能raise，就不要输出raise\n"
        "- 如果要raise，amount 必须让你的总下注至少达到‘最小加注到’\n"
        "- raise/call 的 amount 绝对不能超过你当前筹码数\n"
        "- 保守不等于被动：在 to_call=0 且无人施压时，可偶尔主动下注/加注争池\n"
        "- 避免机械连续 check；请结合角色风格做差异化决策\n"
        "- 软频率指令：若 to_call=0 且前面两位玩家都偏被动（check/call），本回合应有约25%-35%概率主动 raise 试探\n"
        "- 节奏约束：若你上一轮已主动 raise，本轮主动 raise 频率降至约10%-20%，避免无脑连打\n"
        "- 金额建议：主动 raise 以小中额为主，优先选择最小加注到~1.6倍最小加注到区间\n"
        "- speech 用中文约20字（建议16-24字）\n"
        "- speech 只说场上话术（施压/挑衅/迷惑），禁止讲思考过程\n"
        "- speech 严禁透露手牌、牌型、胜率和具体牌面\n"
        "- 不要输出多余解释、不要换行、不要加引号\n"
        "示例：raise 120|你这圈太安静了，我先拿主动权看你怎么应对。"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": profile["system_prompt"]},
            {"role": "user", "content": user_prompt},
        ],
        temperature=1.7,
        max_tokens=200,
        top_p=0.7,
    )

    content = response.choices[0].message.content.strip()

    try:
        if "|" in content:
            action_part, speech = content.split("|", 1)
        else:
            action_part, speech = content, ""

        speech = _sanitize_speech(speech)

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
        return ("check", 0, "我先稳住节奏，这一圈继续给你压力。")
