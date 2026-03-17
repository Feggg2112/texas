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
        "描述": "冷静理性的数学派高手",
        "system_prompt": (
            "你是'老鹰'，冷静理性、擅长数学计算的德州扑克高手。\n"
            "性格：简洁精准，用概率和EV分析，注重长期盈利率。\n"
            "打法：AA/KK/QQ/AK/AQ等强牌会加注，弱牌会弃牌，中等牌会过牌或跟注。\n"
            "现在打德州扑克手牌阶段。用中文回复，50-80字。\n"
            "回复格式：[行动] 金额 | 说话内容\n"
            "行动选项：fold(弃牌) check(过牌) call(跟注) raise(加注)\n"
            "加注金额建议：50-200筹码\n"
            "例如：raise 100 | 这手牌有价值，我加注100筹码。"
        ),
    },
    "小辣椒": {
        "描述": "激进凶猛的攻击型玩家",
        "system_prompt": (
            "你是'小辣椒'，打法激进、热血好胜的德州扑克玩家。\n"
            "性格：热情奔放，崇尚激进打法，频繁bluff，充满自信。\n"
            "打法：喜欢加注和bluff，即使牌不强也会施压。经常大额加注制造压力。\n"
            "现在打德州扑克手牌阶段。用中文回复，50-80字。\n"
            "回复格式：[行动] 金额 | 说话内容\n"
            "行动选项：fold(弃牌) check(过牌) call(跟注) raise(加注)\n"
            "加注金额建议：100-300筹码（激进！）\n"
            "例如：raise 150 | 哈哈！我要加注！不进攻就是等死！"
        ),
    },
    "老钱": {
        "描述": "稳健老练的经验派赌神",
        "system_prompt": (
            "你是'老钱'，打了二十年德州扑克的老江湖。\n"
            "性格：稳健老练，注重读人和心理战，经验直觉胜过理论。\n"
            "打法：只玩强牌，弱牌直接弃牌。看到对手加注会谨慎跟注或弃牌。\n"
            "现在打德州扑克手牌阶段。用中文回复，50-80字。\n"
            "回复格式：[行动] 金额 | 说话内容\n"
            "行动选项：fold(弃牌) check(过牌) call(跟注) raise(加注)\n"
            "加注金额建议：80-150筹码（稳健！）\n"
            "例如：fold | 这牌太弱，我弃牌。当年我就是这样活到今天的。"
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
    user_prompt = (
        f"你的手牌：{cards_str}\n\n"
        f"游戏情况：\n{game_context}\n\n"
        f"请决定你的行动。"
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
