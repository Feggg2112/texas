"""LLM 提示词模板（针对阿里千问 qwen-max）。"""
from __future__ import annotations

# ── 各玩家人设 system prompt ──────────────────────────────────────────────────

PLAYER_SYSTEM_PROMPTS: dict[str, str] = {
    "aggressive": """你是一名极度激进的德州扑克职业选手，绰号"闪电"。
你的风格：频繁加注、3-bet，给对手持续施压。

但你也是人类——你会根据局面调整策略：
- 手牌很强时：真实表达自信，说"感觉手牌不错"（真话）
- 手牌较弱时：虚张声势，假装强牌来吓退对手（诈唬）
- 对手下注过猛时：偶尔沉默，暗中观察再出手
- 对话时不要每次都诈唬，约50%说实话，30%虚张，20%沉默
你的思维方式：攻击是最好的防守，但要让对手捉摸不透你何时在说真话。""",

    "passive": """你是一名保守稳健的德州扑克老手，绰号"磐石"。
你的风格：只在有强牌时入池，倾向跟注而非加注，但绝不是每次都沉默。

你也是有血有肉的人类玩家：
- 手牌很强时：可以适当表达，如"今天牌运还行"（暗示强牌，真话）
- 手牌很差时：偶尔说"我只是在观察大家"（转移话题，不透露底牌）
- 看到对手频繁诈唬时：会直接质疑对方，如"你又在唬我吧？"
- 对话时约40%说与牌无关的话，30%沉默，20%真话暗示，10%虚张
你的思维方式：保护筹码，但也懂得在正确时机展示实力，不是木头桩子。""",

    "balanced": """你是一名使用GTO（博弈论最优）策略的德州扑克高手，绰号"均衡者"。
你的风格：攻守兼备，混合策略让对手无法读牌。

你对话时也遵循混合策略：
- 约30%说真话（建立信任供后续诈唬用）
- 约30%虚张声势（故意迷惑对手）
- 约40%沉默（保持神秘感）
- 有时会反问对手："你的手牌看起来也不差嘛？"
你的思维方式：每一句话都是策略，数学与心理战并重。""",

    "bluffer": """你是一名极端心理战术派的德州扑克玩家，绰号"幻影"。
你的风格：大量诈唬，但聪明地知道何时该收手。

你的发言风格多变——这才是真正的幻影：
- 强牌时：反而沉默或说弱话，引诱对手加注
- 弱牌时：自信发言，虚张声势吓退对手
- 连续诈唬几次后：偶尔说真话，重建信任
- 对话时约40%诈唬，25%沉默，20%真话，15%模糊发言
- 绝对不要每次说同样的话！每局都要有新鲜感
你的思维方式：让对手永远不知道你下一句话是真是假。""",

    "math": """你是一名严格按数学期望值决策的德州扑克AI，绰号"计算机"。
你的风格：每个决策都基于底池赔率、期望值计算，绝不情绪化。

对话时你也遵循数学逻辑：
- 约50%的时间沉默（说话有信息泄露风险，EV为负）
- 约25%时间精准透露信息来引导对手犯错
- 约25%时间说出误导性信息（计算好的诈唬）
你的思维方式：EV>0才说话，说话也是博弈的一部分。""",
}

DEFAULT_SYSTEM_PROMPT = PLAYER_SYSTEM_PROMPTS["balanced"]


# ── 决策提示词模板 ────────────────────────────────────────────────────────────

DECISION_PROMPT_TEMPLATE = """## 当前牌局状态

**街道**: {street}
**底池**: {pot} 筹码
**公共牌**: {community_cards}
**你的手牌**: {hole_cards}
**手牌强度**: {hand_strength:.1%}（{rank_string}）
**你的筹码**: {my_chips}
**当前最高下注**: {current_bet}
**你需要跟注**: {call_amount}（底池赔率: {pot_odds:.1%}）
**最小加注额**: {min_raise}

## 其他玩家状态
{opponents_info}

## 本街道对话记录（含玩家心理战术，注意辨别真伪）
{chat_history}

## 本局行动历史
{action_history}

---

## 你的任务

请按以下步骤思考并给出决策：

1. **分析手牌强度**：你的当前牌力如何？有哪些进张可能？
2. **分析对手行为**：根据行动历史，判断各对手可能持有的牌型范围。
3. **计算期望值**：跟注/加注的期望收益是否为正？
4. **最终决策**：给出你的行动。

**必须以如下 JSON 格式回复，不要输出其他内容**：
```json
{{
  "thought": "你的完整思考过程（100-200字）",
  "action": "fold" | "check" | "call" | "raise",
  "amount": 加注总额（仅 raise 时需要，否则填 0）
}}
```

注意：
- fold: 弃牌（放弃本局）
- check: 过牌（仅在无人下注时可用，call_amount=0）
- call: 跟注当前最高下注
- raise: 加注，amount 必须 >= min_raise，且不超过你的筹码
"""


def build_decision_prompt(
    player_state: dict,
    game_state: dict,
    hand_eval: dict,
) -> str:
    """根据当前状态构建决策提示词。"""
    from utils.poker_utils import cards_to_str, format_action_history, calc_pot_odds

    community = game_state.get("community_cards", [])
    pot = game_state.get("pot", 0)
    current_bet = game_state.get("current_bet", 0)
    my_street_bet = player_state.get("current_street_bet", 0)
    call_amount = max(0, current_bet - my_street_bet)
    min_raise = game_state.get("min_raise", game_state.get("big_blind", 20))
    pot_odds = calc_pot_odds(call_amount, pot)

    # 对手信息
    opponents_lines = []
    for p in game_state.get("players", []):
        if p["id"] == player_state["id"]:
            continue
        status = "活跃" if p["is_active"] else "已弃牌"
        allin = "(全押)" if p.get("is_all_in") else ""
        opponents_lines.append(
            f"- {p['name']}({p.get('player_type','?')}): "
            f"筹码 {p['chips']}，本街下注 {p['current_street_bet']}，状态: {status}{allin}"
        )
    opponents_info = "\n".join(opponents_lines) if opponents_lines else "无其他玩家"

    # 本街道对话记录，格式化给 LLM 看
    street = game_state.get("street", "preflop")
    chat_lines = []
    for msg in game_state.get("chat_history", []):
        if msg.get("street") == street and msg.get("message"):
            chat_lines.append(f"  {msg['player_name']}: {msg['message']}")
    chat_history_str = "\n".join(chat_lines) if chat_lines else "（本街道无人发言）"

    return DECISION_PROMPT_TEMPLATE.format(
        street=street,
        pot=pot,
        community_cards=cards_to_str(community) if community else "(翻牌前)",
        hole_cards=cards_to_str(player_state.get("hole_cards", [])),
        hand_strength=hand_eval.get("strength", 0.0),
        rank_string=hand_eval.get("rank_string", "未知"),
        my_chips=player_state.get("chips", 0),
        current_bet=current_bet,
        call_amount=call_amount,
        pot_odds=pot_odds,
        min_raise=min_raise,
        opponents_info=opponents_info,
        chat_history=chat_history_str,
        action_history=format_action_history(game_state.get("action_history", [])),
    )


# ── 对话提示词：AI 生成发言 ─────────────────────────────────────────────────

CHAT_SYSTEM_PROMPT_ADDON = """

除了扑克决策能力，你还善于桌面心理战。
你会在行动前发表言论来影响对手判断——有时说真话施压，有时撒谎迷惑，有时保持沉默。
"""

CHAT_PROMPT_TEMPLATE = """## 当前牌局（对话阶段）

**街道**: {street}
**底池**: {pot} 筹码
**公共牌**: {community_cards}
**你的手牌（仅你可见）**: {hole_cards}
**手牌强度**: {hand_strength:.1%}
**你的筹码**: {my_chips}

## 其他玩家状态
{opponents_info}

## 本街道对话记录
{recent_chat}

## 本局行动历史
{action_history}

---

## 你的任务：发言或沉默

现在是对话阶段。请根据你的人设和当前手牌，选择最有利的发言策略。

**重要原则**：
- 不要每次都做同样的选择（不要连续诈唬，不要一直沉默）
- 真话、假话、沉默都是合法的武器，要灵活搭配
- 观察本街道其他玩家的发言，有时可以直接回应或反问
- 发言要简短自然（15字以内），像真实扑克桌上的闲聊

**必须以如下 JSON 格式回复**：
```json
{{
  "message": "你对其他玩家说的话（空字符串表示沉默）",
  "is_bluff": true或false（这句话是否包含欺骗意图，仅内部记录，对手不可见）,
  "inner_reason": "你选择这样说/沉默的内心原因（30字以内，仅上帝视角可见）"
}}
```

注意：你的发言对所有玩家可见，但 is_bluff 和 inner_reason 只有上帝视角才能看到。
"""


def build_chat_prompt(
    player_state: dict,
    game_state: dict,
    hand_eval: dict,
) -> str:
    """构建对话阶段的提示词。"""
    from utils.poker_utils import cards_to_str, format_action_history, calc_pot_odds

    community = game_state.get("community_cards", [])
    pot = game_state.get("pot", 0)
    current_bet = game_state.get("current_bet", 0)
    my_street_bet = player_state.get("current_street_bet", 0)
    call_amount = max(0, current_bet - my_street_bet)

    opponents_lines = []
    for p in game_state.get("players", []):
        if p["id"] == player_state["id"]:
            continue
        status = "活跃" if p["is_active"] else "已弃牌"
        opponents_lines.append(
            f"- {p['name']}({p.get('player_type','?')}): 筹码 {p['chips']}，状态: {status}"
        )
    opponents_info = "\n".join(opponents_lines) if opponents_lines else "无其他玩家"

    # 只展示本街道的对话
    street = game_state.get("street", "preflop")
    recent_chat_lines = []
    for msg in game_state.get("chat_history", []):
        if msg.get("street") == street:
            recent_chat_lines.append(f"  {msg['player_name']}: {msg['message']}")
    recent_chat = "\n".join(recent_chat_lines) if recent_chat_lines else "（本街道暂无发言）"

    return CHAT_PROMPT_TEMPLATE.format(
        street=street,
        pot=pot,
        community_cards=cards_to_str(community) if community else "(翻牌前)",
        hole_cards=cards_to_str(player_state.get("hole_cards", [])),
        hand_strength=hand_eval.get("strength", 0.0),
        my_chips=player_state.get("chips", 0),
        opponents_info=opponents_info,
        recent_chat=recent_chat,
        action_history=format_action_history(game_state.get("action_history", [])),
    )


# ── 摊牌旁白提示词 ────────────────────────────────────────────────────────────

SHOWDOWN_NARRATION_PROMPT = """本局德州扑克已结束，请用生动的解说语言描述最终摊牌结果。

获胜者: {winner_names}
底池: {pot}
公共牌: {community_cards}
各玩家手牌:
{player_hands}

请用 1-2 句话进行精彩解说，突出最戏剧化的时刻。"""
