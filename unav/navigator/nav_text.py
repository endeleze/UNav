# nav_text.py

NAV_TEXT = {
    "start_in": {
        "en": "You are currently in {room} on {floor} of {building}, {place}.",
        "zh": "您目前在{place}{building}{floor}的{room}。",
        "th": "ขณะนี้คุณอยู่ที่ {room} ชั้น {floor} อาคาร {building} ใน {place}"
    },
    "start_nav": {
        "en": "Starting navigation.",
        "zh": "开始导航。",
        "th": "เริ่มการนำทาง"
    },
    "forward": {
        "en": "Forward {dist}",
        "zh": "直行{dist}",
        "th": "เดินตรงไป {dist}"
    },
    "forward_door": {
        "en": "Forward {dist} and go through a door in {door_dist}",
        "zh": "直行{dist}，{door_dist}后穿过一扇门",
        "th": "เดินตรงไป {dist} แล้วผ่านประตูที่ {door_dist}"
    },
    "transition_place": {
        "en": "You are approaching the transition to {place}.",
        "zh": "您正接近{place}。",
        "th": "คุณกำลังจะถึง {place}"
    },
    "proceed_to": {
        "en": "Proceed to {floor} of {building} in {place}.",
        "zh": "前往{place}{building}{floor}。",
        "th": "ไปที่ชั้น {floor} อาคาร {building} ใน {place}"
    },
    "approaching_stair": {
        "en": "You are approaching the staircase.",
        "zh": "您正接近楼梯。",
        "th": "คุณกำลังจะถึงบันได"
    },
    "approaching_elevator": {
        "en": "You are approaching the elevator.",
        "zh": "您正接近电梯。",
        "th": "คุณกำลังจะถึงลิฟต์"
    },
    "approaching_escalator": {
        "en": "You are approaching the escalator.",
        "zh": "您正接近扶梯。",
        "th": "คุณกำลังจะถึงบันไดเลื่อน"
    },
    "go_up_stair": {
        "en": "Go {direction} to {floor} of {building} via the staircase.",
        "zh": "通过楼梯{direction}到{building}{floor}。",
        "th": "ใช้บันไดไป {direction} ถึงชั้น {floor} อาคาร {building}"
    },
    "go_up_elevator": {
        "en": "Press the {direction} button to {floor} of {building} using the elevator.",
        "zh": "乘电梯{direction}到{building}{floor}。",
        "th": "ใช้ลิฟต์ไป {direction} ถึงชั้น {floor} อาคาร {building}"
    },
    "go_up_escalator": {
        "en": "Take the escalator {direction} to {floor} of {building}.",
        "zh": "乘扶梯{direction}到{building}{floor}。",
        "th": "ใช้บันไดเลื่อนไป {direction} ถึงชั้น {floor} อาคาร {building}"
    },
    "proceed_to_floor": {
        "en": "Proceed to {floor} of {building}.",
        "zh": "前往{building}{floor}。",
        "th": "ไปที่ชั้น {floor} อาคาร {building}"
    },
    "turn": {
        "en": "{qual} {direction} to {hour} o'clock",
        "zh": "{hour}点方向{qual}{direction}转弯",
        "th": "{qual} เลี้ยว{direction} ไปทาง {hour} นาฬิกา"
    },
    "u_turn": {
        "en": "Make a U-turn (6 o'clock)",
        "zh": "掉头（6点方向）",
        "th": "กลับรถ (6 นาฬิกา)"
    },
    "arrive": {
        "en": "{label} on {hour} o'clock {dir_word}",
        "zh": "{label}在{hour}点方向{dir_word}",
        "th": "{label} ที่ {hour} นาฬิกา {dir_word}"
    }
}

UNIT_TEXT = {
    "meter": {
        "en": "{v} meters",
        "zh": "{v}米",
        "th": "{v} เมตร"
    },
    "meter_1": {
        "en": "1 meter",
        "zh": "1米",
        "th": "1 เมตร"
    },
    "feet": {
        "en": "{v} feet",
        "zh": "{v}英尺",
        "th": "{v} ฟุต"
    },
    "feet_1": {
        "en": "1 foot",
        "zh": "1英尺",
        "th": "1 ฟุต"
    }
}


def nav_text(key: str, lang: str, **kwargs) -> str:
    """Get navigation instruction by key and language, with formatted params."""
    tpl = NAV_TEXT.get(key, {})
    text = tpl.get(lang) or tpl.get("en") or ""
    return text.format(**kwargs)

def unit_text(value: float, unit: str, lang: str) -> str:
    """Get a localized distance string for value/unit/language."""
    v_int = int(round(value))
    if unit == "meter":
        if v_int == 1:
            return UNIT_TEXT["meter_1"][lang]
        else:
            return UNIT_TEXT["meter"][lang].format(v=v_int)
    elif unit == "feet":
        if v_int == 1:
            return UNIT_TEXT["feet_1"][lang]
        else:
            return UNIT_TEXT["feet"][lang].format(v=v_int)
    else:
        raise ValueError("Unit must be 'meter' or 'feet'")
