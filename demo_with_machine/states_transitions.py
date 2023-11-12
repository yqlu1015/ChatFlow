# Conversation stages for the `包裹卡获客` demo
from enum import Enum, auto


class StateEnum(Enum):
    """Custom-defined enumeration of states.

    """
    def _generate_next_value_(name, start, count, last_values):
        return count
    Dummy = auto()
    IceBreaking = auto()
    CustomerClassifying = auto()
    NewCustomer = auto()  # todo: knowledgeable agent
    NoIntention = auto()
    Coupon = auto()
    Lottery = auto()
    Waiting = auto()  # todo: implement state function
    Group = auto()  # todo: knowledgeable agent
    End = auto()


# descriptions of states as a part of the prompt.
STATE_DESC = {
    StateEnum.Dummy: "对话开始前的初始阶段",
    StateEnum.IceBreaking: "对话开始，询问客户是否是来领取优惠券的",
    StateEnum.CustomerClassifying: "如果客户想领优惠券，询问客户之前是否喝过纤维粉",
    StateEnum.NewCustomer: "如果客户想了解纤维粉，介绍纤维粉的好处",
    StateEnum.NoIntention: "如果客户不想了解纤维粉或不想领优惠券，向客户说明可以送任何感兴趣产品的优惠券",
    StateEnum.Coupon: "说明优惠券的实惠",
    StateEnum.Lottery: "向客户说明用优惠券下单后发订单截图可以参与抽奖，奖品有免单和红包，中奖率100%",
    StateEnum.Waiting: "委婉地提醒客户下单后发订单号",
    StateEnum.Group: "向客户说明宠粉福利群有丰富的活动",
    StateEnum.End: "向客户说明有其他问题可以随时联系，然后结束对话",

}

# states with callbacks.
STATE_CALLBACKS = {
    StateEnum.Waiting: {"name": StateEnum.Waiting, "timeout": 10, "on_timeout": "timeout_action"},
    # StateEnum.Group: {"name": StateEnum.Group, "timeout": 10, "on_timeout": "Group_to_End"},
}
# STATE_CALLBACKS = {}

# explicitly define next states
STATE_NEXT = {
    StateEnum.Dummy: [StateEnum.IceBreaking],
    StateEnum.IceBreaking: [StateEnum.NoIntention, StateEnum.CustomerClassifying],
    StateEnum.CustomerClassifying: [StateEnum.NewCustomer, StateEnum.Coupon],
    StateEnum.NewCustomer: [StateEnum.NewCustomer, StateEnum.Coupon],
    StateEnum.NoIntention: [StateEnum.NoIntention, StateEnum.Lottery],
    StateEnum.Coupon: [StateEnum.Coupon, StateEnum.Lottery],
    StateEnum.Lottery: [StateEnum.Waiting],
    StateEnum.Waiting: [StateEnum.Group],
    StateEnum.Group: [StateEnum.End],
    StateEnum.End: [StateEnum.Dummy],
}

TRANSITIONS = [
    {"trigger": f"Dummy_to_IceBreaking", "source": StateEnum.Dummy, "dest": StateEnum.IceBreaking, "after": "fixed_words_enter"},
    {"trigger": "IceBreaking_to_NoIntention", "source": StateEnum.IceBreaking, "dest": StateEnum.NoIntention},
    {"trigger": "IceBreaking_to_CustomerClassifying", "source": StateEnum.IceBreaking, "dest": StateEnum.CustomerClassifying},
    {"trigger": "CustomerClassifying_to_NewCustomer", "source": StateEnum.CustomerClassifying, "dest": StateEnum.NewCustomer, "after": "fixed_words_enter"},
    {"trigger": "CustomerClassifying_to_Coupon", "source": StateEnum.CustomerClassifying, "dest": StateEnum.Coupon},
    {"trigger": "NewCustomer_to_NewCustomer", "source": StateEnum.NewCustomer, "dest": StateEnum.NewCustomer},
    {"trigger": "NewCustomer_to_Coupon", "source": StateEnum.NewCustomer, "dest": StateEnum.Coupon},
    {"trigger": "NoIntention_to_NoIntention", "source": StateEnum.NoIntention, "dest": StateEnum.NoIntention},
    {"trigger": "NoIntention_to_Lottery", "source": StateEnum.NoIntention, "dest": StateEnum.Lottery, "after": "fixed_words_enter"},
    {"trigger": "Coupon_to_Coupon", "source": StateEnum.Coupon, "dest": StateEnum.Coupon},
    {"trigger": "Coupon_to_Lottery", "source": StateEnum.Coupon, "dest": StateEnum.Lottery, "after": "fixed_words_enter"},
    {"trigger": "Lottery_to_Waiting", "source": StateEnum.Lottery, "dest": StateEnum.Waiting},
    {"trigger": "Waiting_to_Group", "source": StateEnum.Waiting, "dest": StateEnum.Group, "after": "fixed_words_enter", "before": "fixed_words_leave"},
    {"trigger": "Group_to_End", "source": StateEnum.Group, "dest": StateEnum.End},
    {"trigger": "End_to_Dummy", "source": StateEnum.End, "dest": StateEnum.Dummy},
    {"trigger": "to_Dummy", "source": "*", "dest": StateEnum.Dummy},  # used for reset
]

# transition after which llm does not need to be called
NO_CHAT_AFTER_TRANSITION = [
    f"{StateEnum.Dummy.name}_to_{StateEnum.IceBreaking.name}",
    f"{StateEnum.CustomerClassifying.name}_to_{StateEnum.NewCustomer.name}",
]

# functions that will be called after the user input inside some state
STATE_FUNCTIONS = {
    StateEnum.Waiting: "check_order",
}

# fixed words to send when leaving some state.
FIXED_WORDS_ENTER = {
    StateEnum.IceBreaking: ["HI, 小姐妹, 是要领纤维粉立减100的优惠券吗", "可爱哈喽你好表情包, 是要领纤维粉立减100的优惠券吗"],
    StateEnum.NewCustomer: ["之前有没有喝过呀，需要我介绍一下吗☺"],
    StateEnum.Lottery: ["对啦，还有1个粉丝福利"],
    StateEnum.Group: ["对啦~我们还有宠粉福利群哦~"],
}

# fixed words to send when entering some state.
FIXED_WORDS_LEAVE = {
    StateEnum.Waiting: ["收到，感谢支持, 今天晚上晚点我会给你发抽奖链接哦~我们是统一发抽奖链接哒~希望你能抽到免单，嘿嘿"],
}

FIXED_WORD_REPEAT = "抱歉，我现在还不能理解你说的内容，可以再说一遍吗？"

TIMEOUT_WORDS = {
    StateEnum.Waiting: "下单成功后可以把订单截图发我哦",
}
