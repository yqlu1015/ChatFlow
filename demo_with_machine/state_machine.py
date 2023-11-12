import random
from typing import Callable, Optional, Union, Any
from copy import deepcopy

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.agents import AgentExecutor
from langchain.chains.base import Chain
from langchain.chat_models import ChatLiteLLM
from langchain.llms.base import create_base_retry_decorator
from transitions import Machine, State
from transitions.extensions.states import add_state_features, Tags, Timeout
from transitions.extensions.asyncio import AsyncTimeout, AsyncMachine

from litellm import acompletion
from pydantic import Field

from demo_with_machine.chains import SalesConversationChain, StageAnalyzerChain, KnowledgeableAgent, UseToolAgent
from utils.logger import time_logger
from demo_with_machine.tools import get_tools, setup_knowledge_base
from demo_with_machine.states_transitions import (StateEnum, STATE_CALLBACKS, TRANSITIONS, FIXED_WORD_REPEAT,
                                                  TIMEOUT_WORDS, FIXED_WORDS_LEAVE, FIXED_WORDS_ENTER, STATE_NEXT,
                                                  STATE_DESC, STATE_FUNCTIONS, NO_CHAT_AFTER_TRANSITION)
from demo_with_machine.utils import dict2str


@add_state_features(Tags, Timeout)
class CustomMachine(Machine):
    pass


class AgentMachine(CustomMachine):
    """LLM agent with a state machine.

    """
    chat_flag: bool = True  # whether to use llm to chat or not
    initial_state: StateEnum = StateEnum.Dummy

    conversation_history: list[str] = []
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    sales_knowledgeable_conversation_agent: Optional[KnowledgeableAgent] = Field(...)
    use_tool_agent: Optional[UseToolAgent] = Field(...)

    model_name: str = "gpt-3.5-turbo-0613"

    use_tools: bool = False
    use_knowledge: bool = False
    salesperson_name: str = "Ted Lasso"
    salesperson_role: str = "Business Development Representative"
    company_name: str = "Sleep Haven"
    company_business: str = "Sleep Haven is a premium mattress company that provides customers with the most comfortable and supportive sleeping experience possible. We offer a range of high-quality mattresses, pillows, and bedding accessories that are designed to meet the unique needs of our customers."
    company_values: str = "Our mission at Sleep Haven is to help people achieve a better night's sleep by providing them with the best possible sleep solutions. We believe that quality sleep is essential to overall health and well-being, and we are committed to helping our customers achieve optimal sleep by offering exceptional products and customer service."
    conversation_purpose: str = "find out whether they are looking to achieve better sleep via buying a premier mattress."
    conversation_type: str = "call"
    human_prefix = "客户"

    def __init__(self, llm: ChatLiteLLM, initial_state: StateEnum = StateEnum.Dummy, verbose: bool = False, **kwargs):
        states = self.get_all_states()
        transitions = TRANSITIONS
        super().__init__(self, states=states, initial=initial_state, auto_transitions=True)
        self.add_transitions(transitions)
        # called when state analyzer fails
        self.add_transition(trigger="repeat", source=[state for state in StateEnum], dest="=", after="retry")
        self.chat_flag = True
        self.initialize(llm, verbose, **kwargs)
        # machine = agent.set_machine(initial_state=initial_state)
        print("successfully initialize a machine!")

    # ======== agent-related methods below ========

    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        # self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []
        self.reset()

    @time_logger
    def transition_state(self):
        """Transition to the next state by a combination of llm-based state analyzer and the state machine.

        """
        state_function = self._get_state_function()
        if self.use_tools and state_function != "":
            # call the state function before determining the next state
            # todo
            raise NotImplementedError("Function call before the transition is not implemented yet.")
        else:
            # determine the next state by llm and state machine
            next_states_dict = self.get_next_states_desc(self.state)
            if len(next_states_dict) == 0:
                raise RuntimeError(f"next states not defined for {self.state.name}.")
            if len(next_states_dict) == 1:
                next_state_name = list(next_states_dict.keys())[0].name
                print(f"Deterministic Next State: {next_state_name}")
            else:
                predicted_state_id = self.stage_analyzer_chain.run(
                    conversation_history="\n".join(self.conversation_history).rstrip("\n"),
                    current_state=self._get_state_desc(),
                    next_states=dict2str(next_states_dict),
                )
                print(f"Raw Predicted State Value: {predicted_state_id}")
                predicted_state = self._get_state_from_value(str(predicted_state_id).strip())
                if predicted_state is not None:
                    next_state_name = predicted_state.name
                    print(f"Predicted State: {next_state_name}")
                else:
                    self.repeat()
                    return
            current_state_name = self.state.name
            transition_name = f"{current_state_name}_to_{next_state_name}"
            transition_func = getattr(self, transition_name)
            transition_func()
            self._determine_llm_chat(transition_name)

    @time_logger
    def step(self, stream: bool = False):
        """
        Args:
            stream (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not stream:
            self._call()

    def human_step(self, human_input):
        # process human input
        human_input = self.human_prefix + ": " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)
        # print(human_input.replace("<END_OF_TURN>", ""))

    def _call(self):
        """Run one step of the sales agent.

        """
        # transition to the next state after the human input
        self.transition_state()

        if not self.chat_flag:
            # skip llm chat if not needed
            self.chat_flag = True
            return
        if self.use_knowledge:
            ai_message = self.sales_knowledgeable_conversation_agent.run(
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                # conversation_stages=self.all_state_desc,
                current_stage=self._get_state_desc()
            )
        else:
            ai_message = self.sales_conversation_utterance_chain.run(
                conversation_history="\n".join(self.conversation_history),
                salesperson_name=self.salesperson_name,
                salesperson_role=self.salesperson_role,
                company_name=self.company_name,
                company_business=self.company_business,
                conversation_purpose=self.conversation_purpose,
                conversation_type=self.conversation_type,
                # conversation_stages=self.all_state_desc,
                current_stage=self._get_state_desc()
            )

        # Add agent's response to conversation history
        ai_message = ai_message.split(self.salesperson_name + ": ")[0].split(self.human_prefix + ": ")[0].strip()
        self._fixed_words_chat(ai_message)

    def initialize(self, llm: ChatLiteLLM, verbose: bool = False, **kwargs):
        """Initialize the llm part."""
        # set up stage analyzer
        self.stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        # set up conversation chain
        if "custom_prompt" in kwargs.keys() and kwargs["custom_prompt"] != "":
            custom_prompt = deepcopy(kwargs["custom_prompt"])
            self.sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose,
                                                                                      custom_prompt=custom_prompt)
        else:
            self.sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose)
        kwargs.pop("custom_prompt", None)

        if "use_tools" in kwargs.keys() and (kwargs["use_tools"] == "True" or kwargs["use_tools"] is True):
            # set up agent with tools
            self.use_tools = True
            # todo
            self.use_tool_agent = None

        if "use_knowledge" in kwargs.keys() and (kwargs["use_knowledge"] == "True" or kwargs["use_knowledge"] is True):
            # set up agent with knowledge base
            self.use_knowledge = True
            if "knowledge_file" in kwargs.keys() and kwargs["knowledge_file"] != "":
                knowledge_file = kwargs["knowledge_file"]
                knowledge_base = setup_knowledge_base(knowledge_file)
            else:
                knowledge_base = setup_knowledge_base()

            tools = get_tools(knowledge_base)
            self.sales_knowledgeable_conversation_agent = KnowledgeableAgent.from_llm(
                llm=llm,
                tools=tools,
                ai_prefix=kwargs["salesperson_name"],
                verbose=verbose)
        kwargs.pop("knowledge_file", None)

        self.model_name = llm.model
        for key, value in kwargs.items():
            setattr(self, key, value)

    # ======== machine-related methods below ========

    def reset(self):
        """Reset state machine to the initial state [dummy]

        """
        func = getattr(self, f"to_{StateEnum.Dummy.name}")
        func()

    def retry(self):
        # print(f"revisit {self.state.name}")
        self._fixed_words_chat(FIXED_WORD_REPEAT)

    def timeout_action(self):
        """Chat using some fixed word when state timeout.

        """
        if self.state not in TIMEOUT_WORDS.keys():
            return

        words = TIMEOUT_WORDS.get(self.state)
        self._fixed_words_chat(words)

    def fixed_words_enter(self):
        """Chat using some fixed words when entering some state, instead of the llm.

        """
        words = self._get_fixed_words(enter=True)
        if words == "":
            return
        # print(self.salesperson_name + ": " + words)
        self._fixed_words_chat(words)

    def fixed_words_leave(self):
        """Chat using some fixed words when leaving some state, instead of the llm.

        """
        words = self._get_fixed_words(enter=False)
        if words == "":
            return
        # print(self.salesperson_name + ": " + words)
        self._fixed_words_chat(words)

    def _get_state_from_value(self, value: str) -> Optional[StateEnum]:
        """Get state from a value.
        Return None if the value is not valid.

        Args:
            value: value of a StateEnum.

        Returns:

        """
        possible_states = list(self.get_next_states_desc(self.state).keys())
        # todo: value start at 1
        if value.isdigit() and int(value) in range(len(self.states)) and StateEnum(
                int(value)) in possible_states:
            return StateEnum(int(value))
        else:
            # TODO: real person customer service intervened
            print(f"state analyzer fails at {self.state.name}")
            return None

    def _get_state_function(self) -> str:
        """Get the function name corresponding to the current state.

        """
        current_state = self.state
        return STATE_FUNCTIONS.get(current_state, "")

    def _get_state_desc(self) -> str:
        """Get current state description in the following format:
            1: 对话开始，询问客户是否是来领取优惠券的

        """
        current_state = self.state
        return str(current_state.value) + ": " + STATE_DESC.get(current_state, "")

    def _determine_llm_chat(self, transition_name: str):
        """ determine whether to use llm chat or not in the next step

        """
        if transition_name in NO_CHAT_AFTER_TRANSITION:
            self.chat_flag = False

    def _get_fixed_words(self, enter: bool) -> str:
        """Randomly select some words from the fixed words dictionary.

        Args:
            enter: true if selecting from fixed_words_enter, false if from fixed_words_leave

        Returns:
            The selected words, an empty str if there is no word corresponding to the current state.

        """
        if enter:
            if self.state in FIXED_WORDS_ENTER:
                return random.choice(FIXED_WORDS_ENTER.get(self.state, [""]))
            raise RuntimeError(f"No fixed words set for entering the current state: {self.state.name}.")
        else:
            if self.state in FIXED_WORDS_LEAVE:
                return random.choice(FIXED_WORDS_LEAVE.get(self.state, [""]))
            raise RuntimeError(f"No fixed words set for leaving the current state: {self.state.name}.")

    def _fixed_words_chat(self, words: str):
        """Chat using some fixed words.

        Args:
            words: input words.

        """
        ai_message = self.salesperson_name + ": " + words + " <END_OF_TURN>"
        self.conversation_history.append(ai_message)
        print(ai_message.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", ""))

    @staticmethod
    def get_all_states() -> list[Union[State, dict]]:
        """Get the list of states.

        Returns:

        """
        return [
            State(state) if state not in STATE_CALLBACKS
            else STATE_CALLBACKS[state]
            for state in StateEnum
        ]

    @staticmethod
    def get_next_states_desc(current_state: StateEnum) -> dict:
        """Return all possible next states and their descriptions.

        Args:
            current_state: current state

        Returns:

        """
        if current_state in STATE_NEXT:
            next_states = STATE_NEXT[current_state]
            next_states_desc = {}
            for s in next_states:
                next_states_desc[s] = STATE_DESC.get(s, "")
            return next_states_desc
        return {}

    @staticmethod
    def get_all_states_desc() -> str:
        """Return the description of all states as a str.

        Returns: state description str.

        """
        return dict2str(STATE_DESC)
