from copy import deepcopy
from typing import Any, Callable, Optional

from langchain.agents import AgentExecutor
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chat_models import ChatLiteLLM
from langchain.llms.base import create_base_retry_decorator
from litellm import acompletion
from pydantic import Field

from demo_with_machine.chains import SalesConversationChain, StageAnalyzerChain, KnowledgeableAgent, UseToolAgent
from utils.logger import time_logger
from demo_with_machine.tools import get_tools, setup_knowledge_base


def _create_retry_decorator(llm: Any) -> Callable[[Any], Any]:
    import openai

    errors = [
        openai.error.Timeout,
        openai.error.APIError,
        openai.error.APIConnectionError,
        openai.error.RateLimitError,
        openai.error.ServiceUnavailableError,
    ]
    return create_base_retry_decorator(error_types=errors, max_retries=llm.max_retries)


class SalesGPT(Chain):
    """Controller model for the Sales Agent."""

    conversation_history: list[str] = []
    # conversation_stage_id: str = "1"
    # current_conversation_stage: str = CONVERSATION_STAGES.get("1")
    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    sales_conversation_utterance_chain: SalesConversationChain = Field(...)
    sales_knowledgeable_conversation_agent: Optional[KnowledgeableAgent] = Field(...)
    use_tool_agent: Optional[UseToolAgent] = Field(...)
    # conversation_stage_dict: Dict = CONVERSATION_STAGES
    # conversation_stage_str: str = dict2str(CONVERSATION_STAGES)

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

    # def retrieve_conversation_stage(self, key):
    #     return self.conversation_stage_dict.get(key, "1")

    @property
    def input_keys(self) -> list[str]:
        return []

    @property
    def output_keys(self) -> list[str]:
        return []

    @time_logger
    def seed_agent(self):
        # Step 1: seed the conversation
        # self.current_conversation_stage = self.retrieve_conversation_stage("1")
        self.conversation_history = []
        self.reset()
        self.transition_state()

    def _fixed_words_chat(self, words: str):
        ai_message = self.salesperson_name + ": " + words + " <END_OF_TURN>"
        self.conversation_history.append(ai_message)
        print(ai_message.replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", ""))

    def human_step(self, human_input):
        # process human input
        human_input = self.human_prefix + ": " + human_input + " <END_OF_TURN>"
        self.conversation_history.append(human_input)
        # print(human_input.replace("<END_OF_TURN>", ""))

    @time_logger
    def step(self, stream: bool = False):
        """
        Args:
            stream (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not stream:
            self._step()
        else:
            return self._streaming_generator()

    @time_logger
    def astep(self, stream: bool = False):
        """
        Args:
            stream (bool): whether or not return
            streaming generator object to manipulate streaming chunks in downstream applications.
        """
        if not stream:
            self._acall(inputs={})
        else:
            return self._astreaming_generator()

    @time_logger
    def acall(self, *args, **kwargs):
        raise NotImplementedError("This method has not been implemented yet.")

    @time_logger
    def _prep_messages(self):
        """
        Helper function to prepare messages to be passed to a streaming generator.
        """
        prompt = self.sales_conversation_utterance_chain.prep_prompts(
            [
                dict(
                    conversation_stage=self.current_conversation_stage,
                    conversation_history="\n".join(self.conversation_history),
                    salesperson_name=self.salesperson_name,
                    salesperson_role=self.salesperson_role,
                    company_name=self.company_name,
                    company_business=self.company_business,
                    company_values=self.company_values,
                    conversation_purpose=self.conversation_purpose,
                    conversation_type=self.conversation_type,
                    conversation_stages=self.conversation_stage_str,
                )
            ]
        )

        inception_messages = prompt[0][0].to_messages()

        message_dict = {"role": "system", "content": inception_messages[0].content}

        if self.sales_conversation_utterance_chain.verbose:
            print("\033[92m" + inception_messages[0].content + "\033[0m")
        return [message_dict]

    @time_logger
    def _streaming_generator(self):
        """
        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._streaming_generator()
        # Now I can loop through the output in chunks:
        >> for chunk in streaming_generator:
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return self.sales_conversation_utterance_chain.llm.completion_with_retry(
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    async def acompletion_with_retry(self, llm: Any, **kwargs: Any) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(llm)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            # Use OpenAI's async api https://github.com/openai/openai-python#async-api
            return await acompletion(**kwargs)

        return await _completion_with_retry(**kwargs)

    async def _astreaming_generator(self):
        """
        Asynchronous generator to reduce I/O blocking when dealing with multiple
        clients simultaneously.

        Sometimes, the sales agent wants to take an action before the full LLM output is available.
        For instance, if we want to do text to speech on the partial LLM output.

        This function returns a streaming generator which can manipulate partial output from an LLM
        in-flight of the generation.

        Example:

        >> streaming_generator = self._astreaming_generator()
        # Now I can loop through the output in chunks:
        >> async for chunk in streaming_generator:
            await chunk ...
        Out: Chunk 1, Chunk 2, ... etc.
        See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_stream_completions.ipynb
        """

        messages = self._prep_messages()

        return await self.acompletion_with_retry(
            llm=self.sales_conversation_utterance_chain.llm,
            messages=messages,
            stop="<END_OF_TURN>",
            stream=True,
            model=self.model_name,
        )

    def _call(
            self,
            inputs: dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> dict[str, Any]:
        """Run one step of the sales agent.

        """
        if not self.chat_flag:
            # skip llm chat if not needed
            self.chat_flag = True
            return {}
        ai_message = self.sales_conversation_utterance_chain.run(
            conversation_history="\n".join(self.conversation_history),
            salesperson_name=self.salesperson_name,
            salesperson_role=self.salesperson_role,
            company_name=self.company_name,
            company_business=self.company_business,
            conversation_purpose=self.conversation_purpose,
            conversation_type=self.conversation_type,
            # conversation_stages=self.all_state_desc,
            current_stage=self.get_state_desc()
        )

        # Add agent's response to conversation history
        ai_message = ai_message.split(self.salesperson_name + ": ")[0].split(self.human_prefix + ": ")[0].strip()
        self._fixed_words_chat(ai_message)
        return {}

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM, verbose: bool = False, **kwargs) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        # set up stage analyzer
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)

        # set up conversation chain
        if "custom_prompt" in kwargs.keys() and kwargs["custom_prompt"] != "":
            custom_prompt = deepcopy(kwargs["custom_prompt"])
            sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose,
                                                                                 custom_prompt=custom_prompt)
        else:
            sales_conversation_utterance_chain = SalesConversationChain.from_llm(llm, verbose=verbose)
        kwargs.pop("custom_prompt", None)

        use_tool_agent = None
        if "use_tools" in kwargs.keys() and kwargs["use_tools"] == "True":
            # set up agent with tools
            kwargs["use_tools"] = True
            # todo
            use_tool_agent = None

        sales_knowledgeable_conversation_agent = None
        if "use_knowledge" in kwargs.keys() and kwargs["use_knowledge"] == "True":
            # set up agent with knowledge base
            kwargs["use_knowledge"] = True
            if "knowledge_file" in kwargs.keys() and kwargs["knowledge_file"] != "":
                knowledge_file = kwargs["knowledge_file"]
                knowledge_base = setup_knowledge_base(knowledge_file)
            else:
                knowledge_base = setup_knowledge_base()

            tools = get_tools(knowledge_base)
            sales_knowledgeable_conversation_agent = KnowledgeableAgent.from_llm(
                llm=llm,
                tools=tools,
                ai_prefix=kwargs["salesperson_name"],
                verbose=verbose)
        kwargs.pop("knowledge_file", None)

        agent = cls(
            stage_analyzer_chain=stage_analyzer_chain,
            sales_conversation_utterance_chain=sales_conversation_utterance_chain,
            sales_knowledgeable_conversation_agent=sales_knowledgeable_conversation_agent,
            use_tool_agent=use_tool_agent,
            model_name=llm.model,
            verbose=verbose,
            **kwargs
        )
        return agent

    # @time_logger
    # def determine_conversation_stage(self):
    #     predicted_id = self.stage_analyzer_chain.run(
    #         conversation_history="\n".join(self.conversation_history).rstrip("\n"),
    #         conversation_stage_id=self.conversation_stage_id,
    #         conversation_stages=self.conversation_stage_str,
    #         salesperson_role=self.salesperson_role,
    #     )
    #     self.conversation_stage_id = str(predicted_id).strip()
    #
    #     print(f"Conversation Stage ID: {self.conversation_stage_id}")
    #     self.current_conversation_stage = self.retrieve_conversation_stage(
    #         self.conversation_stage_id
    #     )
    #
    #     print(f"Conversation Stage: {self.current_conversation_stage}")
