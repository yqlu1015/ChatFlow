from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatLiteLLM
from langchain.agents import Tool, LLMSingleActionAgent, AgentExecutor

from utils.logger import time_logger
from demo_with_machine.prompts import (SALES_AGENT_INCEPTION_PROMPT, SALES_AGENT_KNOWLEDGE_PROMPT,
                                       STAGE_ANALYZER_INCEPTION_PROMPT)
from utils.parsers import SalesConvoOutputParser
from utils.templates import CustomPromptTemplateForTools


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into.

    """

    @classmethod
    @time_logger
    def from_llm(cls, llm: ChatLiteLLM,
                 prompt_template: str = STAGE_ANALYZER_INCEPTION_PROMPT, verbose: bool = False) -> "StageAnalyzerChain":
        """Get the response parser."""
        stage_analyzer_inception_prompt_template = prompt_template
        prompt = PromptTemplate(
            template=stage_analyzer_inception_prompt_template,
            input_variables=[
                "conversation_history",
                "current_state",
                "next_states",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class SalesConversationChain(LLMChain):
    """Chain to generate the next utterance for the conversation.

    """

    @classmethod
    @time_logger
    def from_llm(
            cls,
            llm: ChatLiteLLM,
            verbose: bool = True,
            custom_prompt: str = SALES_AGENT_INCEPTION_PROMPT,
    ) -> "SalesConversationChain":
        """Get the response parser."""
        prompt = PromptTemplate(
            template=custom_prompt,
            input_variables=[
                "salesperson_name",
                "salesperson_role",
                "company_name",
                "company_business",
                "conversation_purpose",
                "conversation_type",
                "conversation_history",
                # "conversation_stages",
                "current_stage",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class KnowledgeableAgent(AgentExecutor):
    """ Conversation agent with some knowledge base.

    """

    @classmethod
    @time_logger
    def from_llm(
            cls,
            llm: ChatLiteLLM,
            tools: list[Tool],
            ai_prefix: str,
            verbose: bool = True,
            prompt_template: str = SALES_AGENT_KNOWLEDGE_PROMPT,
    ) -> "KnowledgeableAgent":
        prompt = CustomPromptTemplateForTools(
            template=prompt_template,
            tools_getter=lambda x: tools,
            # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
            # This includes the `intermediate_steps` variable because that is needed
            input_variables=[
                "input",
                "intermediate_steps",
                "salesperson_name",
                "salesperson_role",
                "company_name",
                "company_business",
                "conversation_purpose",
                "conversation_type",
                "conversation_history",
                # "conversation_stages",
                "current_stage",
            ],
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=verbose)

        tool_names = [tool.name for tool in tools]

        # WARNING: this output parser is NOT reliable yet
        # It makes assumptions about output from LLM which can break and throw an error
        output_parser = SalesConvoOutputParser(ai_prefix=ai_prefix)

        sales_agent_with_tools = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=output_parser,
            stop=["\nObservation:"],
            allowed_tools=tool_names,
        )

        return cls.from_agent_and_tools(
            agent=sales_agent_with_tools, tools=tools, verbose=verbose
        )


class UseToolAgent(AgentExecutor):
    pass
