import json
from langchain.chat_models import ChatLiteLLM

from demo.agents import SalesGPT

GPT_MODEL = "gpt-3.5-turbo-0613"
# GPT_MODEL_16K = "gpt-3.5-turbo-16k-0613"


class SalesGPTAPI:
    USE_TOOLS = False

    def __init__(
        self, config_path: str, verbose: bool = False, max_num_turns: int = 10
    ):
        self.config_path = config_path
        self.verbose = verbose
        self.max_num_turns = max_num_turns
        self.llm = ChatLiteLLM(temperature=0.2, model_name=GPT_MODEL)
        if self.config_path == "":
            print("No agent config specified, using a standard config")
            # USE_TOOLS = True
            if self.USE_TOOLS:
                self.sales_agent = SalesGPT.from_llm(
                    self.llm,
                    use_tools=True,
                    product_catalog="../examples/sample_product_catalog.txt",
                    salesperson_name="Ted Lasso",
                    verbose=self.verbose,
                )
            else:
                self.sales_agent = SalesGPT.from_llm(self.llm, verbose=self.verbose)

        else:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            if self.verbose:
                print(f"Agent config {config}")
            self.sales_agent = SalesGPT.from_llm(self.llm, verbose=self.verbose, **config)
        # seed
        self.sales_agent.seed_agent()

    def do(self, human_input=None):
        #  check turns
        current_turns = len(self.sales_agent.conversation_history) + 1
        if current_turns >= self.max_num_turns:
            # todo:
            # if self.verbose:
            print("Maximum number of turns reached - ending the conversation.")
            return "<END_OF_>"

        if human_input is not None:
            self.sales_agent.human_step(human_input)
            # sales_agent.determine_conversation_stage()
            # print('=' * 10)
            # print(f"conversation_stage_id:{sales_agent.conversation_stage_id}")

        self.sales_agent.step()
        reply = self.sales_agent.conversation_history[-1].replace("<END_OF_TURN>", "").replace("<END_OF_CALL>", "")
        # end conversation
        if "<END_OF_CALL>" in self.sales_agent.conversation_history[-1]:
            print("Sales Agent determined it is time to end the conversation.")
            return reply.split(": ")

        if self.verbose:
            print("=" * 10)
            print(f"{self.sales_agent.salesperson_name}: {reply}")
        return reply.split(": ")
