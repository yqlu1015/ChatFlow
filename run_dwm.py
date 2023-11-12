import argparse
import json

from dotenv import load_dotenv
from langchain.chat_models import ChatLiteLLM

from demo_with_machine.state_machine import AgentMachine

load_dotenv(dotenv_path="./.env")  # loads .env file

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser(description="Description of your program")

    # Add arguments
    parser.add_argument(
        "--config", type=str, help="Path to agent config file", default=""
    )
    parser.add_argument("--verbose", type=bool, help="Verbosity",
                        default=False)
    parser.add_argument(
        "--max_num_turns",
        type=int,
        help="Maximum number of turns in the sales conversation",
        default=100,
    )

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    config_path = args.config
    verbose = args.verbose
    max_num_turns = args.max_num_turns

    llm = ChatLiteLLM(temperature=0.2, model_name="gpt-3.5-turbo-instruct")

    with open(config_path, "r", encoding="UTF-8") as f:
        config = json.load(f)
        print(f"Agent config {config}")
        sales_agent = AgentMachine(llm, verbose=verbose, **config)
        print(sales_agent)
        print(sales_agent.get_all_states())
        # machine = CustomMachine(model=sales_agent, states=states, transitions=TRANSITIONS,
        #                         initial=sales_agent.initial_state, auto_transitions=True)

    sales_agent.seed_agent()
    print("=" * 10)
    cnt = 0
    while cnt != max_num_turns:
        cnt += 1
        if cnt == max_num_turns:
            print("Maximum number of turns reached - ending the conversation.")
            break
        # message = sales_agent.step()
        # print(message)
        sales_agent.step()

        # end conversation
        if sales_agent.state.name == "End":
            print("Sales Agent determined it is time to end the conversation.")
            break
        human_input = input("您的回复: ")
        sales_agent.human_step(human_input)
        print("=" * 10)
