# ReAct (Reasoning + Acting) Pattern

## Overview
ReAct (Reasoning + Acting) is a powerful pattern that combines explicit reasoning with action-taking capabilities in LLMs. First introduced in the paper "ReAct: Synergizing Reasoning and Acting in Language Models" by Yao et al. (2022), this approach enables models to interleave thought generation with actions, creating a synergistic loop of thinking, acting, and observation.

This pattern extends Chain-of-Thought prompting by adding the ability for the model to interact with external environments or tools. By explicitly reasoning about observations and planning next steps, ReAct enables LLMs to solve complex, multi-step tasks that require both deliberation and information-gathering, significantly improving accuracy and reliability in interactive problem-solving scenarios.

## How It Works

The ReAct pattern creates a structured cycle with three key phases:

1. **Reasoning (Thought)**: The model explicitly verbalizes its reasoning process, analyzing the current state, available information, and potential approaches.

2. **Acting (Action)**: Based on its reasoning, the model selects and executes a specific action, such as using a tool, querying an external source, or making a decision.

3. **Observing (Observation)**: The model receives and processes feedback from the action, incorporating new information into its understanding of the problem.

This cycle repeats until the model reaches a conclusion or completes the task, with each phase informing the next in a coherent problem-solving flow.

## Key Components

### 1. Structured Thinking-Acting Format

The ReAct pattern follows a specific structure that makes the reasoning and acting processes explicit:

```
THOUGHT: [Reasoning about the current situation and deciding what to do]
ACTION: [tool_name](parameters)
OBSERVATION: [Results from executing the action]
THOUGHT: [Reasoning about the observation and determining next steps]
...
THOUGHT: [Final reasoning after gathering sufficient information]
ANSWER: [Final response to the query]
```

### 2. Tool Integration

ReAct requires a well-defined set of tools that the model can invoke:

- **Search tools**: Access to search engines, knowledge bases, or databases
- **Computation tools**: Calculators, code interpreters, or specialized algorithms
- **Interaction tools**: APIs, file systems, or other external systems
- **Memory tools**: Ability to store and retrieve information from previous steps

### 3. Observation Processing

How the model interprets and incorporates results from actions:

- **Information extraction**: Identifying key facts from observations
- **Context updating**: Integrating new information with existing knowledge
- **Contradiction handling**: Managing conflicts between observations and expectations
- **Planning revision**: Adjusting strategy based on new information

### 4. Decision Logic

Rules and heuristics for determining when to act vs. think:

- **Information need assessment**: Recognizing when more information is required
- **Tool selection logic**: Choosing the most appropriate tool for a given situation
- **Task decomposition**: Breaking complex tasks into manageable action steps
- **Completion recognition**: Determining when sufficient information has been gathered

## Implementation Approaches

### Basic Implementation

Here's a simple implementation of the ReAct pattern using a function-based approach:

```python
def react_agent(question, tools, max_steps=10):
    context = f"Question: {question}\n\n"
    
    for step in range(max_steps):
        # Generate thought
        thought = generate_thought(context)
        context += f"THOUGHT: {thought}\n"
        
        # Check if we've reached a conclusion
        if "I now have the answer" in thought or "The answer is" in thought:
            answer = generate_answer(context)
            context += f"ANSWER: {answer}\n"
            return context
        
        # Generate action
        action = generate_action(context, available_tools=tools)
        context += f"ACTION: {action}\n"
        
        # Execute action and get observation
        tool_name, parameters = parse_action(action)
        observation = execute_tool(tool_name, parameters)
        context += f"OBSERVATION: {observation}\n"
    
    # If we've reached max steps without a conclusion
    answer = generate_answer(context + "THOUGHT: I need to provide my best answer based on the information gathered so far.\n")
    return context + f"ANSWER: {answer}\n"
```

### Prompt-Based Implementation

```python
react_prompt_template = """
You are a helpful assistant that solves problems step-by-step.
First, think about the problem.
Then, decide if you need more information.
If you do, use one of the available tools.
After getting information, think again about what you've learned.
Continue this process until you can provide a final answer.

Use the following format:
THOUGHT: [Your reasoning about the current situation]
ACTION: [tool_name](parameters)
OBSERVATION: [Results from the action will appear here]
... (continue this pattern as needed)
THOUGHT: [Your final reasoning]
ANSWER: [Your final answer to the original question]

Available tools:
{tools_description}

Question: {question}
"""
```

### LangChain Implementation

```python
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Useful for finding information about events, people, or facts."
    ),
    Tool(
        name="Calculator",
        func=calculator_function,
        description="Useful for performing mathematical calculations."
    ),
]

# Create ReAct agent
llm = OpenAI(temperature=0)
react_agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor.from_agent_and_tools(agent=react_agent, tools=tools, verbose=True)

# Run the agent
response = agent_executor.run("What is the population of Canada divided by the square root of 2?")
```

## Advantages and Limitations

### Advantages

- **Enhanced Problem-Solving**: Combines deliberative reasoning with information-gathering
- **Transparency**: Makes both reasoning and actions explicit and auditable
- **Reduced Hallucination**: Grounds responses in external information sources
- **Adaptability**: Can handle unexpected situations by adjusting strategy based on observations
- **Tool Synergy**: Effectively combines multiple tools to solve complex problems

### Limitations

- **Complexity**: More complex to implement than simpler patterns like Chain-of-Thought
- **Token Consumption**: Requires significant context space to track the reasoning-acting process
- **Error Propagation**: Mistakes in early steps can compound through the process
- **Tool Dependency**: Effectiveness limited by the quality and coverage of available tools
- **Step Limitations**: May struggle with problems requiring many action steps due to context limitations

## Best Practices

### Agent Design

- **Clear Tool Descriptions**: Provide precise descriptions of each tool's capabilities and limitations
- **Thought Prompting**: Encourage detailed reasoning before taking actions
- **Observation Processing**: Prompt the model to explicitly analyze observations
- **Error Recovery**: Build in mechanisms for recovering from tool errors or unexpected results

### Tool Selection

- **Complementary Tools**: Provide tools that cover different aspects of problem-solving
- **Abstraction Levels**: Offer both high-level and low-level tools for different tasks
- **Documentation**: Include usage examples and parameter specifications for each tool
- **Error Handling**: Ensure tools provide useful error messages when used incorrectly

### Prompt Engineering

- **Task Decomposition Guidance**: Help the model break down complex tasks
- **Example Traces**: Include examples of successful reasoning-acting chains
- **Tool Selection Guidance**: Provide heuristics for choosing appropriate tools
- **Reflection Prompting**: Encourage the model to evaluate the sufficiency of gathered information

## Examples

### Information Gathering and Analysis

```
Question: What is the current population of Tokyo, and how does it compare to the population in 2010?

THOUGHT: To answer this question, I need to find two pieces of information: the current population of Tokyo, and the population of Tokyo in 2010. Let me search for this information.

ACTION: search(current population of Tokyo)
OBSERVATION: According to the Tokyo Metropolitan Government, the current population of Tokyo prefecture is approximately 14.04 million (2023 estimate).

THOUGHT: Now I have the current population of Tokyo. Next, I need to find the population in 2010 for comparison.

ACTION: search(Tokyo population 2010)
OBSERVATION: According to official census data, the population of Tokyo prefecture in 2010 was 13.16 million.

THOUGHT: Now I have both pieces of information:
- Current population (2023): 14.04 million
- Population in 2010: 13.16 million

To compare them, I need to calculate the difference and percentage increase.

ACTION: calculator(14.04 - 13.16)
OBSERVATION: 0.88

ACTION: calculator((14.04 - 13.16) / 13.16 * 100)
OBSERVATION: 6.69

THOUGHT: I now have all the information needed to answer the question.

ANSWER: The current population of Tokyo is approximately 14.04 million (2023 estimate), compared to 13.16 million in 2010. This represents an increase of 0.88 million people, or about 6.69% growth over this 13-year period.
```

### Problem Solving with Multiple Tools

```
Question: If I invest $10,000 in a stock that grows by 7% annually, how much will I have after 20 years, and what will be its value in today's dollars assuming 2.5% annual inflation?

THOUGHT: This problem requires calculating compound growth and then adjusting for inflation. I'll break this into steps:
1. Calculate the future value with compound interest
2. Calculate the inflation-adjusted value (real value)

ACTION: calculator(10000 * (1 + 0.07)^20)
OBSERVATION: 38,696.84

THOUGHT: So after 20 years at 7% annual growth, the investment will be worth $38,696.84. Now I need to adjust for inflation to find the value in today's dollars.

ACTION: calculator(38696.84 / (1 + 0.025)^20)
OBSERVATION: 23,647.92

THOUGHT: I've now calculated both values requested.

ANSWER: If you invest $10,000 in a stock that grows by 7% annually, after 20 years it will be worth $38,696.84. Adjusted for inflation at 2.5% annually, that amount would be equivalent to $23,647.92 in today's dollars. This means your investment will have more than doubled in real purchasing power.
```

## Further Reading

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

2. Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

3. Weng, L. (2023). LLM Powered Autonomous Agents. [Lil'Log](https://lilianweng.github.io/posts/2023-06-23-agent/)

4. Chase, H. (2022). LangChain: Building applications with LLMs through composability. [GitHub repository](https://github.com/hwchase17/langchain)

5. OpenAI. (2023). Function calling and other API updates. [OpenAI Blog](https://openai.com/blog/function-calling-and-other-api-updates)