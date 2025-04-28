# Agents and Tools Pattern

## Overview
The Agents and Tools pattern represents a powerful paradigm that transforms LLMs from passive text generators into active problem solvers by enabling them to use external tools and make decisions about which tools to apply. This approach significantly extends LLM capabilities beyond their intrinsic knowledge, allowing them to perform actions like searching the web, executing code, accessing databases, making API calls, and manipulating data.

At its core, this pattern treats the LLM as an "agent" that can reason about the user's goal, plan a sequence of actions using available tools, observe results, and adapt its approach until the task is complete. Unlike standard prompting, which produces a single response, the agent pattern creates an interactive loop of reasoning, tool selection, execution, and result interpretation that mimics human problem-solving behavior.

The emergence of this pattern has enabled AI systems to tackle complex, multi-step tasks that were previously unachievable with direct prompting alone, representing a significant advancement toward more general and autonomous AI assistants.

## How It Works

The Agents and Tools pattern operates through a structured loop that includes several key phases:

1. **Task Analysis**: The agent receives a user request and analyzes what needs to be accomplished
2. **Planning**: The agent determines which tools might be helpful and in what sequence
3. **Tool Selection**: The agent chooses the most appropriate tool for the current step
4. **Tool Execution**: The system executes the selected tool with parameters provided by the agent
5. **Observation**: The agent receives and interprets the results from the tool
6. **Decision Making**: Based on the observations, the agent decides to either:
   - Continue with another tool action
   - Request more information from the user
   - Provide a final answer

This process continues iteratively until the agent determines that it has gathered sufficient information to provide a complete response to the user's request.

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  Agent Analysis ├─────►│   Final Answer  │
└────────┬────────┘      └─────────────────┘
         │ (if tools needed)       ▲
         ▼                         │
┌─────────────────┐      ┌─────────────────┐
│  Tool Selection ├─────►│   Observation   │
└────────┬────────┘      └────────┬────────┘
         │                        │
         ▼                        │
┌─────────────────┐               │
│ Tool Execution  ├───────────────┘
└─────────────────┘
```

## Key Components

### 1. Agent Architectures

Different approaches to structuring LLM-based agents:

#### ReAct Agent
Based on the ReAct pattern with explicit reasoning steps:
```python
def react_agent(query, tools):
    context = f"Query: {query}\n\n"
    
    while True:
        # Generate thought
        thought = llm.generate(context + "Thought: ")
        context += f"Thought: {thought}\n"
        
        # Check if we should finish
        if "I can now answer the query" in thought:
            final_answer = llm.generate(context + "Answer: ")
            return final_answer
        
        # Generate action
        action = llm.generate(context + "Action: ")
        context += f"Action: {action}\n"
        
        # Execute tool
        tool_name, params = parse_action(action)
        result = execute_tool(tool_name, params)
        
        # Record observation
        context += f"Observation: {result}\n"
```

#### Function Calling Agent
Using structured function calling interfaces:
```python
# Define available tools as functions
tools = [
    {
        "name": "search",
        "description": "Search the web for information",
        "parameters": {
            "query": {
                "type": "string",
                "description": "The search query"
            }
        }
    },
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate"
            }
        }
    }
]

def function_calling_agent(query):
    messages = [{"role": "user", "content": query}]
    
    while True:
        response = llm.chat(
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        # Check if the model wants to call a function
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Execute the function
                result = execute_tool(function_name, arguments)
                
                # Add the function response to the messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
        else:
            # Model provided a regular response
            return response.content
```

#### Planning-Based Agent
Agents that create explicit plans before execution:
```python
def planning_agent(query, tools):
    # First generate a plan
    plan_prompt = f"""
    User query: {query}
    Available tools: {format_tools(tools)}
    
    Create a step-by-step plan to answer this query, 
    identifying which tools to use at each step.
    """
    
    plan = llm.generate(plan_prompt)
    
    # Execute each step in the plan
    steps = parse_plan(plan)
    
    results = []
    for step in steps:
        if step.requires_tool:
            tool_result = execute_tool(step.tool_name, step.parameters)
            results.append((step, tool_result))
    
    # Generate final answer based on plan and results
    answer_prompt = f"""
    User query: {query}
    Plan: {plan}
    Results: {format_results(results)}
    
    Provide a comprehensive answer to the user's query.
    """
    
    return llm.generate(answer_prompt)
```

### 2. Tool Integration Framework

Components required for effective tool usage:

#### Tool Definition System
How tools are specified and described to the agent:
```python
# Example tool definition
tool = {
    "name": "weather_api",
    "description": "Get current weather information for a location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City or address"
            },
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    },
    "returns": {
        "type": "object",
        "description": "Weather information including temperature, conditions, etc."
    }
}
```

#### Tool Execution System
Mechanism for calling tools and handling results:
```python
def execute_tool(tool_name, parameters):
    # Input validation
    tool = find_tool_by_name(tool_name)
    validate_parameters(tool, parameters)
    
    try:
        # Tool execution with timeout and error handling
        with timeout(seconds=5):
            result = tool["function"](**parameters)
        
        # Result formatting
        formatted_result = format_tool_result(result, tool)
        return formatted_result
    
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"
```

#### Tool Discovery and Selection
How agents find and choose appropriate tools:
```python
def suggest_relevant_tools(query, available_tools):
    tool_suggestion_prompt = f"""
    User query: {query}
    
    Given the following available tools, list the tools (if any) that would be 
    helpful in answering this query. For each tool, explain why it would be useful.
    
    Available tools:
    {format_tools_for_prompt(available_tools)}
    """
    
    suggestions = llm.generate(tool_suggestion_prompt)
    relevant_tools = parse_tool_suggestions(suggestions)
    
    return relevant_tools
```

### 3. Memory and State Management

Systems for maintaining context between tool uses:

#### Working Memory
Keeping track of the current conversation and tool results:
```python
class AgentMemory:
    def __init__(self):
        self.conversation_history = []
        self.tool_results = {}
        self.key_facts = set()
    
    def add_message(self, role, content):
        self.conversation_history.append({"role": role, "content": content})
    
    def add_tool_result(self, tool_name, parameters, result):
        execution_id = str(uuid.uuid4())
        self.tool_results[execution_id] = {
            "tool": tool_name,
            "parameters": parameters,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        return execution_id
    
    def add_key_fact(self, fact):
        self.key_facts.add(fact)
    
    def get_context(self, max_tokens=2000):
        # Create a context summary within token limits
        context = {
            "recent_messages": self.conversation_history[-5:],
            "key_facts": list(self.key_facts),
            "recent_tool_results": dict(list(self.tool_results.items())[-3:])
        }
        return context
```

#### Long-term Memory
Storing information across sessions:
```python
class LongTermMemory:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def store_interaction(self, query, response, tool_uses):
        # Create embedding and store interaction
        interaction = {
            "query": query,
            "response": response,
            "tool_uses": tool_uses,
            "timestamp": datetime.now().isoformat()
        }
        
        self.vector_store.add(
            text=f"{query} {response}",
            metadata=interaction
        )
    
    def retrieve_relevant_memories(self, current_query, k=3):
        results = self.vector_store.similarity_search(current_query, k=k)
        return results
```

### 4. Error Handling and Recovery

Mechanisms for dealing with tool failures and unexpected results:

```python
def execute_tool_with_recovery(agent, tool_name, parameters, max_retries=2):
    retries = 0
    while retries <= max_retries:
        try:
            result = execute_tool(tool_name, parameters)
            
            # Check if result suggests an error
            if error_detector.is_error(result):
                # Let the agent interpret the error
                recovery_action = agent.generate_recovery_action(
                    tool_name, parameters, result
                )
                
                if recovery_action.type == "retry":
                    # Try again with modified parameters
                    parameters = recovery_action.parameters
                    retries += 1
                    continue
                elif recovery_action.type == "alternative_tool":
                    # Try a different tool
                    return execute_tool_with_recovery(
                        agent, 
                        recovery_action.tool_name, 
                        recovery_action.parameters
                    )
                elif recovery_action.type == "abort":
                    # Give up on this approach
                    return f"Unable to execute {tool_name}: {result}"
            
            # Success
            return result
            
        except Exception as e:
            retries += 1
            if retries > max_retries:
                return f"Failed to execute {tool_name} after {max_retries} attempts: {str(e)}"
```

## Implementation Approaches

### Basic Implementation with OpenAI Function Calling

```python
import openai
import json

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state or country"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Tool implementation functions
def search_web(query):
    # Implement actual search functionality
    return f"Search results for '{query}'..."

def get_weather(location):
    # Implement actual weather API call
    return f"Weather in {location}: 72°F, Sunny"

# Function execution dispatcher
def execute_function(name, arguments):
    if name == "search_web":
        return search_web(**arguments)
    elif name == "get_weather":
        return get_weather(**arguments)
    else:
        return f"Function {name} not found"

# Agent implementation
def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]
    
    # Maximum number of steps to prevent infinite loops
    max_steps = 10
    step = 0
    
    while step < max_steps:
        step += 1
        
        # Get model response
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        # Check if the model wants to use a tool
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the function
                function_response = execute_function(function_name, function_args)
                
                # Add the function response to the messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
        else:
            # If no tool calls, we're done
            return response_message.content
    
    # If we've reached max steps, generate a final response
    final_response = openai.chat.completions.create(
        model="gpt-4",
        messages=messages + [{"role": "user", "content": "Please provide your final answer."}]
    )
    
    return final_response.choices[0].message.content
```

### Implementation with LangChain

```python
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI

# Define tools
class SearchTool(BaseTool):
    name = "search_web"
    description = "Search the web for information"
    
    def _run(self, query: str) -> str:
        # Implement actual search
        return f"Search results for '{query}'..."
    
    def _arun(self, query: str) -> str:
        # Async implementation
        return self._run(query)

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "Get the current weather for a location"
    
    def _run(self, location: str) -> str:
        # Implement actual weather lookup
        return f"Weather in {location}: 72°F, Sunny"
    
    def _arun(self, location: str) -> str:
        # Async implementation
        return self._run(location)

# Create the tools list
tools = [
    SearchTool(),
    WeatherTool()
]

# Set up the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that can use tools to answer questions."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Set up memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the model
llm = ChatOpenAI(model="gpt-4")

# Create the agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Create the executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Run the agent
def run_agent(query):
    return agent_executor.invoke({"input": query})
```

## Advantages and Limitations

### Advantages

- **Extended Capabilities**: Enables LLMs to perform actions beyond text generation
- **Real-time Information**: Access to current data through API calls
- **Computation Powers**: Ability to perform calculations and data analysis
- **Multi-step Problem Solving**: Can break down complex tasks into manageable steps
- **Adaptability**: Dynamically chooses different approaches based on intermediate results
- **Transparency**: Reasoning steps provide insight into the agent's decision-making

### Limitations

- **Complexity**: More complex to implement and maintain than simple prompting
- **Tool Dependencies**: Quality depends on available tools and their reliability
- **Latency Issues**: Multiple tool calls increase response time
- **Error Propagation**: Mistakes in early steps can compound through the process
- **Hallucinated Tool Usage**: Models may fabricate tool capabilities or results
- **Security Concerns**: Tool access presents potential security and safety risks

## Best Practices

### Tool Design

- **Clear Descriptions**: Write precise, concise descriptions of each tool
- **Explicit Parameters**: Clearly specify all required and optional parameters
- **Error Messages**: Design informative error messages for invalid inputs
- **Output Format Consistency**: Maintain consistent output structures
- **Rate Limiting**: Implement rate limiting to prevent excessive tool usage

```python
# Example of a well-designed tool description
{
    "name": "weather_api",
    "description": "Get current weather information for a location. Returns temperature, conditions, and humidity.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name and optionally country code, e.g., 'Paris, FR' or 'New York'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units (default: celsius)"
            }
        },
        "required": ["location"]
    },
    "returns": {
        "type": "object",
        "description": "Weather information object with temperature, conditions, and humidity"
    },
    "errors": [
        "Location not found",
        "API limit exceeded",
        "Connection timeout"
    ]
}
```

### Agent Prompting

- **Clear Instructions**: Provide clear instructions about when to use tools
- **Few-shot Examples**: Include examples of effective tool usage
- **Reasoning Encouragement**: Prompt the agent to explain its reasoning
- **Verification Instructions**: Encourage checking results before finalizing answers

```
You are a helpful assistant with access to the following tools:
{tools_description}

When using tools, follow these guidelines:
1. Think carefully about which tool is appropriate for the user's request
2. Format your tool calls exactly according to the tool's required parameters
3. Always check if the tool response contains errors and handle them appropriately
4. After receiving tool outputs, verify if you need additional information
5. Provide your final answer with a clear explanation

Here's an example of good tool usage:
User: What's the weather in Paris right now?
Assistant thought: I should use the weather_api tool to get current weather in Paris.
Tool call: weather_api(location="Paris, FR")
Tool response: {"temperature": 22, "unit": "celsius", "conditions": "Partly Cloudy", "humidity": 65}
Assistant: The current weather in Paris is 22°C (Partly Cloudy) with 65% humidity.
```

### Error Handling

- **Graceful Degradation**: Fallback strategies when tools fail
- **Retry Logic**: Implement smart retry mechanisms
- **User Communication**: Clearly communicate tool failures to users
- **Alternative Tools**: Suggest alternative approaches when preferred tools are unavailable

### Security Considerations

- **Access Controls**: Limit tool capabilities based on security requirements
- **Input Validation**: Validate all parameters before execution
- **Output Filtering**: Filter sensitive information from tool outputs
- **Monitoring**: Track and audit tool usage
- **Rate Limiting**: Prevent abuse through rate limiting

## Examples

### Research Assistant Agent

```
User: I need to write a report on sustainable energy in Denmark. Can you help me gather some information?

Agent Thought: This requires finding current information about sustainable energy initiatives and statistics in Denmark. I should use a search tool to gather this information.

Agent Action: search_web(query="sustainable energy Denmark statistics 2023")

Observation: Denmark is a world leader in wind energy, with wind power producing 46.8% of Denmark's electricity consumption in 2022. Denmark aims to reduce greenhouse gas emissions by 70% by 2030 compared to 1990 levels. The country is investing heavily in offshore wind farms and green hydrogen production.

Agent Thought: I have some basic statistics, but I need more specific information about different types of renewable energy and recent developments.

Agent Action: search_web(query="Denmark renewable energy mix solar wind biomass recent developments")

Observation: Denmark's renewable energy mix in 2022: Wind 46.8%, Biomass 18.2%, Solar 5.7%, and other renewables 2.3%. The country is rapidly expanding solar capacity with a 36% increase in installations during 2022. The new "energy islands" project will create the world's first artificial islands dedicated to producing and distributing green energy.

Agent Thought: Now I have good information about the energy mix and recent projects. Let me get some information about challenges and future plans.

Agent Action: search_web(query="Denmark sustainable energy challenges future plans 2030")

Observation: Challenges include grid infrastructure upgrades needed to handle increasing renewable capacity, energy storage solutions for intermittent wind and solar, and reducing reliance on biomass which has sustainability concerns. Future plans include 4 GW of additional offshore wind by 2030, expanding electric vehicle infrastructure, and developing green hydrogen production capabilities.

Agent Final Response: Here's the information I've gathered about sustainable energy in Denmark for your report:

Current Status:
- Denmark is a global leader in wind energy, which provided 46.8% of electricity consumption in 2022
- Their renewable energy mix includes: Wind (46.8%), Biomass (18.2%), Solar (5.7%), and other renewables (2.3%)
- Solar capacity is growing rapidly with a 36% increase in installations during 2022

Key Initiatives:
- "Energy islands" project: The world's first artificial islands dedicated to renewable energy production
- Ambitious goal to reduce greenhouse gas emissions by 70% by 2030 (compared to 1990)
- Significant investment in offshore wind farms and green hydrogen production

Challenges and Future Plans:
- Grid infrastructure requires upgrades to handle increasing renewable capacity
- Energy storage solutions are needed for intermittent wind and solar power
- Plans include adding 4 GW of offshore wind by 2030
- Expanding electric vehicle infrastructure nationwide
- Developing green hydrogen production while reducing reliance on biomass

Would you like me to search for more specific information about any of these aspects for your report?
```

### Personal Finance Assistant Agent

```
User: Can you help me figure out how much I'd save if I refinanced my mortgage?

Agent Thought: I need to calculate mortgage payments before and after refinancing, but I need several pieces of information from the user.

Agent: I'd be happy to help you calculate potential savings from refinancing your mortgage. I'll need some information about your current mortgage and the potential refinancing options. Could you provide:

1. Current mortgage details:
   - Original loan amount
   - Current remaining balance
   - Current interest rate
   - Original loan term (e.g., 30 years) and time remaining
   - Current monthly payment

2. Potential refinancing details:
   - New interest rate you're considering
   - New loan term
   - Any closing costs or fees for refinancing

User: My current mortgage is $300,000 with 25 years remaining at 4.5% interest. Monthly payment is $1,520. I could refinance at 3.25% for a new 30-year term with $4,000 in closing costs.

Agent Thought: I have the necessary information now. I need to calculate the new monthly payment and compare the total costs.

Agent Action: calculator(expression="300000 * 0.0325/12 * (1 + 0.0325/12)**(30*12) / ((1 + 0.0325/12)**(30*12) - 1)")

Observation: 1305.62

Agent Thought: I've calculated the new monthly payment would be $1,305.62. Now I need to calculate the actual savings.

Agent Action: calculator(expression="1520 - 1305.62")

Observation: 214.38

Agent Thought: The monthly savings would be $214.38. Over time this adds up, but I need to account for the closing costs and the longer term.

Agent Action: calculator(expression="214.38 * 12 * 25")

Observation: 64314.0

Agent Action: calculator(expression="214.38 * 12 * 30 - 4000")

Observation: 73174.8

Agent Thought: I have all the calculations now. I can provide the analysis.

Agent Final Response: Based on the information you've provided, here's an analysis of refinancing your mortgage:

Current mortgage:
- Remaining balance: $300,000
- Monthly payment: $1,520
- Interest rate: 4.5%
- Term remaining: 25 years

Refinancing option:
- New loan amount: $300,000
- Monthly payment: $1,305.62
- Interest rate: 3.25%
- New term: 30 years
- Closing costs: $4,000

Savings analysis:
- Monthly payment savings: $214.38
- Total savings over the original 25-year term: $64,314
- Net savings over 30 years (after closing costs): $73,174.80

Important considerations:
1. Refinancing would extend your mortgage by 5 years (from 25 to 30 years)
2. You'd break even on the closing costs in about 19 months ($4,000 ÷ $214.38)
3. If you kept the same 25-year term instead of extending to 30 years, your monthly savings would be less but you'd pay off the home sooner

Would you like me to calculate the numbers for a 25-year refinance term as well for comparison?
```

## Further Reading

1. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. & Cao, Y. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)

2. Qin, Y., Liang, S., Zhu, B., Liu, Y., Wang, T., Li, X. & Zhou, M. (2023). Tool Learning with Foundation Models. [arXiv:2304.08354](https://arxiv.org/abs/2304.08354)

3. Nakano, R., Hilton, J., Kelly, S., Taylor, C., Sezener, E., Icard, T., & Steinhardt, J. (2021). WebGPT: Browser-assisted question-answering with human feedback. [arXiv:2112.09332](https://arxiv.org/abs/2112.09332)

4. Chase, H. (2023). LangChain: Building applications with LLMs through composability. [GitHub repository](https://github.com/hwchase17/langchain)

5. Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)

6. Mialon, G., Dessì, R., Lomeli, M., Nalmpantis, C., Pasunuru, R., Raileanu, R., Rozière, B., Schick, T., Dwivedi-Yu, J., Celikyilmaz, A., Grave, E., LeCun, Y., & Scialom, T. (2023). Augmented Language Models: a Survey. [arXiv:2302.07842](https://arxiv.org/abs/2302.07842)

7. Patil, S., Tang, X., Zhuang, N., Gonzalez, J.E. & Olsson, C. (2023). Gorilla: Large Language Model Connected with Massive APIs. [arXiv:2305.15334](https://arxiv.org/abs/2305.15334