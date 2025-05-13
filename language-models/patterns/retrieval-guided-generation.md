# Retrieval-Guided Generation

## Overview
Retrieval-Guided Generation represents an evolution of traditional Retrieval-Augmented Generation (RAG) where the large language model (LLM) itself actively decides when to retrieve information and what to search for, rather than having retrieval happen automatically for every query. This pattern gives the LLM agency over the retrieval process, enabling more efficient and targeted information gathering.

Unlike standard RAG systems where retrieval always precedes generation regardless of necessity, Retrieval-Guided Generation empowers the model to:
1. Recognize when external information is needed versus when it can rely on its parametric knowledge
2. Formulate specific, targeted queries to obtain relevant information
3. Determine when sufficient information has been gathered to answer the user's question

This approach addresses key limitations of traditional RAG, including unnecessary retrievals that waste computational resources, imprecise search queries that retrieve irrelevant information, and the inefficient handling of complex, multi-faceted questions that require multiple retrievals. By enabling LLMs to guide their own retrieval process, this pattern creates more efficient, accurate, and transparent information-seeking systems.

## How It Works

Retrieval-Guided Generation operates through a dynamic, multi-stage process:

1. **Self-Assessment**: The LLM evaluates whether it has sufficient knowledge to answer the user's query or needs additional information.

2. **Query Formulation**: If external information is needed, the LLM generates specific search queries designed to retrieve the most relevant information.

3. **Information Retrieval**: The system executes the LLM-generated searches against appropriate knowledge sources.

4. **Integration & Evaluation**: The LLM incorporates retrieved information and assesses whether additional information is needed.

5. **Iterative Retrieval**: If necessary, the LLM formulates new, refined queries based on the current state of knowledge.

6. **Response Generation**: Once sufficient information is gathered, the LLM synthesizes a comprehensive response.

This creates a dynamic loop of assessment, retrieval, and integration that continues until the model determines it has sufficient information to provide an accurate and complete response.

```
┌───────────────┐
│  User Query   │
└───────┬───────┘
        │
        ▼
┌───────────────┐     ┌───────────────┐
│ Self-Assessment├────►  Final Answer  │
└───────┬───────┘     └───────────────┘
        │ (if information needed)   ▲
        ▼                           │
┌───────────────┐                   │
│Query Formulation                  │
└───────┬───────┘                   │
        │                           │
        ▼                           │
┌───────────────┐     ┌───────────────┐
│  Information  │     │ Sufficient    │
│   Retrieval   ├────►  Information?  │
└───────────────┘     └───────┬───────┘
                              │ (no)
                              ▼
                     ┌───────────────┐
                     │Refine Queries │
                     └───────────────┘
```

## Key Components

### 1. Self-Assessment Mechanisms

Methods for enabling the model to evaluate its knowledge and information needs:

#### Confidence Estimation
```python
def assess_information_need(query, llm):
    """Determine if external information is needed to answer the query."""
    assessment_prompt = f"""
    Question: {query}
    
    Before answering, assess whether you need to search for additional information.
    Rate your confidence on a scale of 1-5, where:
    1 = Very uncertain (definitely need to search)
    2 = Somewhat uncertain (probably should search)
    3 = Moderate confidence (could benefit from verification)
    4 = Confident (likely don't need to search)
    5 = Very confident (definitely don't need to search)
    
    Explain your rating and specifically what information you would need to search for, if any.
    
    Confidence assessment:
    """
    
    assessment = llm.generate(assessment_prompt)
    
    # Parse confidence level
    confidence_level = extract_confidence(assessment)
    
    # Determine if search is needed
    needs_search = confidence_level <= 3
    
    # Extract search topics if needed
    search_topics = extract_search_topics(assessment) if needs_search else []
    
    return {
        "needs_search": needs_search,
        "confidence": confidence_level,
        "search_topics": search_topics,
        "reasoning": assessment
    }
```

#### Knowledge Boundary Recognition
```python
def identify_knowledge_gaps(query, llm, domain_classifier):
    """Identify specific knowledge gaps related to the query."""
    
    # Classify the query domain
    domain = domain_classifier.classify(query)
    
    # Get knowledge cutoff for the domain
    cutoff_date = domain_knowledge_cutoffs.get(domain, DEFAULT_CUTOFF)
    
    knowledge_gap_prompt = f"""
    Question: {query}
    
    Analyze this question and identify any aspects that require information that:
    1. May have changed since {cutoff_date}
    2. Involves specialized knowledge you may not have been extensively trained on
    3. Requires specific data or statistics
    4. References very recent events, people, or developments
    
    For each identified gap, explain why you think external information is needed.
    """
    
    analysis = llm.generate(knowledge_gap_prompt)
    
    # Extract structured knowledge gaps
    knowledge_gaps = parse_knowledge_gaps(analysis)
    
    return knowledge_gaps
```

### 2. Query Formulation Strategies

Techniques for generating effective search queries:

#### Direct Query Generation
```python
def generate_search_queries(query, knowledge_gaps, llm):
    """Generate specific search queries based on identified knowledge gaps."""
    
    if not knowledge_gaps:
        return []
    
    query_generation_prompt = f"""
    Original question: {query}
    
    Information needed:
    {format_knowledge_gaps(knowledge_gaps)}
    
    Generate 1-3 specific search queries that would help find this information.
    For each query:
    1. Make it precise and focused
    2. Include key terms and any relevant date ranges
    3. Prioritize finding authoritative sources
    
    Format each query on a new line starting with "QUERY:"
    """
    
    query_text = llm.generate(query_generation_prompt)
    search_queries = extract_queries(query_text)
    
    return search_queries
```

#### Progressive Query Refinement
```python
def refine_search_query(original_query, previous_results, llm):
    """Refine a search query based on previous search results."""
    
    refinement_prompt = f"""
    Original question: {original_query}
    
    Previous search query: {previous_results['query']}
    
    Search results:
    {format_search_results(previous_results['results'])}
    
    Based on these results, create an improved search query that will help answer the original question.
    Consider:
    1. What relevant information is still missing?
    2. Which terms should be added, removed, or modified?
    3. How can the query be made more specific?
    
    Improved search query:
    """
    
    refined_query = llm.generate(refinement_prompt).strip()
    
    return refined_query
```

### 3. Retrieval Integration Framework

Systems for incorporating retrieved information into the generation process:

#### Progressive Context Building
```python
def build_retrieval_context(original_query, search_results, llm):
    """Build a context from multiple search results, extracting only relevant information."""
    
    # Initialize with empty context
    accumulated_context = ""
    
    for i, result in enumerate(search_results):
        # Assess relevance of this result
        relevance_prompt = f"""
        Original question: {original_query}
        
        Current accumulated information:
        {accumulated_context if accumulated_context else "[No information yet]"}
        
        New information:
        {result['content']}
        
        Extract only the information from this new source that:
        1. Is directly relevant to the original question
        2. Adds new information not already covered
        3. Is factual rather than opinions or speculation
        
        Relevant information (be concise):
        """
        
        relevant_content = llm.generate(relevance_prompt)
        
        # Add to accumulated context if non-empty
        if relevant_content and not relevant_content.isspace():
            accumulated_context += f"\nFrom source {i+1} ({result['source']}):\n{relevant_content}\n"
    
    return accumulated_context
```

#### Sufficiency Assessment
```python
def assess_information_sufficiency(original_query, current_context, llm):
    """Determine if enough information has been gathered to answer the query."""
    
    sufficiency_prompt = f"""
    Original question: {original_query}
    
    Information gathered so far:
    {current_context}
    
    Assess whether this information is sufficient to fully answer the original question.
    
    1. Is all necessary information present? Yes/No
    2. What specific information is still missing, if any?
    3. On a scale of 1-5, how complete is the information (where 5 is completely sufficient)?
    
    Analysis:
    """
    
    assessment = llm.generate(sufficiency_prompt)
    
    # Parse assessment
    is_sufficient = "yes" in extract_sufficiency_decision(assessment).lower()
    missing_information = extract_missing_info(assessment)
    completeness_score = extract_completeness_score(assessment)
    
    return {
        "is_sufficient": is_sufficient,
        "completeness_score": completeness_score,
        "missing_information": missing_information,
        "full_assessment": assessment
    }
```

### 4. Retrieval Orchestration

Mechanisms for managing the overall retrieval process:

```python
def retrieval_guided_generation(query, llm, search_engine, max_iterations=3):
    """End-to-end retrieval-guided generation process."""
    
    # Step 1: Assess if external information is needed
    assessment = assess_information_need(query, llm)
    
    # If confident, answer directly
    if not assessment["needs_search"]:
        direct_answer = llm.generate(f"Question: {query}\n\nAnswer:")
        return {
            "answer": direct_answer,
            "searches_performed": 0,
            "retrieval_path": ["Used existing knowledge"]
        }
    
    # Step 2: Identify specific knowledge gaps
    knowledge_gaps = identify_knowledge_gaps(query, llm, domain_classifier)
    
    # Step 3: Generate initial search queries
    search_queries = generate_search_queries(query, knowledge_gaps, llm)
    
    # Initialize tracking variables
    accumulated_context = ""
    search_history = []
    
    # Iterative retrieval process
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        if not search_queries:
            break
            
        # Execute the most promising query
        current_query = search_queries[0]
        search_queries = search_queries[1:]
        
        search_results = search_engine.search(current_query)
        search_history.append({
            "iteration": iteration,
            "query": current_query,
            "results": [r["title"] for r in search_results]
        })
        
        # Extract relevant information
        new_context = build_retrieval_context(query, search_results, llm)
        accumulated_context += new_context
        
        # Assess if we have enough information
        sufficiency = assess_information_sufficiency(query, accumulated_context, llm)
        
        if sufficiency["is_sufficient"]:
            break
            
        # If more information needed, refine queries
        if iteration < max_iterations:
            refined_query = refine_search_query(query, 
                                              {"query": current_query, "results": search_results}, 
                                              llm)
            search_queries.append(refined_query)
    
    # Generate final answer using accumulated context
    final_answer_prompt = f"""
    Question: {query}
    
    Based on the following information:
    {accumulated_context}
    
    Provide a comprehensive answer to the question.
    """
    
    final_answer = llm.generate(final_answer_prompt)
    
    return {
        "answer": final_answer,
        "searches_performed": iteration,
        "retrieval_path": search_history,
        "context_used": accumulated_context
    }
```

## Implementation Approaches

### Basic Implementation

```python
import openai
import json

def retrieval_guided_answer(question, search_function):
    """
    A simple implementation of retrieval-guided generation
    using a two-stage approach.
    """
    # First, ask the model if it needs to search
    assessment_messages = [
        {"role": "system", "content": 
         "You are an assistant that determines when external information is needed."},
        {"role": "user", "content": f"""
         Question: {question}
         
         Do you need to search for external information to answer this question accurately?
         If yes, what specific search query would be most helpful?
         
         Respond in JSON format with two fields:
         1. "need_search": true or false
         2. "search_query": your suggested search query if needed
         """}
    ]
    
    assessment_response = openai.chat.completions.create(
        model="gpt-4",
        messages=assessment_messages,
        response_format={"type": "json_object"}
    )
    
    assessment = json.loads(assessment_response.choices[0].message.content)
    
    # If no search is needed, answer directly
    if not assessment["need_search"]:
        direct_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        direct_response = openai.chat.completions.create(
            model="gpt-4",
            messages=direct_messages
        )
        
        return {
            "answer": direct_response.choices[0].message.content,
            "search_performed": False,
            "explanation": "Answered using existing knowledge."
        }
    
    # If search is needed, perform the search
    search_query = assessment["search_query"]
    search_results = search_function(search_query)
    
    # Generate answer with search results
    search_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"""
         Question: {question}
         
         I searched for: "{search_query}"
         
         Search results:
         {search_results}
         
         Please answer the original question based on this information.
         """}
    ]
    
    search_response = openai.chat.completions.create(
        model="gpt-4",
        messages=search_messages
    )
    
    return {
        "answer": search_response.choices[0].message.content,
        "search_performed": True,
        "search_query": search_query,
        "explanation": "Answered using retrieved information."
    }
```

### Advanced Implementation with LangChain

```python
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.tools import BaseTool
from langchain.callbacks import get_openai_callback
from typing import Dict, List, Optional, Any
import json

class SearchDecisionTool(BaseTool):
    name = "search_decision"
    description = "Determines if a search is needed to answer a question."
    
    def _run(self, query: str) -> Dict[str, Any]:
        llm = ChatOpenAI(temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You determine if a question requires searching for external information."),
            HumanMessage(content=f"""
            Question: {query}
            
            Assess if you need to search for external information to answer this accurately.
            Consider:
            1. Is this about recent events or rapidly changing information?
            2. Does it request specific data or statistics?
            3. Would your answer be significantly improved with current information?
            
            Respond with JSON:
            {{
                "need_search": true/false,
                "confidence": 1-5 scale (5 being very confident),
                "reasoning": "brief explanation of your decision",
                "search_queries": ["query1", "query2"] if search is needed
            }}
            """)
        ])
        
        response = llm.predict_messages(prompt.format_messages())
        return json.loads(response.content)

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information."
    
    def _run(self, query: str) -> str:
        # Implement actual search functionality here
        # This is a placeholder
        return f"Search results for: {query}"

class RetrievalGuidedAgent:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.search_decision_tool = SearchDecisionTool()
        self.web_search_tool = WebSearchTool()
        
        self.tools = [self.web_search_tool]
        self.agent = create_structured_chat_agent(self.llm, self.tools, verbose=True)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True
        )
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        # First, decide if we need to search
        with get_openai_callback() as cb:
            search_decision = self.search_decision_tool._run(question)
            
            if not search_decision["need_search"]:
                # Answer directly without search
                direct_response = self.llm.predict(f"""
                Question: {question}
                
                Please provide a comprehensive answer based on your knowledge.
                """)
                
                return {
                    "answer": direct_response,
                    "search_performed": False,
                    "token_usage": cb.total_tokens
                }
            
            # If search is needed, use the agent executor with search capability
            search_queries = search_decision["search_queries"]
            context = ""
            
            for query in search_queries:
                results = self.web_search_tool._run(query)
                context += f"\nSearch for '{query}':\n{results}\n"
            
            # Use the agent to formulate a final answer with the search results
            final_response = self.agent_executor.run(
                f"""
                Question: {question}
                
                Use the following search results to help answer the question:
                {context}
                """
            )
            
            return {
                "answer": final_response,
                "search_performed": True,
                "search_queries": search_queries,
                "token_usage": cb.total_tokens
            }
```

## Advantages and Limitations

### Advantages

- **Efficiency**: Reduces unnecessary retrievals for questions the model can answer directly
- **Precision**: Generates targeted search queries rather than generic ones
- **Transparency**: Makes the information-seeking process explicit and auditable
- **Adaptability**: Adjusts retrieval depth based on question complexity
- **Reduced Hallucination**: Improves factual accuracy through targeted information gathering
- **Better Resource Utilization**: Minimizes retrieval operations and associated costs

### Limitations

- **Complexity**: More complex to implement than standard RAG
- **Retrieval Quality Dependency**: Effectiveness depends on the model's ability to formulate good queries
- **Overconfidence Risk**: Models may incorrectly believe they know answers without searching
- **Latency Concerns**: Multiple retrieval rounds can increase response time
- **Tool Capabilities**: Limited by the capabilities of available search tools
- **Orchestration Challenges**: Requires careful integration of assessment, retrieval, and generation

## Best Practices

### Self-Assessment Design

- **Explicit Confidence Instructions**: Provide clear guidelines for confidence assessment
- **Domain-Specific Knowledge Boundaries**: Define domain-specific knowledge cutoffs
- **Calibration Examples**: Include examples of when to search vs. when to answer directly
- **Self-critique Prompting**: Encourage the model to critically evaluate its knowledge

```python
# Example confidence assessment prompt
confidence_prompt_template = """
Question: {question}

Before answering, assess your knowledge on this topic:

For questions about general knowledge, historical events before 2021, 
fundamental scientific concepts, or common literary works, you can often 
answer directly.

For questions about current events, recent statistics, specific technical 
details, or rapidly evolving fields, you should seek external information.

Rate your confidence about answering without searching:
1. Very uncertain - I definitely need to search
2. Somewhat uncertain - I should probably search
3. Moderate confidence - I know some information but could benefit from verification
4. Confident - I likely have sufficient knowledge
5. Very confident - I definitely have accurate information

Also explain specifically what information you would need to search for, if any.

Your assessment:
"""
```

### Query Formulation

- **Query Specificity**: Generate focused rather than broad queries
- **Diversity in Retrieval**: Create multiple queries for different aspects of complex questions
- **Temporal Awareness**: Include date ranges for time-sensitive information
- **Source Consideration**: Target authoritative sources when appropriate
- **Iterative Refinement**: Improve queries based on previous search results

```python
# Example query formulation strategies for different question types

query_strategies = {
    "factual": {
        "template": "{entity} {attribute} {time_frame}",
        "example": "tokyo population 2023 official"
    },
    "comparison": {
        "template": "{entity1} vs {entity2} {attribute} {time_frame}",
        "example": "tesla model 3 vs nissan leaf range 2023"
    },
    "how_to": {
        "template": "how to {action} {subject} step by step",
        "example": "how to install python on windows 11 step by step"
    },
    "explanation": {
        "template": "{concept} explanation {level}",
        "example": "quantum entanglement explanation simple"
    }
}
```

### Retrieval Integration

- **Source Attribution**: Maintain clear links between information and sources
- **Information Synthesis**: Combine information from multiple sources coherently
- **Contradiction Handling**: Develop strategies for conflicting information
- **Context Management**: Optimize use of the context window for relevant information
- **Progressive Summarization**: Summarize accumulated information to manage context size

### End-to-End Process

- **Maximum Iterations**: Set reasonable limits on retrieval iterations
- **Fallback Mechanisms**: Have graceful fallbacks when retrieval fails
- **User Involvement**: Consider when to involve users in the retrieval process
- **Logging and Tracing**: Maintain detailed logs of the retrieval path
- **Evaluation Metrics**: Develop metrics specific to retrieval-guided generation

## Examples

### Historical Research Question

```
User: How did the economic policies of the Tang Dynasty influence medieval European trade networks?

System: [SELF-ASSESSMENT]
This question requires specialized historical knowledge about:
1. Tang Dynasty economic policies (7th-10th centuries)
2. Medieval European trade networks of the same era
3. The specific influences between these distant regions
My confidence is 2/5 (Somewhat uncertain). While I have general knowledge about both the Tang Dynasty and medieval European trade, I lack specific information about their economic interconnections.

System: [QUERY FORMULATION]
I'll search for: "Tang Dynasty economic policies influence on medieval European trade silk road"

System: [INFORMATION RETRIEVAL]
Retrieved key information:
- The Tang Dynasty (618-907 CE) established policies promoting the Silk Road trade
- Tang China exported silk, porcelain, and spices westward
- Byzantine and later Italian merchants were the primary European receivers
- Trade was largely indirect, passing through Central Asian, Persian, and Arab intermediaries
- Tang silver mining and standardized coinage affected valuations along the entire route
- Chinese paper-making technology gradually moved westward during this period

System: [SUFFICIENCY ASSESSMENT]
I have good information on the trade routes and general connections, but need more specifics on economic policies and their direct effects.

System: [QUERY REFINEMENT]
I'll search for: "Tang Dynasty maritime trade policy foreign merchants European goods imported"

System: [ADDITIONAL RETRIEVAL]
Retrieved additional information:
- Tang established special foreign trade offices called "Sitianfang"
- Tang policies allowed foreign merchant communities in major cities 
- European goods reaching Tang China included glassware, amber, and textiles
- The An Lushan Rebellion (755-763) disrupted land routes, shifting emphasis to maritime trade
- Tang taxation policies on luxury goods influenced pricing structures across Eurasia

System: [FINAL ANSWER]
The economic policies of the Tang Dynasty (618-907 CE) influenced medieval European trade networks in several significant ways, despite the vast geographical separation between these regions.

First, the Tang government established formal trade institutions, including the "Sitianfang" (Foreign Trade Offices), which regulated commerce with foreign merchants. This administrative structure created stable trading conditions that reverberated throughout the Silk Road network, ultimately affecting European merchants operating in the eastern Mediterranean.

Second, Tang monetary policies, particularly their standardized silver coinage and government-controlled minting, created reliable currency standards that influenced valuations of goods across Eurasia. This monetary stability facilitated trade across long distances, benefiting European merchants who otherwise had to navigate multiple currency systems.

Third, the Tang Dynasty implemented policies allowing foreign merchant communities to establish themselves in major Chinese cities like Chang'an and Guangzhou. These policies created multicultural commercial hubs where trading techniques, financial instruments, and market knowledge were exchanged, eventually filtering westward to Europe through intermediaries.

Fourth, when the An Lushan Rebellion (755-763) disrupted the overland Silk Road, Tang policies shifted to promote maritime trade routes. This pivotal change accelerated the development of Indian Ocean trade networks that would eventually connect to Mediterranean shipping lanes, fundamentally altering how European goods reached Asian markets.

Finally, Tang luxury taxation policies affected pricing structures across the entire Silk Road, influencing which high-value goods (like silk, spices, and porcelain) were profitable enough to justify the long journey to European markets.

Although direct contact between Tang China and medieval Europe was minimal, with Persian, Arab, and Byzantine merchants serving as intermediaries, the economic ripple effects of Tang policies created the commercial frameworks that would later enable more direct European-Asian trade during the Mongol Empire and beyond.
```

### Technical Troubleshooting Question

```
User: My Docker containers keep restarting with "Exit 137" errors. How can I fix this?

System: [SELF-ASSESSMENT]
This is a technical troubleshooting question about Docker containers exiting with code 137. I know that exit code 137 typically indicates that the container was killed due to running out of memory (OOM - Out Of Memory). However, to provide a comprehensive answer with current best practices, I should search for specific troubleshooting steps and solutions.
Confidence: 3/5 (Moderate confidence)

System: [QUERY FORMULATION]
I'll search for: "docker container exit 137 out of memory troubleshooting solutions"

System: [INFORMATION RETRIEVAL]
Retrieved information:
- Exit code 137 confirms this is an Out Of Memory (OOM) issue
- Common causes: insufficient memory limits, memory leaks, sudden spikes in workload
- Solutions include: increasing container memory limits, optimizing application memory usage, monitoring memory consumption

System: [SUFFICIENCY ASSESSMENT]
I have good general information but need more specific diagnostic steps and configuration examples.

System: [QUERY REFINEMENT]
I'll search for: "docker container memory limits configuration docker-compose swarm kubernetes"

System: [ADDITIONAL RETRIEVAL]
Retrieved additional information:
- Docker run: use --memory and --memory-swap flags
- Docker Compose: use mem_limit and memswap_limit in compose file
- Docker Swarm: use resources.limits.memory in service definition
- Kubernetes: use resources.limits.memory in pod spec
- Memory monitoring commands: docker stats, kubectl top pods

System: [FINAL ANSWER]
Your Docker containers are exiting with code 137, which specifically indicates they're being terminated by the kernel's Out-Of-Memory (OOM) killer. This happens when a container tries to use more memory than is available or allowed. Here's how to fix it:

1. **Diagnose the memory issue**:
   - Check current memory usage: `docker stats`
   - Look at container logs for memory-related messages: `docker logs [container_id]`
   - Check if your host system has enough available memory: `free -h`
   - Review application logs for memory spikes or leaks

2. **Increase memory limits** based on your environment:
   - **Docker run**: Add or increase memory limits
     ```
     docker run --memory=2g --memory-swap=2g your_image
     ```
   
   - **Docker Compose**: Modify your docker-compose.yml
     ```yaml
     services:
       your_service:
         image: your_image
         deploy:
           resources:
             limits:
               memory: 2G
     ```
   
   - **Docker Swarm**: Update your service definition
     ```
     docker service update --limit-memory 2G your_service
     ```
   
   - **Kubernetes**: Update pod resource limits
     ```yaml
     resources:
       limits:
         memory: "2Gi"
       requests:
         memory: "1Gi"
     ```

3. **Optimize your application**:
   - Monitor memory usage patterns to identify leaks
   - Add garbage collection tuning if it's a Java/Node.js application
   - Consider implementing memory-efficient algorithms or data structures
   - Break monolithic applications into smaller services with individual limits

4. **Implement proper monitoring**:
   - Set up alerts for when containers approach memory limits
   - Use tools like Prometheus and Grafana to visualize memory usage trends
   - Configure health checks to detect potential issues before OOM occurs

5. **Check for specific application problems**:
   - Some applications need specific environment variables to control memory usage
   - For Java applications, you may need to set `-Xmx` and `-Xms` flags
   - For Node.js, consider the `--max-old-space-size` flag

Start by increasing the memory limits to a reasonable value based on your application needs, then monitor and optimize as necessary. The right approach balances giving your application enough memory to function properly while not over-allocating resources.
```

## Further Reading

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

2. Gao, L., et al. (2023). RAFT: Adapting Language Model to Domain Specific RAG. [arXiv:2310.11689](https://arxiv.org/abs/2310.11689)

3. Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

4. Zheng, L., et al. (2023). Adaptive Retrieval-Augmented Generation. [arXiv:2305.06991](https://arxiv.org/abs/2305.06991)

5. Jiang, Z., et al. (2023). Active Retrieval Augmented Generation. [arXiv:2305.06983](https://arxiv.org/abs/2305.06983)

6. Nakano, R., et al. (2021). WebGPT: Browser-assisted question-answering with human feedback. [arXiv:2112.09332](https://arxiv.org/abs/2112.09332)

7. Peng, B., et al. (2023). Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback. [arXiv:2302.12813](https://arxiv.org/abs/2302.12813)