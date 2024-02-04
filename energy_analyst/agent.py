from crewai import Agent, Crew, Task, Process
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool

search = DuckDuckGoSearchAPIWrapper()

# Create tool to be used by agent
serper_tool = Tool(
  name="Intermediate Answer",
  func=search.run,
  description="useful for when you need to ask with search",
)

energy_analyst = Agent(
    role="You are an expert energy market analyst specializing in all forms of energy Conventional sources like Oil "
         "and Gas,"
         "as as well as the renewable energy sources. You work at a energy trading firm",
    goal="Your goal is to answer question carefully analyzing the impact of current situation including geo-political,"
         "energy trading, answer question after critically understanding the volatility of energy markets",
    backstory="You will be asked question and the expectation will be that you will provide great insights on how "
              "wars, political situations or"
              "even intricacies and dependency of multiple factors can affect volatility of  energy markets."
              "You are allowed to use the tools at your disposal",
    verbose=True,
    allow_delegation=False
)

information_integrity_verifier = Agent(
    role="You are an senior energy market analyst who is also expert verifier, you verify the authenticity "
         "and the veracity of the energy market analyst result",
    goal="Your task is to verify the information giev by an energy market analyst, and make sure it does not contain "
         "factual inaccuracies, or"
         "hallucinations or made up facts. Also, make sure to correct any speculations in the infomration given by the "
         "analyst. Your job is to"
         "go through the reports submitted by a market analyst and submit your own findings. In some ways your are an "
         "sophisticated proof reader of the"
         "reports submitted by a market analyst. You are allowed to use the tools at your disposal. "
         "return the report only, do not add your comments, only edit the report if needed",
    backstory="You are an senior energy market analyst who is also expert verifier, you verify the authenticity "
              "and the veracity of the energy market analyst result. The anlyst often halucinates or speculates and is "
              "prone to giving out speculations in order to overstate trading revenues",
    verbose=True,
    allow_delegation=False,
)

def energy_analyst_opinion(question: str):
    energy_market_research = Task(
        description=f"Research and answer the {question}",
        agent=energy_analyst,
        tools=[serper_tool]
    )

    cross_check_information = Task(
        description="Verify the informational research done by a energy market analyst, filter out "
                    "hallucinations and factual speculations and inaccuracies fro the report",
        agent=information_integrity_verifier,
        tools=[serper_tool]
    )

    crew = Crew(
        agents=[energy_analyst, information_integrity_verifier],
        tasks=[energy_market_research, cross_check_information],
        verbose=2,
        process=Process.sequential
    )

    results = crew.kickoff()
    return results


if __name__ == '__main__':
    energy_analyst_opinion(question="whats the outlook for energy sector after 2040")
