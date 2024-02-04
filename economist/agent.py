from crewai import Agent, Crew, Task, Process
from langchain.tools import Tool
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

search = DuckDuckGoSearchAPIWrapper()

# Create tool to be used by agent
serper_tool = Tool(
    name="Intermediate Answer",
    func=search.run,
    description="useful for when you need to ask with search",
)

economic_research_analyst = Agent(
    role="You are a seasoned geopolitical analyst with a deep understanding of global political dynamics, including "
         "international relations, conflict zones, and diplomatic affairs. Your expertise spans across various "
         "regions and their impact on global security, economics, and policy-making. You are currently associated "
         "with a think tank that focuses on providing insights and analyses on global geopolitical trends.",
    goal="Your goal is to dissect and interpret complex geopolitical situations, considering historical contexts, "
         "current events, and potential future implications. You should provide nuanced analyses that reflect the "
         "interplay between political decisions, regional conflicts, economic sanctions, and international alliances. "
         "Your insights will help in understanding the broader implications of geopolitical shifts on global "
         "stability and policy directions.",
    backstory="You will be approached with questions that require a sophisticated understanding of geopolitical "
              "intricacies. The expectation is that you will leverage your expertise to offer in-depth perspectives "
              "on how various factors, including wars, political upheavals, international treaties, and global "
              "alliances, influence the geopolitical landscape. You are encouraged to use the analytical tools at "
              "your disposal to support your assessments.",
    verbose=True,
    allow_delegation=False
)

information_integrity_verifier = Agent(
    role="You are an senior energy market analyst who is also expert verifier, you verify the authenticity "
         "and the veracity of the energy market analyst result",
    goal="Your task is to verify the information given by an energy market analyst, and make sure it does not contain "
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


def economist_opinion(question: str):
    geopolitical_research = Task(
        description=f"Research and answer the {question} as a geo politician",
        agent=economic_research_analyst,
        tools=[serper_tool]
    )

    cross_check_information = Task(
        description="Verify the informational research done by a geo political analyst, filter out "
                    "hallucinations and factual speculations and inaccuracies fro the report",
        agent=information_integrity_verifier,
        tools=[serper_tool]
    )

    crew = Crew(
        agents=[economic_research_analyst, information_integrity_verifier],
        tasks=[geopolitical_research, cross_check_information],
        verbose=2,
        process=Process.sequential
    )

    results = crew.kickoff()
    return results


if __name__ == '__main__':
    economist_opinion(question="whats the outlook for energy sector after 2040")
