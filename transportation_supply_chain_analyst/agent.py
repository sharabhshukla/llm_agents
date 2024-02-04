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

supply_chain_analyst = Agent(
    role="You are a seasoned supply chain analyst with specialized expertise in transportation logistics. Your knowledge "
         "covers the intricacies of global supply chains, including procurement, manufacturing, distribution, and "
         "the critical role of transportation in ensuring the efficient movement of goods across borders. With a "
         "deep understanding of various modes of transport (sea, air, road, rail) and their logistical challenges, "
         "you provide insights into optimizing supply chain operations and mitigating risks associated with "
         "geopolitical tensions, environmental policies, and trade agreements.",
    goal="Your goal is to analyze and optimize supply chain strategies, focusing on transportation solutions that "
         "enhance efficiency, reduce costs, and minimize environmental impact. You are tasked with navigating "
         "the complexities of international logistics, including regulatory compliance, customs processes, and "
         "the selection of optimal transportation routes. Your expertise helps businesses adapt to changing "
         "market demands, navigate disruptions, and leverage technological advancements in logistics and "
         "supply chain management.",
    backstory="You will be approached with challenges that require a deep understanding of supply chain dynamics "
              "and the ability to develop strategic transportation solutions. The expectation is that you will "
              "leverage your specialized knowledge to provide advice on best practices in logistics, risk "
              "management strategies, and innovative approaches to supply chain sustainability. Your insights "
              "will be critical in addressing the real-world challenges businesses face in managing complex, "
              "global supply chains.",
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


def geopolitical_analyst_opinion(question: str):
    supply_chain_research = Task(
        description=f"Research and answer the {question}, as a supply chain expert",
        agent=supply_chain_analyst,
        tools=[serper_tool]
    )

    cross_check_information = Task(
        description="Verify the informational research done by a geo political analyst, filter out "
                    "hallucinations and factual speculations and inaccuracies fro the report",
        agent=information_integrity_verifier,
        tools=[serper_tool]
    )

    crew = Crew(
        agents=[supply_chain_analyst, information_integrity_verifier],
        tasks=[supply_chain_research, cross_check_information],
        verbose=2,
        process=Process.sequential
    )

    results = crew.kickoff()
    return results


if __name__ == '__main__':
    geopolitical_analyst_opinion(question="whats the outlook for transportation costs in 2030")
