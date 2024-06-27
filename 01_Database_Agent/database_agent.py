
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
import os
import pandas as pd

#client = ChatGroq()
GROQ_API_KEY = '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'
#os.environ["GROQ_API_KEY']= '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'
#Api_key= os.getenv('GROQ_API_KEY')
# model = ChatGroq(api_key=GROQ_API_KEY)

# importing for setting the langchain agent
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent



#load the data
file_dir = "./data/all-states-history.csv"
df = pd.read_csv(file_dir).fillna(value=0)

model = ChatGroq(api_key=GROQ_API_KEY, temperature=0, max_tokens = 500)
# create pandas dataframe agent
agent = create_pandas_dataframe_agent(llm= model, df=df, verbose=False, allow_dangerous_code= True, )


# # crafting our own prompt
# CSV_PROMPT_PREFIX = """
# First set the pandas display options to all the columns,
# get the column names, then answer the question. 
# """
# CSV_PROMPT_SUFFIX = """
# - **ALWAYS** before giving the final Answer, try another method.
#  Then reflect on the answers of the two methods you did and ask yourself
#  if it answers correctly the original question.
#  If you are not sure, try another method.
#  - If the methods tried do not give the same result, reflect and try again
#  untill you have two methods that have the same result.
#  - If you still cannot arrive to a consistent result, say that
#  you are not sure of the answer.
#  If you are sure of the correct answer, create a beautiful and thorough response
#  using Markdown.
#  - **Do NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
#  ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**
#  - **ALWAYS**  as part of your "Final Answer", explain how you got to the answer
#  on a section start with: "\n\nExplanation:\n". 
#  In the explanation, mention the column names that you used to get 
#  to the final answer.
# """

# QUESTION = "How many patients were hospitalized during July 2020"
# "in Texas, and nationwide as the total of all states?"
# "Use the hospitalizedIncrease column"
# #calling the agent for inference
# try:
#     #response = agent.invoke("How many rows are there")
#     response = agent.invoke(CSV_PROMPT_PREFIX+QUESTION+CSV_PROMPT_SUFFIX)
#     # print(response)
# except Exception as e:
#     print("output parsing error occurred")
#     print(e)

# message = HumanMessage(
#     content = " Give me five breif sentences about running excercise"
# )

# print(model.invoke([message]))
# print(response)



# Creating SQL Agent
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from sqlalchemy import create_engine

database_file_path = "./db/test.db"

engine = create_engine(f'sqlite:///{database_file_path}')
df.to_sql(
    'all_states_history',
    con =engine,
    if_exists= 'replace',
    index=False
)

MSSQL_AGENT_PREFIX = """
you are an agent designed to iteract with SQL database.

## Instructions:
- Given aninput question, create a systacticallyt correct {dialect} query
to run, then look at the results of the query and return the answer.
- unless the user specifies a specfic number of examples they wish to
obtain, **ALWAYS** limit your query to at most {top_k} results.
- You can order the results by a relevant column to return the most
interesting examples in the database.
- Never query for all the columns from a specific table, only ask for
the relevant columns given the question.
- You have access to tools for interacting with the database.
- You MUST double check your query before executing it. If you get an error 
while executing a query, rewrite the query and try again.
- Do NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, etc.)
to the database.
- Do NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE, ONLY USE THE RESULTS
OF THE CALCULATIONS YOU HAVE DONE.
- YOur response should be in Markdown. However, **when running a SQL Query
in "Action Input", do not include the markdown backticks**.
Those are only for formatting the response, not for executing the commands.
- ALWAYS, as part of your final answer, explain how you got to the answer
on a section that starts with: "Explanation:". Include the SQL query as part
of the explanation section.
- If the question does not seem related to the database, just return
"I don\'t know" as the answer.
- Only use the below tools. Only use the information returned by the 
below tools to construct your query and final answer.
- Do not make up table names, only use the tables returned by any of the 
tools below.

##Tools:

"""




MSSQL_AGENT_FORMAT_INSTRUCTIONS = """
Question: the input question you must answer.
Thought: you should always think what to do.
Action: the action to take, should be on of [{tool_names}].
Action Input: the input to the action.
Observation: the result of the action.
...(this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer.
Final Answer: the final answer to the original input question.

Example of Final Answer:
<=== Beginning of example

Action: query_sql_db
Action: Input:
SELECT TOP (10) [death]
FROM covidtracking
WHERE state = 'TX' AND date LIKE '2020%'

Observation:
[(27437.0,), (27088.0,), (26762.0,), (26521.0,), (26472.0,), (26421.0,), (26408.0,)]
Thought: I now know the final answer
Final Answer: There were 27437 people who died of covid in Texas in 2020.


Explanation:
I queried the 'covidtracking' table for the 'death' column where the state 
is 'TX' and the datae starts with '2020'. The query returned a list of tuple 
with the number of deaths for each day in 2020. To answer the question, 
I took the sum of all the deaths in the list, which is 27437.
I used the following query

'''sql
SELECT[death] FROM covidtracking WHERE state = 'TX' AND  date LIKE '2020'"

===> End of Example
"""


db = SQLDatabase.from_uri(f'sqlite:///{database_file_path}')
toolkit = SQLDatabaseToolkit(db=db, llm= model)


QUESTION = """ How many patients were hospitalized during october 2020
in New York, and nationwide as the total of all states?
Use the hospitalizedIncrease column

"""

agent_executor_SQL = create_sql_agent(
    prefix = MSSQL_AGENT_PREFIX,
    format_instructions= MSSQL_AGENT_FORMAT_INSTRUCTIONS,
    llm = model, 
    toolkit = toolkit, 
    top_k = 20,
    verbose = True
)


response = model.invoke(Question)
print(response['output'])