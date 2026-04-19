# Databricks notebook source
# #Objective: 
# Create an Agent that can process unstructured data to their specifications
#
# Broad Ideas: 
#
# 1. Process a PDF and convert to structured Data
# 2. Create Vector Search Index off of it 
# 3. Use Genie Spaces to link data 
# 4. Identify necessary information 
# 5. Define a function to create necessary visuals or take necessary actions
#
# You agent should do the following: 
# 1. Be able to hand off to another Agent 
# 2. Have access to multiple tools 
# 3. Demonstrate switching between LLMs 
# 4. Use a combination of Python Code and calls to the Agent to improve your answer 
#
# UIs are not necessary but highly encouraged

# COMMAND ----------

# #Don't have an idea? 
#
# Here's one: 
#
# Goal: Make an agent that uses a genie space to query stock data when dates on when the stock price dropped. You want the agent to use this structured data to query an external web search tool (hosted as a UC function) based on this information. 
#
# If possible, try to spin up a vector search index. 
#
# Some data is provided for you below to use in a genie space. 
#
# There is also an example python function to call a UC function

# COMMAND ----------

# %pip install --upgrade dspy openai litellm "mlflow[databricks]>=3.1.0" "databricks-connect>=16.1" unitycatalog-ai[databricks] databricks-sdk databricks-vectorsearch databricks-agents databricks-dspy
# dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd

df = pd.read_csv("./financial.csv")
spark_df = spark.createDataFrame(df)
spark_df.write.format("delta").mode("overwrite").option("delta.columnMapping.mode", "name").saveAsTable('you_delta_table here')

# COMMAND ----------

from unitycatalog.ai.core.databricks import DatabricksFunctionClient

def web_search(query):
  """this tool is to query a function to query the web"""
  
  client = DatabricksFunctionClient(execution_mode="local")
  result = client.execute_function(
    "catalog.schema.your_uc_function",
    parameters={"query": query}
  )
  return result.value

# COMMAND ----------

# #Recommendations for ETL 
# 1. Use ai_parse_document for PDF
# Use Databricks' ai_parse_document function to automatically extract text, tables, and structured data from your PDF files. This AI-powered tool understands document layout and converts PDFs into clean, structured JSON format. It works much better than basic text extraction for complex documents.
# 2. Store Information into Delta Tables. We need existing Delta Tables to write back to existing tables. 
# Take the parsed PDF data and save it into a Delta Table, which is Databricks' optimized storage format. Delta Tables provide reliable data storage with features like automatic versioning and schema management. This becomes your clean, queryable data source for the next steps.
# 3. Create Vector Search Index
# Convert your text data into vector embeddings and create a searchable index that enables semantic search. This allows you to find documents based on meaning rather than just keywords. The vector index is essential for building AI applications like chatbots or document Q&A systems.
# 4. There are some provided notebooks that you can use to parse PDFs more quickly and reliably than ai_parse_document if you wish
#

# COMMAND ----------

# #Section 2: Functions
# 1. Create X amount of managed functions to complete task. This can be Genie Spaces, Model serving Endpoints, Agent Bricks
# 2. Create regular python functions that can execute code and create visuals. 
# 3. Optional: Try out Managed MCP and incorporate that as a tool

# COMMAND ----------

# #Section 3: Deployment
#
# 1. Use Agents.deploy to deploy your agent to an agent endpoint 
# 2. Make sure to install mlflow 3.0 to take advantage of the latest experiment and traces tracking. 
# 3. (Optional) Deploy to a Databricks apps UI
#
# Databricks is taking a model as code approach for deploying agents since there are so many difference pieces that can be defined in many different ways. For maximum compatibility, we recommend making a agent.py file to deploy as a model. 
#
# The workflow is shown below. You will take advantange of the typical mlflow capabilities to deploy this

# COMMAND ----------

# %%writefile agent.py
#
# from typing import Any, Generator, Optional
# from databricks.sdk.service.dashboards import GenieAPI
# import mlflow
# from databricks.sdk import WorkspaceClient
# from mlflow.entities import SpanType
# from mlflow.pyfunc.model import ChatAgent
# from mlflow.types.agent import (
#     ChatAgentMessage,
#     ChatAgentResponse,
#     ChatContext,
# )
# import dspy
# import uuid
#
# # Autolog DSPy traces to MLflow
# mlflow.dspy.autolog()
#
# # Set up DSPy with a Databricks-hosted LLM
# LLM_ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"
# lm = dspy.LM(model=f"databricks/{LLM_ENDPOINT_NAME}")
# dspy.settings.configure(lm=lm)
#
# ######################################
# ## Create our Signature. Make as many as you need
# ######################################
# class genie_selector_agent(dspy.Signature):
#   """
#   Given the sql_instructions, determine which genie space tool to call, send the exact sql_instruction text to the tool and answer the question given the response from the tool.
#   """ 
#   sql_instruction: str = dspy.InputField()
#   response: str = dspy.OutputField() 
#   sql_query_output:  list = dspy.OutputField()
#
# ######################################
# ## Create custom modules 
# ######################################
#
# #this is entire up to you if you want to prepackage some modules together to collective complete a task
#
# ######################################
# ## Create our ChatAgent. This is an object MLflow needs to recognize what kind of model this is. You'll notice its design is very similar to a custom module
# ######################################
#
# class DSPyChatAgent(ChatAgent):     
#     def __init__(self): #instantiate the agents or signatures that you need
#       self.genie_selector_agent = genie_selector_agent
#       self.multi_genie_agent = dspy.ReAct(self.genie_selector_agent, tools=[self.hls_patient_genie, self.investment_portfolio_genie], max_iters=1)
#
#     ######################################
#     ## Define your tools below within the ChatAgent so that the class knows these exist. You can define these outside the class as well if you like
#     ######################################
#
#     def hls_patient_genie(self, sql_instruction):
#
#       w = WorkspaceClient()
#       genie_space_id = "01effef4c7e113f9b8952cf568b49ac7"
#
#       # Start a conversation
#       conversation = w.genie.start_conversation_and_wait(
#           space_id=genie_space_id,
#           content=f"{sql_instruction} always limit to one result"
#       )
#
#       response = w.genie.get_message_attachment_query_result(
#         space_id=genie_space_id,
#         conversation_id=conversation.conversation_id,
#         message_id=conversation.message_id,
#         attachment_id=conversation.attachments[0].attachment_id
#       )
#
#       return response.statement_response.result.data_array
#
#     def investment_portfolio_genie(self, sql_instruction):
#
#       w = WorkspaceClient()
#       genie_space_id = "01f030d91cc6165d88aaee122a274294"
#
#       # Start a conversation
#       conversation = w.genie.start_conversation_and_wait(
#           space_id=genie_space_id,
#           content=f"{sql_instruction} always limit to one result"
#       )
#
#       response = w.genie.get_message_attachment_query_result(
#         space_id=genie_space_id,
#         conversation_id=conversation.conversation_id,
#         message_id=conversation.message_id,
#         attachment_id=conversation.attachments[0].attachment_id
#       )
#
#       return response.statement_response.result.data_array
#   
#     #very basic memory implementation
#     def prepare_message_history(self, messages: list[ChatAgentMessage]):
#         history_entries = []
#         # Assume the last message in the input is the most recent user question.
#         for i in range(0, len(messages) - 1, 2):
#             history_entries.append({"question": messages[i].content, "answer": messages[i + 1].content})
#         return dspy.History(messages=history_entries)
#
#     ######################################
#     ## This predict method is where the interaction first starts. If you want to change what happens here, you can. It must return ChatAgentResponse to be compatible with agents.deploy
#     ######################################
#     @mlflow.trace(span_type=SpanType.AGENT)
#     def predict(
#         self,
#         messages: list[ChatAgentMessage],
#         context: Optional[ChatContext] = None,
#         custom_inputs: Optional[dict[str, Any]] = None,
#     ) -> ChatAgentResponse:
#         latest_question = messages[-1].content
#         response = self.multi_genie_agent(sql_instruction=latest_question).response
#         return ChatAgentResponse(
#             messages=[ChatAgentMessage(role="assistant", content=response, id=uuid.uuid4().hex)]
#         )
#
# # Set model for logging or interactive testing
# from mlflow.models import set_model
# AGENT = DSPyChatAgent()
# set_model(AGENT)

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# ### Log your Agent with passthrough authentication

# COMMAND ----------

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Pydantic V1 functionality isn't compatible with Python 3\.\d+.*",
)
import mlflow
from agent import LLM_ENDPOINT_NAME
from mlflow.models.resources import (
    DatabricksFunction,
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksVectorSearchIndex
)
from pkg_resources import get_distribution


# TODO : set the genie_space_id for each Genie Space you want to call
resources = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT_NAME),
    DatabricksGenieSpace(genie_space_id= "01f0357a714f14b39ec53dfeb7c916b5"),
    DatabricksGenieSpace(genie_space_id= "01f0357a519d17cd96ad784b8afce762"),
    DatabricksVectorSearchIndex(index_name="jai_behl.ias.knowledge_base")
]

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model="agent.py",
        # input_example=input_example,
        extra_pip_requirements=[f"databricks-connect=={get_distribution('databricks-connect').version}"],
        resources=resources,
    )

# COMMAND ----------

# ### Register your Agent to Unity Catalog

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model.
catalog = "jai_behl"
schema = "ias"
model_name = "dspy_multi_genie"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# #Dev Time

# COMMAND ----------

