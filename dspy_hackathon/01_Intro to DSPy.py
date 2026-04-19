# DSPy introduction tutorial (plain Python script)
# Introduction DSPy

# Working with Agents doesn't have to be difficult. We introduce DSPy, a modular, declarative and lightweight framework 
# that helps you organize and optimize your DSPy program.

# In this hackathon, you'll learn how to use DSPy, create your own agents, and practice core DSPy patterns with a Gemini-backed language model.

# Run: pip install --upgrade dspy==3.0.0b4 git+https://github.com/BerriAI/litellm.git openai python-dotenv "mlflow>=3.1.0"

# Restart the Python kernel after installing packages so imports resolve (e.g. Jupyter: Kernel > Restart).

# Setup:

# Let's first configure an LLM to use

# DSPy maintains the global configuration through internal context management. The configuration is stored in DSPy's internal state and accessed by modules when they need to make API calls.

# You can explicitly state which LLM to use for each LLM call but we will see that implementation later.

import os
import warnings
from pathlib import Path

import dspy

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Pydantic V1 functionality isn't compatible with Python 3\.\d+.*",
)
import mlflow
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

_arxiv_path = Path(__file__).resolve().parent / "dataset" / "arxiv_paper_attention.txt"
arxiv_paper = _arxiv_path.read_text(encoding="utf-8")


def gemini_llm(model, cache=False):
    return dspy.LM(model, api_key=os.environ["GOOGLE_API_KEY"], cache=cache)


lm = gemini_llm("gemini/gemini-2.0-flash")
dspy.configure(lm=lm)
mlflow.dspy.autolog() #necessary for mlflow traces!


# Section 1: DSPy Inline signatures

# DSPy signatures are a way to define the input-output interface for language model operations in a structured, type-safe manner. Think of them as function signatures, but specifically designed for working with language models. The signature acts as a blueprint that DSPy uses to construct appropriate prompts and parse responses from the language model.

# A signature describes what you want your language model to do by specifying:

# 1. Input fields: What information goes in
# 2. Output fields: What you want to get back
# 3. Instructions: How the model should process the inputs

# ###Why Use Signatures?

# 1. **Structure**: They force you to think clearly about what inputs you need and what outputs you expect
# 2. **Reusability**: Once defined, signatures can be used with different DSPy modules
# 3. **Optimization**: DSPy can automatically optimize prompts based on your signature definition
# 4. **Type Safety**: They provide a clear contract for what data flows through your pipeline

# Below is an example of how to use DSPy in its simplest form, **Inline DSPy Signatures**


math_problem = dspy.Predict("math_problem -> answer")
print(math_problem(math_problem="what is one plus one?"))


# ### Task 1:
# Task: Try to create a Q/A module using DSPy.Predict using the **Inline DSPy Signatures** method


qna = dspy.Predict('question: str -> answer: str')
print(qna(question='what is the number of life'))


# ###How does this all work?

# Under the hood, if you're interested:

# DSPy processes inline signatures (like "question -> answer") through several transformation steps before sending them to the LLM. Here are the steps:

# 1. **Signature Parsing**: When DSPy encounters "question -> answer", it parses this string to identify:

#   - Input fields: question
#   - Output fields: answer
#   - Field types: Both default to dspy.InputField() and dspy.OutputField()

# 2. **Prompt Template Generation**: DSPy converts the signature into a structured prompt template. For "question -> answer", it might generate something like:

# `Question: {question}`

# `Answer: [to be filled by LLM]`

# 3. **Field Processing**: Each field gets processed with:

#   - Field names: Converted to human-readable labels
#   - Descriptions: Added if provided (defaults to field name)
#   - Formatting: Applied based on field type

# 4. **Instruction Generation**: DSPy automatically generates instructions based on the signature structure

# 5. **Full Prompt Assembly**: The final prompt sent to the LLM combines:

#   - System instructions
#   - Field descriptions
#   - Input values
#   - Output format expectations

#     For example:

#     `Answer questions with short factual answers.`

#     `---`

#     `Follow the following format.`

#     `Question: ${question}`
#     `Answer: ${answer}`

#     `---`

#     `Question: What is the capital of France?`
#     `Answer:`

# DSPy then puts the question provided as the input and the answer from the LLM into answer

# 6. **Response Parsing**: After the LLM responds, DSPy:

# - Extracts the answer from the response
# - Maps it to the output field (answer)
# - Returns a structured object with the result

# DSPy's modules like `dspy.Predict` handles this process for us


# While not shown in the example above, we can add typing to enforce and structure our inputs and outputs.

# We can use typing to guide our inputs and outputs. We let DSPy handle the structuring instead of enforcing it in a long prompt.

# Let's try it with our Math example. In this example, we saw the LLM give a string response. But, since it's a math problem, we want to see an integer rather than a text response.


math_problem = dspy.Predict("math_question: str -> answer: int") #all we do is add the typing we want next to the input and output name 
print(math_problem(math_question="what is one divided by one?"))


# Great! Now we can see the LLM is only giving us a number. This is very handy when we need to send the LLM multiple inputs and expect multiple ouputs. We can now programatically retrieve the outputs.

# Take a look at the example below where we add an input_number to our math_question just to add another input. Then, I add a math_reasoning output as a string to see why the LLM came up with that specific answer but I want it as a string. For good measure, I added one more output called "correct" that's a boolean.

# This separation allows us to focus on programming the prompts for the LLM, instead of trying to write a long prompt to get what we want


#notice how the output changes when I change the typing of correct. Currently it's a string
math_problem = dspy.Predict("input_number: int, math_question: str -> answer: int, math_reasoning: str, correct: str")

result = math_problem(input_number=1, math_question="add 100")
print(result)
result.correct


#notice how the output change for correct when I change the typing to boolean

math_problem = dspy.Predict("input_number: int, math_question: str -> answer: int, math_reasoning: str, correct: bool")  
result = math_problem(input_number=1, math_question="add 100")
print(result)
result.correct


# Then, if I want to access the three outputs, I can access the values of each output using attribute access from the dspy.Prediction object


print(f"This is the answer: {result.answer}\n")
print(f"This is the math_reasoning: {result.math_reasoning}\n")
print(f"This is the boolean: {result.correct}\n")


# Convenient right?

# Try it yourself with your QnA signature!

# ### Task 2:
# Update your QnA Inline Signature to do the following:
# 1. Additional input called context (provided below)
# 2. Additional output called answer_confidence_score that should give us an integer of confidence.
# 3. Use attribute access to access the confidence score and simply divide it by 10


# Sample context for RAG-style QnA (fictitious company)
context = """Example Analytics, Inc. builds cloud software for data pipelines and machine learning. Founded in 2015, it offers a managed platform for preparing data, training models, and deploying applications with governance.

Its flagship product combines warehouse-style SQL analytics with flexible lake storage so teams can query structured tables and semi-structured files together. The company also maintains an open-source table format focused on reliable incremental processing for analytics and ML."""


qna = dspy.Predict("context: str, question: str -> answer: str, answer_confidence_score: int")
result = qna(context=context, question="what is the data lakehouse")
print(result)
print(result.answer_confidence_score/10) #This is testing that DSPy is enforcing a type
"""
Expected Output:
Prediction(
    answer='...a concise answer grounded in the sample context above...',
    answer_confidence_score=100
)
"""


#Answer Key
qna = dspy.Predict("context: str, question: str -> answer: str, answer_confidence_score: int")
result = qna(context=context, question="what is the data lakehouse")
print(result)
print(result.answer_confidence_score/10)


# Section 2: DSPy Class Signatures

# As you may have started to notice, programming or doing development like this is great for quick testing and iterative workloads. 
# However, this would be unacceptable for a full production ready application. Inline signatures are too restrictive to use.

# As we begin to integrate more complex logic like function calling, multi-modal activities and external depedencies, 
# we will need to begin using DSPy Class Signatures to give us the control and flexibility we need.

# DSPy Signatures more closely resembles data modeling patterns and practices. Here's a breakdown of how the signature allows you to do this:
# 1. **Schema Definition and Type Safety**: You specifically define field names and data types like int, bool, etc. It can be ANY type from typing or pydantic.
# 2. **Field Role Specification**: `InputField()` and `OutputField()` categorize things by their purpose. It's clearly deliminated.
# 3. **Metadata and Constraints**: You can use the keyword argument `desc` within the `InputField()` and `OutputField()` to add additional metadata about what this input or output should be
# 4. **Docstring Documentation**: The docstring encourages good documentation as its also used in the prompt
# 5. **Data Contract**: The signature clearly defines what inputs are expected and what outputs will be provided.
# 6. **Object Orientated Programming**: Class based signatures enable object orientated approaches improving maintainability, reusability and type safey

# DSPy converts EVERYTHING within a DSPy Class Signature so we can use every part of the class to influence how we want the LLM to accomplish our task.

# Let's convert our existing Inline Signatures to a Class Signature. Below I walk through an example with the Math Inline Signature


# This line of code below is what we are converting
# math_problem = dspy.Predict("input_number: int, math_question: str -> answer: int, math_reasoning: str, correct: bool") 

class math_problem(dspy.Signature): #first, we must define a class with dspy.Signature
  """This is to solve math problems asked by the user""" #we can use the docstring to add additional instructions. DSPy will incorporate this into the final prompt sent to the LLM 
  input_number: int = dspy.InputField() #now we need to assign dspy.InputField() or dspy.OutputField() to tell DSPy what is an input and what is an output variable 
  math_question: str = dspy.InputField() 
  answer: int = dspy.OutputField()
  math_reasoning: dict = dspy.OutputField(desc="{'text': 'must be 5 sentences'}") #You can add what DSPy calls "hints" through a kwarg called desc. This further helps guide the outputs in case you aren't seeing outputs you're looking for  
  correct: bool = dspy.OutputField()


#To use the signature, we pass it in like we would an Inline Signature
math_questions = dspy.Predict(math_problem)
result = math_questions(input_number=1, math_question="add 100")
print(result) #Notice how, thanks to the desc kwarg, the math_reasoning answer is now 5 sentences long. 


# Try it yourself! Convert your QnA Inline signature to a DSPy Class Signature
print("Section 2 Quiz: Convert your QnA Inline signature to a DSPy Class Signature")

class qna_signature(dspy.Signature):
  """this is to answer questions"""
  question: str = dspy.InputField()
  answer: str = dspy.OutputField()
    
qna = dspy.Predict(qna_signature)
result = qna(question="hi there")
print(result)


# Great!

# You can see the same dspy.Prediction is returned so you can access the outputs the same way you would an inline signature


# ###Why use class signatures?

# You just defined a DSPy class signature Python object. This object can be passed into any DSPy module, including custom ones you build (more on this later), 
# allowing you to take advantage of the standard behaviors of a Python object.

# Inline signatures is essentially just a string. You would need to copy and paste this each time you run a DSPy module and is difficult to maintain. 
# This defeats the purpose of compiling prompts with DSPy and would be no different from using other frameworks reliant on you providing a prompt. With class signatures, you can take that object and pass it programatically. This gives you a way to reuse your "prompts" in the form of the DSPy class signature while utilizing Type Safety and Validation.

# Now, let's introduce other DSPy modules.


# ###Wait what's a module?
# You've heard me refer to a dspy module and you've been using dspy.Predict to run basic inference on LLMs.

# `dspy.Module` is DSPy's core building block - think of it like a class in object-oriented programming that lets you create reusable components for AI tasks. 
# You saw this in action when we created a class for the signature.

# A DSPy module encapsulates the logic for a specific AI task (like answering questions or summarizing text) and can be composed with other modules to build complex AI systems. 
# The key insight is that instead of writing static prompts, you define the structure and behavior of your AI system, then let DSPy automatically optimize the prompts and model interactions.

# Here are the most commonly used DSPy modules:

# 1. **dspy.Predict** - The most basic module that takes a signature (input/output specification) and generates predictions from a language model.
# 2. **dspy.ChainOfThought** - Extends Predict to make the model show its reasoning process before giving the final answer. At the cost of speed, you can find more performance from your LLMs, especially smaller LLMs
# 3. **dspy.ReAct** - Implements the ReAct pattern where the model alternates between reasoning about what to do and taking actions (like calling tools or APIs).

# With DSPy Modules, you can combine them into larger modules to create sophisticated AI systems. You can even create custom modules (which we will later) which simply inherits the dspy.Module class

# Let's see what it takes to switch our code to use a different module. I will again use the math signature:


math_questions = dspy.ChainOfThought(math_problem) #Chain of Thought reasoning adds some reasoning steps to the LLM even if it was not trainined to be a reasoning model like Deepseek R1 or OpenAI o3. 
result = math_questions(input_number=1, math_question="add 100")
print(result.answer)


# ## Iterative Development with DSPy
# With DSPy, you can quickly do iterative development using these tools. You can switch the model, tweak the signature, change the module or make your own module to find the right combination for your use case.

# Let's iterate through an example with our Math Signature. We are going to try get the LLM to solve a more difficult math problem.

# Let's see how well different models do with the new math problem, then see how we can try a difffernent module to improve performance.

print("Using Gemini 2.5 Flash-Lite")
llm = gemini_llm("gemini/gemini-2.5-flash-lite")

dspy.configure(lm=llm)
math_questions = dspy.Predict(math_problem)
result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
print(result.answer) #answer is 5071 


# Yikes, not great from Flash-Lite. Let's try Pro, then 2.0 Flash



print("Using Gemini 2.5 Pro")
llm = gemini_llm("gemini/gemini-2.5-pro")
# llm = gemini_llm("gemini/gemini-2.0-flash")
dspy.configure(lm=llm)
math_questions = dspy.Predict(math_problem)
result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
print(result.answer) #answer is 5071 


# Still not right...


print("Using Gemini 2.0 Flash")
llm = gemini_llm("gemini/gemini-2.0-flash")
dspy.configure(lm=llm)
math_questions = dspy.Predict(math_problem)
result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
print(result.answer) #answer is 5071 


# ### Switching LLMs

# Since DSPy takes a modular approach, we are able to switch LLMs per signature. We will do this to demonstrate our example below and test multiple models in one cell. `dspy.Context` will allow you to use a different LLM temporarily. We can use this to more quickly test LLMs.

# There are other modules like dspy.BestOfN or dspy.Parallel that you can use to do more testing. You can explore more of these modules at dspy.ai.


#Starting with 8B
with dspy.context(lm=gemini_llm("gemini/gemini-2.5-flash-lite")):
  math_questions = dspy.Predict(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.5 Flash-Lite: {result.answer}") #answer is 5071 

#Switching to 70B
with dspy.context(lm=gemini_llm("gemini/gemini-2.5-pro")):
  math_questions = dspy.Predict(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.5 Pro: {result.answer}") #answer is 5071 

#Switching to Claude
with dspy.context(lm=gemini_llm("gemini/gemini-2.0-flash")):
  math_questions = dspy.Predict(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.0 Flash: {result.answer}") #answer is 5071 

with dspy.context(lm=gemini_llm("gemini/gemini-2.5-flash")):
  math_questions = dspy.Predict(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.5 Flash: {result.answer}") #answer is 5071 


# So we saw not even the best models can do this math problem on its own.

# Could we use `dspy.ChainofThought` to get to the right answer? This module essentially "extends" the model's thinking to improve accuracy


#Starting with 8B
with dspy.context(lm=gemini_llm("gemini/gemini-2.5-flash-lite")):
  math_questions = dspy.ChainOfThought(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.5 Flash-Lite: {result.answer}") #answer is 5071 

#Switching to 70B
with dspy.context(lm=gemini_llm("gemini/gemini-2.5-pro")):
  math_questions = dspy.ChainOfThought(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.5 Pro: {result.answer}") #answer is 5071 

#Switching to Claude
with dspy.context(lm=gemini_llm("gemini/gemini-2.0-flash")):
  math_questions = dspy.ChainOfThought(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.0 Flash: {result.answer}") #answer is 5071 

with dspy.context(lm=gemini_llm("gemini/gemini-2.5-flash")):
  math_questions = dspy.ChainOfThought(math_problem)
  result = math_questions(input_number=1, math_question="add 100, multiply by 100 and then divide by 2, add 21")
  print(f"Output for Gemini 2.5 Flash: {result.answer}") #answer is 5071 


# Yep! Instead of iterating through multiple long text prompts, all we needed to do was a quick line change to iterate through our code to find a more accurate solution. In fact, if Llama-70B is as performant as Claude, we should select Llama-70B for this particular use case since Llama-70b is significantly cheaper than Claude.

# We can see Llama-8B is too weak to do a complex math use case we know we can rule this model out or we will need to do a little more work to get this model working

# You can do this with any model through any provider, including local models on device hosted with Ollama


# Section 3: Advanced Typing for your Signatures

# The typing you provide in your signatures is not limited to the ones available in the Typing library or as simple as just providng a type. Because DSPy signatures are fundamentally built on Pydantic's BaseModel, you can create your own typing using Pydantic BaseModel and pass that in to do data validation or more strict data validation.

# More simply, you can use typing like Literal to enforce certain outputs. The world is your oyster on what you want to define and enforce via typing!

# Below are some examples to help inspire you!


#Classification Example 
from typing import Literal

class Emotion(dspy.Signature):
    """Classify emotion."""

    sentence: str = dspy.InputField()
    sentiment: Literal['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] = dspy.OutputField() # While I specifically defined a list of values here for DSPy to use, you can always say Literal[str] to dynamically pass in a list for DSPy to use

sentence = "weather is INCREDIBLY hot today"

classify = dspy.Predict(Emotion)
classify(sentence=sentence)


# Example using Pydantic BaseModel. You can use them in inline or class based signatures


#Use Pydantic BaseModel
from pydantic import BaseModel

class QueryResult(BaseModel):
    text: str 
    score: float

signature = dspy.Signature("query: str -> result: QueryResult")

example = dspy.Predict(signature)
example(query="What's up")


#Multi-Modal Capabilities
image_url = "https://d323w7klwy72q3.cloudfront.net/i/a/2025/20250813ve/EO4522.JPG"

#Ensure the model you're using is able to receive and use multi-modal inputs
llm = gemini_llm("gemini/gemini-2.0-flash")
# llm = gemini_llm("gemini/gemini-2.5-pro")

dspy.configure(lm=llm)

class image_processor(dspy.Signature):
  """Describe the image"""
  image: dspy.Image = dspy.InputField() #we use the dspy.Image object here as a type
  response: str = dspy.OutputField()

image_assistant = dspy.Predict(image_processor)
result = image_assistant(image=image_url) #dspy.Image has a number of helper functions to convert an image to a dspy.Image. Some other examples include from_file, from_PIL and so on but it handles it for you
print(result.response)


# Workshop: Create a complex signature

# Using everything we learned in the last 3 sections, this workshop will have you create an unstructured to structured signature that takes a long arxiv paper, analyzes it and gives us outputs for us to use downstream.

# Requirements:
# 1. An output with all emails of the authors provided in an output as a list. In the list, the authors should be organized in a custom pydantic base model that contains their name and their emails
# 2. One input being the arxiv paper itself
# 3. One more input being a user's question the model should answer in an output called response. This should be an Optional input
# 4. An output summarizing the paper at a 5th grade level
# 5. An output, rating the market impact of the paper on a scale of 1 to 10. So the output should be integer
# 6. An output, that is responding to the user's question if provided.

# We will test how well the LLM can find specific metrics by asking questions like "what's the WSJ 23 F1 score for Zhu et al"


from typing import Optional
from pydantic import BaseModel

class AuthorDetails(BaseModel):
    name: str
    email: str

# llm = gemini_llm("gemini/gemini-2.5-flash-lite")
llm = gemini_llm("gemini/gemini-2.5-pro")
# llm = gemini_llm("gemini/gemini-2.0-flash")
dspy.configure(lm=llm)

class arxiv_analyzer(dspy.Signature): 
  """Review and provide an analysis of an arxiv paper"""
  arxiv_paper: str = dspy.InputField()
  user_question: Optional[str] = dspy.InputField()
  author_emails: list[AuthorDetails] = dspy.OutputField()
  arxiv_paper_summary: str = dspy.OutputField()
  arxiv_market_impact: int = dspy.OutputField(desc="from 1 to 10")
  user_question_response: Optional[str] = dspy.OutputField(desc="answer if requested by user, there can be multiple")

analyzer = dspy.ChainOfThought(arxiv_analyzer)
result = analyzer(arxiv_paper=arxiv_paper, user_question="what's the WSJ 23 F1 score for Zhu et al")
print(result)


# --- Answer Key
#answer key
from typing import Optional
from pydantic import BaseModel

class AuthorDetails(BaseModel):
    name: str
    email: str

# llm = gemini_llm("gemini/gemini-2.5-flash-lite")
# llm = gemini_llm("gemini/gemini-2.5-pro")
llm = gemini_llm("gemini/gemini-2.0-flash")
dspy.configure(lm=llm)

class arxiv_analyzer(dspy.Signature): 
  """Review and provide an analysis of an arxiv paper"""
  arxiv_paper: str = dspy.InputField() 
  user_question: Optional[str] = dspy.InputField()
  author_emails: list[AuthorDetails] = dspy.OutputField() 
  arxiv_paper_summary: str = dspy.OutputField() 
  arxiv_market_impact: int = dspy.OutputField(desc="from 1 to 10")
  user_question_response: Optional[str] = dspy.OutputField(desc="answer if requested by user, there can be multiple")

analyzer = dspy.ChainOfThought(arxiv_analyzer)
result = analyzer(arxiv_paper=arxiv_paper, user_question="what's the WSJ 23 F1 score for Zhu et al")


print(result)


