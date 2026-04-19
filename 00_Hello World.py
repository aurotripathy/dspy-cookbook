import os, dspy
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
lm = dspy.LM("gemini/gemini-2.0-flash", api_key=os.environ["GOOGLE_API_KEY"])
dspy.configure(lm=lm)
out = dspy.Predict("question -> answer")(question="Say hello in one short line in spanish.")
print(out.answer)
