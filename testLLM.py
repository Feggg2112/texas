from dotenv import load_dotenv
load_dotenv()

from nodes.ai_nodes import _get_llm

llm = _get_llm()
print(llm.invoke("你好"))
