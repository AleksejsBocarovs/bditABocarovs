import time
from pathlib import Path
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.chat_models import ChatLlamaCpp
import re

basePath = Path("S:/modding/mo2/SPE/overwrite/LLM")
inputFile = basePath / "input.txt"
outputFile = basePath / "output.txt"
characterNameFile = basePath / "character.txt"

def ChangeSPronouns(text,character):
	SPronounce = [
		(r"\bYou\b", f"{character}"),
		(r"\byou\b", f"{character}"),
		(r"\bYours\b", f"{character}"),
		(r"\bYours\b", f"{character}"),
		(r"\bYourself\b", f"{character}"),
		(r"\byourself\b", f"{character}")]
	rephrasedInput = text
	for pronounce, name in SPronounce:
		rephrasedInput = re.sub(pronounce, name, rephrasedInput)
	return rephrasedInput

def StartFreeTalk(character):
    systemPrompt = (Path("Characters") / (character + ".txt")).read_text()
    if not ((Path("Memory") / (character + ".txt")).exists()):
        (Path("Memory") / (character + ".txt")).write_text("")
    longTermMemory = (Path("Memory") / (character + ".txt")).read_text()
    systemPrompt += f"\nConversation long-term memory:{longTermMemory}"
    llm = ChatLlamaCpp(
        model_path="S:/LLM/BDllm/models/gemma-3-4b-it.Q4_K_M.gguf",
        n_ctx=3072,
        n_gpu_layers=-1,
        n_threads=16,
        temperature=0,
        top_k=3,
        stop=["Player:"],
        verbose=False)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    retriever = FAISS.load_local("rag_db" + "\\" + character, embeddings,
                                 allow_dangerous_deserialization = True).as_retriever(search_kwargs = {"k": 3})
    summarizePrompt = PromptTemplate.from_template(
        """Update the long-term memory based on the new conversation between the user and the {character}.
		This memory will be used to preserve facts and relationships from previous dialogues.
		Include:
		- Key facts about the user and {character},
		- Important statements or emotions they expressed,
		- How the {character} perceives the user.
		Avoid fact duplication. Rephrase and condense where needed.
		Use neutral phrasing.
		Pay attention to who is speaking and don't confuse facts between characters.
		Do not speak from the character’s perspective.
		Do not confuse the user with the {character}.
		Only extract clean facts.
		Avoid repeating the same fact, even if it appears in both old and new conversation.

		Previous memory:
		{previousMemory}

		New conversation:
		{conversation}

		Updated long-term memory:"""
    )
    chatRagPrompt = ChatPromptTemplate.from_messages([
        ("system", systemPrompt + "\nRelevant knowledge:\n{context}"),
        MessagesPlaceholder(variable_name = "chatHistory"),
        ("user", "{input}")
    ])
    documentChain = create_stuff_documents_chain(
        llm = llm,
        prompt = chatRagPrompt
    )
    chatHistory = []
    while True:
        if inputFile.exists():
            with open(inputFile, "r", encoding="utf-8") as f:
                uInput = f.read().strip()
            inputFile.unlink()
            if uInput.lower() == "exit" or uInput == "":
                break
            rInput = ChangeSPronouns(uInput,character)
            relevantChunks = retriever.invoke(rInput)
            response = documentChain.invoke({
                "input": uInput,
                "context": relevantChunks,
                "chatHistory": chatHistory
            })
            with open(outputFile, "w", encoding="utf-8") as f:
                f.write(response)
            chatHistory.append({"role": "user", "content": uInput})
            chatHistory.append({"role": "ai", "content": response})
        time.sleep(0.5)
        if outputFile.exists():
            with outputFile.open("r", encoding="utf-8") as f:
                output = f.read()
            if "S.H.O.W.E.D" in output:
                outputFile.unlink()
    conversationHistory = "\n".join(f"{'User' if text['role'] == 'user' else 'Character'}: {text['content']}" for text in chatHistory)
    memory = summarizePrompt.format(character = character,
                                    previousMemory = longTermMemory,
                                    conversation = conversationHistory)
    summary = llm.invoke(memory)
    (Path("Memory") / (character + ".txt")).write_text(summary.content.strip(), encoding = "utf-8")
while True:
    if characterNameFile.exists():
        with characterNameFile.open("r", encoding="utf-8") as f:
            characterName = f.read().strip()
        characterNameFile.unlink()
        StartFreeTalk(characterName)
    else :
        time.sleep(0.5)